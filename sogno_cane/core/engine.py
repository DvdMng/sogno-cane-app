"""Realtime engine: clocks a packet source and forwards events to MIDI.

The engine is source-agnostic: it clocks any :class:`PacketSource` — the
synthetic :class:`EEGSimulator` by default, or an
:class:`~sogno_cane.core.sources.ArrayPlaybackSource` when playing back a
recording loaded from disk.

This module deliberately does NOT depend on Qt so it can be exercised by
``pytest`` without a display server. The Qt wrapper in :mod:`sogno_cane.ui`
re-uses this engine.
"""
from __future__ import annotations

import heapq
import itertools
import threading
import time
from typing import Callable, Optional

from sogno_cane.core.sources import PacketSource
from sogno_cane.eeg.profiles import DeviceProfile, HUMAN_PROFILE
from sogno_cane.eeg.simulator import (
    DEFAULT_PACKET_SAMPLES,
    EEGSimulator,
)
from sogno_cane.eeg.unicorn_packet import UnicornPacket
from sogno_cane.midi.mapper import MappingConfig, MappingEngine, MappingEvent
from sogno_cane.midi.output import MidiOutput


class NoteOffScheduler:
    """Send timed note_off events with millisecond precision.

    Strategies run on the 100 ms analysis hop, far too coarse for the short
    "trigger" notes (~15 ms) needed to drive DAW envelopes. When a note_on
    carries a ``duration``, the engine hands the matching note_off to this
    scheduler, which runs its own daemon thread on the wall clock.

    The worker thread is started lazily by :meth:`schedule` whenever no live
    worker exists, so idle engines (and unit tests) cost nothing — and a
    scheduler stopped by :meth:`stop` transparently revives if new work
    arrives (RealtimeEngine supports start() after stop()).
    """

    def __init__(self, send: Callable[[MappingEvent], None]) -> None:
        self._send = send
        # Heap of (due_monotonic, seq, channel, note).
        self._heap: list[tuple[float, int, int, int]] = []
        self._seq = itertools.count()
        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._stopped = False
        # Set by the worker (under the lock) the moment it decides to exit;
        # tested by schedule() so a new worker is spawned exactly when the
        # old one is gone or irrevocably on its way out.
        self._worker_exited = True

    def schedule(self, delay_seconds: float, channel: int, note: int) -> None:
        due = time.monotonic() + max(0.001, float(delay_seconds))
        with self._cond:
            # Revive after stop(): RealtimeEngine supports start() after
            # stop() (and stop() is also used to flush at source exhaustion),
            # so a stopped scheduler must accept new work again — silently
            # dropping it would leave the note stuck (the strategies do not
            # emit their own note_off for gated notes).
            if self._stopped:
                self._stopped = False
            # Supersede any pending release of the SAME (channel, note): a
            # restrike retriggers the sounding voice, so only the newest
            # note_off may fire — the stale one would truncate the new note.
            survivors = [
                e for e in self._heap
                if not (e[2] == channel and e[3] == note)
            ]
            if len(survivors) != len(self._heap):
                self._heap = survivors
                heapq.heapify(self._heap)
            heapq.heappush(self._heap, (due, next(self._seq), channel, note))
            if self._worker_exited:
                self._worker_exited = False
                self._thread = threading.Thread(
                    target=self._run, name="SOGNO_CANE-noteoff", daemon=True,
                )
                self._thread.start()
            self._cond.notify()

    def flush(self) -> None:
        """Send every pending note_off immediately."""
        with self._cond:
            pending = [(ch, n) for (_t, _s, ch, n) in self._heap]
            self._heap.clear()
        for ch, n in pending:
            self._emit(ch, n)

    def stop(self, flush: bool = True) -> None:
        """Stop accepting work; optionally flush what is still pending."""
        with self._cond:
            self._stopped = True
            self._cond.notify()
        if flush:
            self.flush()

    # ------------------------------------------------------------------ #

    def _emit(self, channel: int, note: int) -> None:
        try:
            self._send(MappingEvent(
                kind="note_off", channel=channel, note=note,
            ))
        except Exception:
            pass

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._stopped and (
                    not self._heap
                    or self._heap[0][0] > time.monotonic()
                ):
                    if self._heap:
                        timeout = self._heap[0][0] - time.monotonic()
                        self._cond.wait(timeout=max(0.0005, timeout))
                    else:
                        self._cond.wait()
                if self._stopped:
                    # Declared under the lock BEFORE releasing it, so
                    # schedule() knows deterministically to spawn a
                    # replacement worker.
                    self._worker_exited = True
                    return
                _due, _seq, channel, note = heapq.heappop(self._heap)
            self._emit(channel, note)


class RealtimeEngine:
    """Generate EEG packets at the device cadence and dispatch MIDI events.

    Parameters
    ----------
    profile
        EEG device profile (human or dog). Used to build the default
        simulator and the default mapping sample rate.
    packet_samples
        Samples per packet (default 25 = 100 ms cadence at 250 Hz).
    seed
        Optional RNG seed for reproducible runs.
    mapping_config
        Mapping engine settings (window length, hop).
    source
        Optional packet source. If omitted, a fresh :class:`EEGSimulator`
        built from ``profile`` is used.
    on_packet
        Optional callback ``fn(packet)`` invoked on every produced packet.
    on_event
        Optional callback ``fn(event)`` invoked for every MIDI event before
        it is forwarded to the MIDI output (useful for the UI monitor).
    on_finished
        Optional callback invoked once when a non-looping source is exhausted.
    """

    def __init__(
        self,
        profile: DeviceProfile = HUMAN_PROFILE,
        packet_samples: int = DEFAULT_PACKET_SAMPLES,
        seed: Optional[int] = None,
        mapping_config: Optional[MappingConfig] = None,
        source: Optional[PacketSource] = None,
        on_packet: Optional[Callable[[UnicornPacket], None]] = None,
        on_event: Optional[Callable[[MappingEvent], None]] = None,
        on_finished: Optional[Callable[[], None]] = None,
        midi: Optional[MidiOutput] = None,
    ) -> None:
        if source is None:
            source = EEGSimulator(
                profile=profile, packet_samples=packet_samples, seed=seed,
            )
        self._source = source
        cfg = mapping_config or MappingConfig(
            sample_rate_hz=int(getattr(source, "sample_rate_hz",
                                       profile.sample_rate_hz))
        )
        self._mapper = MappingEngine(cfg)
        self._midi = midi if midi is not None else MidiOutput()
        self._on_packet = on_packet
        self._on_event = on_event
        self._on_finished = on_finished

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        # Slow-down / speed-up factor: 1.0 = realtime; <1 faster; >1 slower.
        self._time_scale: float = 1.0
        # Live metrics. _dispatch_event runs on both the engine thread and
        # the note-off scheduler thread, so the counter needs a lock.
        self._event_count = 0
        self._count_lock = threading.Lock()
        # Wall-clock scheduler for gated (fixed-duration) notes.
        self._noteoff = NoteOffScheduler(self._dispatch_event)

    # ------------------------------------------------------------------ #
    # Public accessors                                                   #
    # ------------------------------------------------------------------ #

    @property
    def source(self) -> PacketSource:
        return self._source

    @property
    def simulator(self) -> Optional[EEGSimulator]:
        return self._source if isinstance(self._source, EEGSimulator) else None

    @property
    def mapper(self) -> MappingEngine:
        return self._mapper

    @property
    def midi(self) -> MidiOutput:
        return self._midi

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def event_count(self) -> int:
        return self._event_count

    def set_time_scale(self, factor: float) -> None:
        if factor <= 0.0:
            raise ValueError("time_scale must be > 0")
        self._time_scale = float(factor)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._running:
            return
        # Strategies are shared live objects that may carry gate/throttle
        # timestamps from a previous engine (whose clock ran far ahead of
        # this one, which starts at zero). Clear them or voices stay mute
        # for up to the previous run's duration.
        self._mapper.reset_strategy_state()
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="SOGNO_CANE-engine", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        if self._running:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=timeout)
            self._running = False
            # Flush any pending note_off events.
            for ev in self._mapper.shutdown():
                self._dispatch_event(ev)
        # Always release gated notes, even if the source already exhausted
        # itself (in that case _running is False but short notes may still
        # be pending on the wall-clock scheduler).
        self._noteoff.stop(flush=True)

    # ------------------------------------------------------------------ #
    # Loop                                                                #
    # ------------------------------------------------------------------ #

    def _packet_period(self) -> float:
        return (
            self._source.packet_samples / float(self._source.sample_rate_hz)
        ) * self._time_scale

    def _run_loop(self) -> None:
        src = self._source
        # A live headset is paced by its own clock: its source blocks until
        # samples arrive. Re-timing it here on the PC clock would cut the
        # stream into lumps and, since the two clocks never agree exactly,
        # let a backlog (latency) build up.
        self_clocked = bool(getattr(src, "self_clocked", False))
        next_t = time.monotonic()
        while not self._stop_event.is_set():
            packet = src.next_packet()
            if packet is None:
                # Source exhausted (non-looping playback). Flush any notes
                # the strategies are still holding BEFORE we drop out, or
                # they hang on the synth (stop() can't help once _running
                # is False).
                self._running = False
                try:
                    for ev in self._mapper.shutdown():
                        self._dispatch_event(ev)
                except Exception:
                    pass
                # stop (not just flush): parks the worker thread so a
                # finished playback does not leak it. The scheduler revives
                # automatically if this engine is ever started again.
                self._noteoff.stop(flush=True)
                if self._on_finished is not None:
                    try:
                        self._on_finished()
                    except Exception:
                        pass
                return
            if self._on_packet is not None:
                try:
                    self._on_packet(packet)
                except Exception:
                    pass
            for ev in self._mapper.process_packet(packet):
                self._dispatch_event(ev)
            if self_clocked:
                continue
            period = self._packet_period()
            next_t += period
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                self._stop_event.wait(timeout=sleep_for)
            else:
                # We are behind; do not accumulate negative slack.
                next_t = time.monotonic()

    def _dispatch_event(self, ev: MappingEvent) -> None:
        with self._count_lock:
            self._event_count += 1
        if self._on_event is not None:
            try:
                self._on_event(ev)
            except Exception:
                pass
        if ev.kind == "note_on":
            self._midi.send_note_on(ev.note, ev.velocity, ev.channel)
            if ev.duration > 0.0:
                # Gated note: the engine owns the release. Scale with the
                # playback speed so slowed-down playback stretches notes too.
                self._noteoff.schedule(
                    ev.duration * self._time_scale, ev.channel, ev.note,
                )
        elif ev.kind == "note_off":
            self._midi.send_note_off(ev.note, ev.channel)
        elif ev.kind == "cc":
            self._midi.send_control_change(ev.control, ev.value, ev.channel)
        elif ev.kind == "pitchbend":
            self._midi.send_pitch_bend(ev.pitch, ev.channel)
