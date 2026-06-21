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
        # Live metrics.
        self._event_count = 0

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
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="SOGNO_CANE-engine", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        if not self._running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._running = False
        # Flush any pending note_off events.
        for ev in self._mapper.shutdown():
            self._dispatch_event(ev)

    # ------------------------------------------------------------------ #
    # Loop                                                                #
    # ------------------------------------------------------------------ #

    def _packet_period(self) -> float:
        return (
            self._source.packet_samples / float(self._source.sample_rate_hz)
        ) * self._time_scale

    def _run_loop(self) -> None:
        src = self._source
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
            period = self._packet_period()
            next_t += period
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                self._stop_event.wait(timeout=sleep_for)
            else:
                # We are behind; do not accumulate negative slack.
                next_t = time.monotonic()

    def _dispatch_event(self, ev: MappingEvent) -> None:
        self._event_count += 1
        if self._on_event is not None:
            try:
                self._on_event(ev)
            except Exception:
                pass
        if ev.kind == "note_on":
            self._midi.send_note_on(ev.note, ev.velocity, ev.channel)
        elif ev.kind == "note_off":
            self._midi.send_note_off(ev.note, ev.channel)
        elif ev.kind == "cc":
            self._midi.send_control_change(ev.control, ev.value, ev.channel)
        elif ev.kind == "pitchbend":
            self._midi.send_pitch_bend(ev.pitch, ev.channel)
