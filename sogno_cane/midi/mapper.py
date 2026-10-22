"""EEG -> MIDI translation engine.

The engine accumulates samples in a sliding window, computes band powers, and
forwards them to a pluggable list of :py:class:`MappingStrategy` instances.
Each strategy emits :py:class:`MappingEvent` objects which the engine batches
and hands to a :py:class:`MidiOutput`.

Multiple strategies can run concurrently on different MIDI channels, giving
an arbitrarily wide musical vocabulary.

Strategies receive a deterministic :class:`WindowContext` (simulation time,
hop, sample rate) so they can implement musical pacing reproducibly. For
backward compatibility a strategy may still expose the old two-argument
``on_window(eeg_window, bands)`` signature; the engine detects which form a
strategy accepts and calls it accordingly.
"""
from __future__ import annotations

import inspect
from collections import deque
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from sogno_cane.eeg.bands import BandPowers, compute_band_powers
from sogno_cane.midi.context import WindowContext


# ---------------------------------------------------------------------- #
# Events                                                                 #
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class MappingEvent:
    """A MIDI event ready to be sent.

    ``kind`` is one of ``"note_on"``, ``"note_off"``, ``"cc"``, ``"pitchbend"``.
    Fields irrelevant to the kind are ignored.

    ``duration`` applies to ``note_on`` only: when > 0 the realtime engine
    schedules the matching ``note_off`` automatically after that many seconds
    (wall clock, millisecond precision). 0 means "held" — the strategy itself
    releases the note later (legato behaviour).
    """

    kind: str
    channel: int = 0
    note: int = 0
    velocity: int = 0
    control: int = 0
    value: int = 0
    pitch: int = 0
    duration: float = 0.0


# ---------------------------------------------------------------------- #
# Strategy protocol                                                      #
# ---------------------------------------------------------------------- #


class MappingStrategy(Protocol):
    """A strategy converts band powers / EEG into MIDI events."""

    name: str
    enabled: bool
    channel: int

    def on_window(
        self,
        eeg_window: np.ndarray,
        bands: BandPowers,
        ctx: WindowContext,
    ) -> list[MappingEvent]:
        ...

    def shutdown(self) -> list[MappingEvent]:
        ...


# ---------------------------------------------------------------------- #
# Engine config                                                          #
# ---------------------------------------------------------------------- #


@dataclass
class MappingConfig:
    """Window settings shared by all strategies."""

    sample_rate_hz: int = 250
    window_seconds: float = 1.0
    hop_seconds: float = 0.1     # window evaluation cadence
    # Power-line frequency excluded from every band (Italy: 50 Hz; 0 = off).
    mains_hz: float = 50.0

    @property
    def window_samples(self) -> int:
        return max(8, int(round(self.window_seconds * self.sample_rate_hz)))

    @property
    def hop_samples(self) -> int:
        return max(1, int(round(self.hop_seconds * self.sample_rate_hz)))


# ---------------------------------------------------------------------- #
# Engine                                                                 #
# ---------------------------------------------------------------------- #


class MappingEngine:
    """Sliding-window EEG -> MIDI dispatcher.

    Usage::

        engine = MappingEngine(MappingConfig(sample_rate_hz=250))
        engine.add_strategy(BandToNoteStrategy(...))
        for packet in stream:
            events = engine.process_packet(packet)
            for ev in events:
                ... # send via MidiOutput
    """

    def __init__(self, config: MappingConfig) -> None:
        self._config = config
        self._strategies: list[MappingStrategy] = []
        self._buffer: deque[np.ndarray] = deque()
        self._buffered_samples: int = 0
        self._samples_since_last_window: int = 0
        # Deterministic simulation clock (seconds), advanced by sample count.
        self._t_seconds: float = 0.0
        self._total_samples: int = 0
        # Cache of whether each strategy accepts a context argument.
        self._accepts_ctx: dict[int, bool] = {}
        # Last sample flagged valid, used to bridge samples the hardware
        # marks as lost (VALID = 0).
        self._last_good: Optional[np.ndarray] = None

    @property
    def config(self) -> MappingConfig:
        return self._config

    @property
    def strategies(self) -> list[MappingStrategy]:
        return list(self._strategies)

    @property
    def elapsed_seconds(self) -> float:
        return self._t_seconds

    def add_strategy(self, strategy: MappingStrategy) -> None:
        self._strategies.append(strategy)
        self._accepts_ctx[id(strategy)] = self._strategy_accepts_ctx(strategy)

    def remove_strategy(self, strategy: MappingStrategy) -> None:
        self._strategies.remove(strategy)
        self._accepts_ctx.pop(id(strategy), None)

    def clear_buffer(self) -> None:
        self._buffer.clear()
        self._buffered_samples = 0
        self._samples_since_last_window = 0

    def reset(self) -> None:
        self.clear_buffer()
        self._t_seconds = 0.0
        self._total_samples = 0
        self._last_good = None

    def reset_strategy_state(self) -> None:
        """Clear per-run runtime state on every attached strategy.

        Strategy objects outlive engines: the UI re-adds the same live
        (panel-edited) instances to every new engine, whose simulation
        clock restarts at zero. Without this, gate/throttle timestamps
        from the previous run mute voices for up to the previous session's
        whole duration — heard as "MIDI stops after a stop/start".
        """
        for s in self._strategies:
            reset = getattr(s, "reset_runtime", None)
            if reset is None:
                continue
            try:
                reset()
            except Exception:
                continue

    def shutdown(self) -> list[MappingEvent]:
        events: list[MappingEvent] = []
        for s in self._strategies:
            try:
                events.extend(s.shutdown())
            except Exception:
                continue
        return events

    # ------------------------------------------------------------------ #
    # Core                                                                #
    # ------------------------------------------------------------------ #

    def _bridge_lost_samples(self, packet, eeg: np.ndarray) -> np.ndarray:
        """Hold the last valid sample across samples flagged as lost.

        A Unicorn marks samples hit by Bluetooth data loss with VALID = 0.
        On DC-coupled hardware their content (often zero) next to a real
        sample tens of thousands of microvolts away is a giant step, which
        the spectrum reads as a burst on every band at once — every voice
        would fire together. Sample-and-hold turns a dropout into a flat
        line instead: honest silence, no explosion.
        """
        try:
            valid = np.asarray(packet.validation)
        except Exception:
            return eeg
        if valid.shape[0] != eeg.shape[0]:
            return eeg
        bad = valid < 0.5
        if not bad.any():
            self._last_good = eeg[-1].copy()
            return eeg
        out = eeg.copy()
        last = self._last_good
        for i in range(out.shape[0]):
            if bad[i]:
                if last is not None:
                    out[i] = last
            else:
                last = out[i]
        if last is not None:
            self._last_good = np.array(last, copy=True)
        return out

    def process_packet(self, packet) -> list[MappingEvent]:
        """Push a packet through the engine; return triggered MIDI events."""
        eeg = packet.eeg.astype(np.float64, copy=False)
        eeg = self._bridge_lost_samples(packet, eeg)
        n = eeg.shape[0]
        self._buffer.append(eeg)
        self._buffered_samples += n
        self._samples_since_last_window += n
        self._total_samples += n
        self._t_seconds = self._total_samples / float(
            self._config.sample_rate_hz
        )

        # Trim buffer to at most one window worth.
        win = self._config.window_samples
        while (
            self._buffer
            and self._buffered_samples - self._buffer[0].shape[0] >= win
        ):
            popped = self._buffer.popleft()
            self._buffered_samples -= popped.shape[0]

        events: list[MappingEvent] = []
        if self._buffered_samples < win:
            return events

        # Time to evaluate? Keep the remainder: a live headset delivers
        # bursts of 4, and restarting the count from zero on the burst that
        # crosses the hop would stretch every hop to 28 samples (112 ms) —
        # the music would run 12% slow.
        hop = self._config.hop_samples
        if self._samples_since_last_window < hop:
            return events
        self._samples_since_last_window %= hop

        eeg_window = self._tail(win)
        bands = compute_band_powers(
            eeg_window,
            self._config.sample_rate_hz,
            mains_hz=getattr(self._config, "mains_hz", 50.0),
        )
        ctx = WindowContext(
            t_seconds=self._t_seconds,
            hop_seconds=self._config.hop_seconds,
            sample_rate_hz=self._config.sample_rate_hz,
            window_samples=win,
        )

        for s in self._strategies:
            if not getattr(s, "enabled", True):
                continue
            try:
                if self._accepts_ctx.get(id(s), True):
                    events.extend(s.on_window(eeg_window, bands, ctx))
                else:
                    events.extend(s.on_window(eeg_window, bands))
            except Exception:
                # Never let a single strategy crash the pipeline.
                continue
        return events

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _strategy_accepts_ctx(strategy) -> bool:
        """True if ``on_window`` takes a third (context) positional arg."""
        try:
            sig = inspect.signature(strategy.on_window)
        except (TypeError, ValueError):
            return True
        if any(
            p.kind == inspect.Parameter.VAR_POSITIONAL
            for p in sig.parameters.values()
        ):
            return True
        params = [
            p for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        # bound method already excludes self: (eeg_window, bands[, ctx])
        return len(params) >= 3

    def _tail(self, n_samples: int) -> np.ndarray:
        """Return the last ``n_samples`` rows from the buffer."""
        arrs = list(self._buffer)
        full = np.concatenate(arrs, axis=0)
        return full[-n_samples:]
