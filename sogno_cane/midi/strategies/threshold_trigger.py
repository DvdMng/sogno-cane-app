"""Threshold-triggered note events with hysteresis.

When a band's power crosses ``threshold_uv2`` upward, the strategy emits a
note_on at a chosen note (or cycles through a chord/sequence). When it falls
below ``threshold_uv2 * release_ratio``, a note_off is emitted. This avoids
chattering when the power hovers near the threshold. An additional
``min_interval_seconds`` re-arm lockout prevents machine-gun retriggering.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sogno_cane.eeg.bands import BandPowers
from sogno_cane.midi.context import WindowContext
from sogno_cane.midi.mapper import MappingEvent
from sogno_cane.midi.scales import build_scale


@dataclass
class ThresholdTriggerStrategy:
    band: str = "beta"
    channels: tuple[int, ...] | None = None
    threshold_uv2: float = 50.0
    release_ratio: float = 0.6
    velocity_at_threshold: int = 60
    velocity_max: int = 120
    velocity_scale_uv2: float = 200.0
    notes: tuple[int, ...] = (60, 64, 67, 72)  # C major chord by default
    sequence_mode: str = "cycle"  # "cycle" or "random"
    min_interval_seconds: float = 0.25
    channel: int = 1
    name: str = "ThresholdTrigger"
    enabled: bool = True
    seed: int | None = None

    _is_active: bool = field(default=False, init=False, repr=False)
    _active_note: int | None = field(default=None, init=False, repr=False)
    _cycle_index: int = field(default=0, init=False, repr=False)
    _last_trigger_t: float = field(default=-1.0e9, init=False, repr=False)
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(),
        init=False, repr=False,
    )

    def __post_init__(self) -> None:
        if self.seed is not None:
            self._rng = np.random.default_rng(self.seed)

    @classmethod
    def from_scale(
        cls,
        scale: str,
        root: str = "C",
        octave_low: int = 4,
        octave_high: int = 5,
        **kwargs,
    ) -> "ThresholdTriggerStrategy":
        """Convenience constructor: build ``notes`` from a scale + range."""
        from sogno_cane.midi.scales import MidiNoteRange
        rng = MidiNoteRange(
            lo=(octave_low + 1) * 12,
            hi=(octave_high + 1) * 12 + 11,
        )
        notes = build_scale(scale, root, rng)
        return cls(notes=notes, **kwargs)

    def on_window(
        self,
        eeg_window: np.ndarray,
        bands: BandPowers,
        ctx: WindowContext,
    ) -> list[MappingEvent]:
        if not self.notes:
            return []
        try:
            power = bands.by_band[self.band]
        except KeyError:
            return []

        if self.channels is None:
            mean_pow = float(np.mean(power))
        else:
            idx = [c for c in self.channels if 0 <= c < power.shape[0]]
            if not idx:
                return []
            mean_pow = float(np.mean(power[idx]))

        events: list[MappingEvent] = []
        release = self.threshold_uv2 * self.release_ratio
        t = ctx.t_seconds

        if (
            not self._is_active
            and mean_pow >= self.threshold_uv2
            and (t - self._last_trigger_t) >= max(0.0, self.min_interval_seconds)
        ):
            # Trigger.
            if self.sequence_mode == "random":
                note = int(self.notes[self._rng.integers(0, len(self.notes))])
            else:
                note = int(self.notes[self._cycle_index % len(self.notes)])
                self._cycle_index += 1
            velocity = self._velocity_for(mean_pow)
            events.append(MappingEvent(
                kind="note_on",
                channel=self.channel,
                note=note,
                velocity=velocity,
            ))
            self._active_note = note
            self._is_active = True
            self._last_trigger_t = t

        elif self._is_active and mean_pow <= release:
            if self._active_note is not None:
                events.append(MappingEvent(
                    kind="note_off",
                    channel=self.channel,
                    note=self._active_note,
                ))
            self._active_note = None
            self._is_active = False

        return events

    def _velocity_for(self, power: float) -> int:
        if self.velocity_scale_uv2 <= 0:
            return self.velocity_at_threshold
        excess = max(0.0, power - self.threshold_uv2)
        frac = min(1.0, excess / self.velocity_scale_uv2)
        v = self.velocity_at_threshold + frac * (
            self.velocity_max - self.velocity_at_threshold
        )
        return int(max(1, min(127, round(v))))

    def shutdown(self) -> list[MappingEvent]:
        if self._active_note is None:
            return []
        ev = MappingEvent(
            kind="note_off", channel=self.channel, note=self._active_note,
        )
        self._active_note = None
        self._is_active = False
        return [ev]
