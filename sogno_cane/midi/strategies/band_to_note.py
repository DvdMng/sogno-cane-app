"""Band power -> continuous note within a configurable scale.

The mean log-power of the chosen band across the channel set is mapped to a
0..1 value with adjustable gain/floor (or a self-adapting AGC window) and
quantized to a note inside the target scale. Notes change only when the
quantized index moves *and* the voice's pacing allows it, giving a musical
(rather than glitchy) output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sogno_cane.eeg.bands import BandPowers
from sogno_cane.midi.context import WindowContext
from sogno_cane.midi.mapper import MappingEvent
from sogno_cane.midi.pacing import AdaptiveNorm, NoteGate
from sogno_cane.midi.scales import (
    MidiNoteRange,
    build_scale,
    quantize_to_scale,
)


@dataclass
class BandToNoteStrategy:
    """Continuous band-power -> melodic note in a scale.

    Attributes
    ----------
    band
        Source band, e.g. ``"alpha"``.
    channels
        EEG channel indices to average. ``None`` means all channels.
    scale, root
        Scale name and root.
    note_range
        Inclusive MIDI note range to populate.
    velocity_min, velocity_max
        Velocity range driven by the same band power.
    adaptive
        If True, ignore log_floor/ceiling and self-calibrate the mapping.
    log_floor_uv2, log_ceiling_uv2
        Fixed log10(power) bounds used when ``adaptive`` is False.
    sustain
        Hold a note until a different scale index is selected (vs retrigger).
    min_interval_seconds, min_duration_seconds, change_threshold_norm
        Musical pacing, honoured live.
    channel
        MIDI channel (0..15).
    """

    band: str = "alpha"
    channels: tuple[int, ...] | None = None
    scale: str = "minor_pentatonic"
    root: str = "A"
    note_range: MidiNoteRange = field(default_factory=MidiNoteRange)
    velocity_min: int = 40
    velocity_max: int = 115
    adaptive: bool = True
    log_floor_uv2: float = 0.0    # log10(power) floor
    log_ceiling_uv2: float = 3.0  # log10(power) ceiling
    sustain: bool = True
    min_interval_seconds: float = 1.0
    min_duration_seconds: float = 0.4
    change_threshold_norm: float = 0.12
    channel: int = 0
    name: str = "BandToNote"
    enabled: bool = True

    _last_note: int | None = field(default=None, init=False, repr=False)
    _gate: NoteGate = field(default_factory=NoteGate, init=False, repr=False)
    _norm: AdaptiveNorm = field(
        default_factory=AdaptiveNorm, init=False, repr=False
    )

    def on_window(
        self,
        eeg_window: np.ndarray,
        bands: BandPowers,
        ctx: WindowContext,
    ) -> list[MappingEvent]:
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

        log_p = float(np.log10(max(mean_pow, 1e-6)))
        if self.adaptive:
            norm = self._norm.normalize(log_p)
        else:
            span = self.log_ceiling_uv2 - self.log_floor_uv2 + 1e-9
            norm = float(np.clip((log_p - self.log_floor_uv2) / span, 0.0, 1.0))

        scale_notes = build_scale(self.scale, self.root, self.note_range)
        if not scale_notes:
            return []

        t = ctx.t_seconds

        # Below floor = silence: turn off current note (subject to min hold).
        if norm <= 0.02:
            if self._last_note is not None and self._gate.may_release(
                t, min_hold=self.min_duration_seconds
            ):
                ev = MappingEvent(
                    kind="note_off",
                    channel=self.channel,
                    note=self._last_note,
                )
                self._last_note = None
                self._gate.commit_off()
                return [ev]
            return []

        new_note = quantize_to_scale(norm, scale_notes)
        velocity = int(round(
            self.velocity_min + norm * (self.velocity_max - self.velocity_min)
        ))
        velocity = max(1, min(127, velocity))

        events: list[MappingEvent] = []
        if self.sustain:
            if new_note == self._last_note:
                return events
            allowed = self._gate.may_change(
                t, norm,
                min_interval=self.min_interval_seconds,
                change_threshold=self.change_threshold_norm,
            )
            if self._last_note is not None and not self._gate.may_release(
                t, min_hold=self.min_duration_seconds
            ):
                allowed = False
            if not allowed:
                return events
            if self._last_note is not None:
                events.append(MappingEvent(
                    kind="note_off", channel=self.channel, note=self._last_note,
                ))
            events.append(MappingEvent(
                kind="note_on",
                channel=self.channel,
                note=new_note,
                velocity=velocity,
            ))
            self._last_note = new_note
            self._gate.commit_on(t, norm)
        else:
            if not self._gate.may_change(
                t, norm,
                min_interval=self.min_interval_seconds,
                change_threshold=0.0,
            ):
                return events
            if self._last_note is not None:
                events.append(MappingEvent(
                    kind="note_off", channel=self.channel, note=self._last_note,
                ))
            events.append(MappingEvent(
                kind="note_on",
                channel=self.channel,
                note=new_note,
                velocity=velocity,
            ))
            self._last_note = new_note
            self._gate.commit_on(t, norm)
        return events

    def shutdown(self) -> list[MappingEvent]:
        if self._last_note is None:
            return []
        ev = MappingEvent(
            kind="note_off", channel=self.channel, note=self._last_note,
        )
        self._last_note = None
        self._gate.commit_off()
        return [ev]
