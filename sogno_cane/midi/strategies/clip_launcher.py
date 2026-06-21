"""Threshold-based clip launcher.

Each rule is an ``(eeg_channel, band) -> midi note`` mapping with a
threshold and a minimum re-trigger interval (in seconds). When the band
power on the chosen channel crosses upward, a single short MIDI note is
fired on the configured channel. In Ableton, mapping the MIDI Note Learn of
a clip slot to that note will launch the clip.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sogno_cane.eeg.bands import BandPowers
from sogno_cane.midi.context import WindowContext
from sogno_cane.midi.mapper import MappingEvent


@dataclass
class ClipRule:
    eeg_channel: int
    band: str
    midi_note: int
    midi_channel: int = 0
    threshold_uv2: float = 30.0
    release_ratio: float = 0.5
    min_interval_seconds: float = 0.5   # re-trigger lockout
    velocity: int = 110
    enabled: bool = True


@dataclass
class ClipLauncherStrategy:
    rules: list[ClipRule] = field(default_factory=list)
    channel: int = 0   # unused; per-rule channel used instead
    name: str = "ClipLauncher"
    enabled: bool = True

    _armed: dict[int, bool] = field(
        default_factory=dict, init=False, repr=False,
    )
    _last_fire_t: dict[int, float] = field(
        default_factory=dict, init=False, repr=False,
    )
    _held: list[tuple[int, int]] = field(
        default_factory=list, init=False, repr=False,
    )

    def on_window(
        self,
        eeg_window: np.ndarray,
        bands: BandPowers,
        ctx: WindowContext,
    ) -> list[MappingEvent]:
        events: list[MappingEvent] = []
        t = ctx.t_seconds

        # Release any notes held from the previous window (short trigger).
        for ch, note in self._held:
            events.append(MappingEvent(
                kind="note_off", channel=ch, note=note,
            ))
        self._held.clear()

        for idx, rule in enumerate(self.rules):
            if not rule.enabled:
                continue
            try:
                arr = bands.by_band[rule.band]
            except KeyError:
                continue
            if not 0 <= rule.eeg_channel < arr.shape[0]:
                continue
            power = float(arr[rule.eeg_channel])

            last_fire = self._last_fire_t.get(idx, -1.0e9)
            locked = (t - last_fire) < max(0.0, rule.min_interval_seconds)
            armed = self._armed.get(idx, True)

            if armed and power >= rule.threshold_uv2 and not locked:
                events.append(MappingEvent(
                    kind="note_on",
                    channel=rule.midi_channel,
                    note=rule.midi_note,
                    velocity=rule.velocity,
                ))
                self._held.append((rule.midi_channel, rule.midi_note))
                self._armed[idx] = False
                self._last_fire_t[idx] = t
            elif (
                not armed
                and power <= rule.threshold_uv2 * rule.release_ratio
            ):
                self._armed[idx] = True
        return events

    def shutdown(self) -> list[MappingEvent]:
        events = [
            MappingEvent(kind="note_off", channel=ch, note=n)
            for (ch, n) in self._held
        ]
        self._held.clear()
        return events
