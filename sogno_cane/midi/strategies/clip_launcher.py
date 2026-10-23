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
from sogno_cane.midi.pacing import BurstDetector


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
    # Adaptive mode (default): each rule fires when its band rises this many
    # typical deviations above its own recent baseline (see BurstDetector),
    # whatever the amplifier's absolute scale. The fixed 60 uV^2 rule
    # thresholds were tuned on the simulator; a real Unicorn's theta/gamma
    # sit at ~4-10 uV^2 (measured) and almost never launched a clip.
    # False = the absolute per-rule thresholds (old behaviour).
    adaptive: bool = True
    relative_threshold: float = 3.0

    _armed: dict[int, bool] = field(
        default_factory=dict, init=False, repr=False,
    )
    _last_fire_t: dict[int, float] = field(
        default_factory=dict, init=False, repr=False,
    )
    _held: list[tuple[int, int]] = field(
        default_factory=list, init=False, repr=False,
    )
    _bursts: dict[int, BurstDetector] = field(
        default_factory=dict, init=False, repr=False,
    )
    # (electrode, band) each rule's baseline was learnt on.
    _burst_keys: dict[int, tuple[int, str]] = field(
        default_factory=dict, init=False, repr=False,
    )

    def _burst(self, idx: int, rule: ClipRule) -> BurstDetector:
        b = self._bursts.get(idx)
        if b is None:
            b = BurstDetector()
            self._bursts[idx] = b
        key = (rule.eeg_channel, rule.band)
        if self._burst_keys.get(idx) != key:
            # Another electrode or band has its own level: measured against
            # the old baseline it fired at once or stayed mute for up to
            # ~35 s.
            b.reset()
            self._burst_keys[idx] = key
        return b

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
            try:
                arr = bands.by_band[rule.band]
            except KeyError:
                continue
            if not 0 <= rule.eeg_channel < arr.shape[0]:
                continue
            power = float(arr[rule.eeg_channel])
            # Learnt even while the rule is off or in fixed mode, so that
            # switching either back needs no warm-up.
            z = self._burst(idx, rule).score(
                float(np.log10(max(power, 1e-6))), ctx.hop_seconds,
            )
            if not rule.enabled:
                continue

            last_fire = self._last_fire_t.get(idx, -1.0e9)
            if last_fire > t:
                # Clock restarted behind our state (fresh engine re-using
                # this live strategy object): drop the stale lockout.
                last_fire = -1.0e9
                self._last_fire_t[idx] = last_fire
            locked = (t - last_fire) < max(0.0, rule.min_interval_seconds)
            armed = self._armed.get(idx, True)

            if self.adaptive:
                thr = max(0.0, self.relative_threshold)
                above = z >= thr
                rearm = z <= thr * rule.release_ratio
            else:
                above = power >= rule.threshold_uv2
                rearm = power <= rule.threshold_uv2 * rule.release_ratio

            if armed and above and not locked:
                events.append(MappingEvent(
                    kind="note_on",
                    channel=rule.midi_channel,
                    note=rule.midi_note,
                    velocity=rule.velocity,
                ))
                self._held.append((rule.midi_channel, rule.midi_note))
                self._armed[idx] = False
                self._last_fire_t[idx] = t
            elif not armed and rearm:
                self._armed[idx] = True
        return events

    def reset_runtime(self) -> None:
        """Forget per-run state so a fresh engine clock starts clean.

        The baselines go too: a new engine may read another source, whose
        level has nothing to do with the old one.
        """
        self._armed.clear()
        self._last_fire_t.clear()
        self._held.clear()
        self._bursts.clear()
        self._burst_keys.clear()

    def shutdown(self) -> list[MappingEvent]:
        events = [
            MappingEvent(kind="note_off", channel=ch, note=n)
            for (ch, n) in self._held
        ]
        self._held.clear()
        return events
