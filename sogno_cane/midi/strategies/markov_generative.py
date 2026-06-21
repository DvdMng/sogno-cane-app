"""Generative Markov-chain melody driven by EEG band powers.

A Markov chain over the scale degrees of a configurable scale produces a
stream of notes. EEG band powers modulate three meta-parameters:

* "energy"   : controls velocity and density (notes per window)
* "darkness" : skews the transition matrix toward lower degrees
* "spread"   : widens the octave range explored

This gives a virtually unbounded vocabulary while still sounding musical.

Phrases are paced by ``min_interval_seconds``: a new phrase is only generated
once that many seconds have elapsed since the previous one (driven by the
deterministic window clock), so the Markov voice stays sparse instead of
emitting on every analysis hop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sogno_cane.eeg.bands import BandPowers
from sogno_cane.midi.context import WindowContext
from sogno_cane.midi.mapper import MappingEvent
from sogno_cane.midi.scales import (
    MidiNoteRange,
    build_scale,
)


@dataclass
class MarkovGenerativeStrategy:
    scale: str = "dorian"
    root: str = "D"
    note_range: MidiNoteRange = field(
        default_factory=lambda: MidiNoteRange(48, 84)
    )
    energy_band: str = "beta"
    darkness_band: str = "theta"
    spread_band: str = "alpha"
    max_notes_per_window: int = 4
    min_interval_seconds: float = 2.5
    channel: int = 3
    name: str = "MarkovGenerative"
    enabled: bool = True
    seed: int | None = None

    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(),
        init=False, repr=False,
    )
    _last_idx: int | None = field(default=None, init=False, repr=False)
    _held: list[int] = field(default_factory=list, init=False, repr=False)
    _last_phrase_t: float = field(default=-1.0e9, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.seed is not None:
            self._rng = np.random.default_rng(self.seed)

    def on_window(
        self,
        eeg_window: np.ndarray,
        bands: BandPowers,
        ctx: WindowContext,
    ) -> list[MappingEvent]:
        scale_notes = build_scale(self.scale, self.root, self.note_range)
        if not scale_notes:
            return []
        events: list[MappingEvent] = []
        t = ctx.t_seconds

        # Only generate a new phrase once the interval has elapsed.
        if t - self._last_phrase_t < max(0.0, self.min_interval_seconds):
            return events

        # Release the previous phrase before starting a new one.
        for n in self._held:
            events.append(MappingEvent(
                kind="note_off", channel=self.channel, note=n,
            ))
        self._held.clear()

        energy = _band01(bands, self.energy_band)
        darkness = _band01(bands, self.darkness_band)
        spread = _band01(bands, self.spread_band)

        if energy < 0.05:
            # Too quiet to speak; keep the timer so we re-check next window.
            self._last_phrase_t = t
            return events

        n_notes = 1 + int(round(energy * (self.max_notes_per_window - 1)))
        velocity = max(1, min(127, int(40 + energy * 80)))
        phrase_notes: set[int] = set()
        for _ in range(n_notes):
            idx = self._next_index(len(scale_notes), darkness, spread)
            note = int(scale_notes[idx])
            self._last_idx = idx
            # Never emit the same pitch twice in one phrase: a duplicate
            # note_on with no intervening note_off stacks/steals voices on a
            # real synth and leaves a hung note when the phrase is released.
            if note in phrase_notes:
                continue
            phrase_notes.add(note)
            events.append(MappingEvent(
                kind="note_on",
                channel=self.channel,
                note=note,
                velocity=velocity,
            ))
            self._held.append(note)
        self._last_phrase_t = t
        return events

    def _next_index(
        self, n_steps: int, darkness: float, spread: float
    ) -> int:
        if n_steps <= 1:
            return 0
        # Transition: random walk biased by darkness (down) and spread (range).
        center = (n_steps - 1) * (1.0 - darkness) * 0.7 + darkness * 0.1
        sigma = max(0.5, 0.5 + spread * (n_steps * 0.3))
        if self._last_idx is None:
            self._last_idx = int(np.clip(center, 0, n_steps - 1))
        # Mean drifts toward `center` with a small pull strength.
        mean = 0.85 * self._last_idx + 0.15 * center
        x = int(round(self._rng.normal(mean, sigma)))
        return int(np.clip(x, 0, n_steps - 1))

    def shutdown(self) -> list[MappingEvent]:
        events = [
            MappingEvent(kind="note_off", channel=self.channel, note=n)
            for n in self._held
        ]
        self._held.clear()
        return events


def _band01(bands: BandPowers, name: str) -> float:
    """Map a band's mean log-power to a 0..1 indicator with sane bounds."""
    try:
        p = float(np.mean(bands.by_band[name]))
    except KeyError:
        return 0.0
    log_p = np.log10(max(p, 1e-6))
    # Calibrated for typical Unicorn-scale signals.
    return float(np.clip((log_p + 1.0) / 4.0, 0.0, 1.0))
