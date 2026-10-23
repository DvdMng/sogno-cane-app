"""Generative Markov-chain melody driven by EEG band powers.

A Markov chain over the scale degrees of a configurable scale produces a
stream of notes. EEG band powers modulate three meta-parameters:

* "energy"   : controls velocity and density (notes per window)
* "darkness" : skews the transition matrix toward lower degrees
* "spread"   : widens the octave range explored

Each meta-parameter is the band's power over the last few seconds, placed
among its own values of the last three minutes (0 = the lowest of them,
1 = the highest), not read on a fixed scale: on real recordings the fixed
scale kept "energy" within 0.37-0.52 (the other two within 0.37-0.77), so
the voice played two notes per step at velocity 70-80 almost every time
and never reached its maximum density.

This gives a virtually unbounded vocabulary while still sounding musical.

Phrases are paced by ``min_interval_seconds``: a new phrase is only generated
once that many seconds have elapsed since the previous one (driven by the
deterministic window clock), so the Markov voice stays sparse instead of
emitting on every analysis hop.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from sogno_cane.eeg.bands import BandPowers
from sogno_cane.midi.context import WindowContext
from sogno_cane.midi.mapper import MappingEvent
from sogno_cane.midi.pacing import FLAT_LOG_POWER
from sogno_cane.midi.scales import (
    MidiNoteRange,
    build_scale,
)

# A phrase reads each band over the last few seconds, not from the last 1 s
# window alone. On real recordings a 1 s reading changes mostly from one
# second to the next (beta: over 80% of its variance is faster than 5 s),
# so consecutive phrases came out unrelated (correlation ~0).
_READING_TAU_SECONDS = 2.5
# The smoothed reading is then placed among its own values of the last
# three minutes (one per second). Measured against stretching a
# self-calibrating range, as the note voices do: phrase-to-phrase
# correlation ~0.5 on the recordings (0.3-0.4), a rise lasting two minutes
# still reads high at its end (0.74 against 0.43), and a 3 s movement
# artefact leaves the next 30 s as they were (0.40 against 0.05: the range
# had been stretched by the artefact).
_HISTORY_SECONDS = 180
_HISTORY_STEP_SECONDS = 1.0


@dataclass
class _Reading:
    """One meta-parameter: its band, the band's smoothed log power and the
    recent values it is ranked against."""

    band: str
    smoothed: float
    history: deque = field(
        default_factory=lambda: deque(maxlen=int(
            _HISTORY_SECONDS / _HISTORY_STEP_SECONDS
        ))
    )
    since_sample: float = 0.0

    def update(self, x: float, dt: float) -> float:
        """Take a new log power; return the smoothed value's rank, 0..1."""
        a = min(1.0, max(0.0, dt) / _READING_TAU_SECONDS)
        self.smoothed += a * (x - self.smoothed)
        if self.history:
            h = np.fromiter(self.history, dtype=np.float64)
            level = float(
                (np.count_nonzero(h < self.smoothed)
                 + 0.5 * np.count_nonzero(h == self.smoothed)) / h.size
            )
        else:
            level = 0.5
        self.since_sample += max(0.0, dt)
        due = self.since_sample >= _HISTORY_STEP_SECONDS - 1e-6
        if due or not self.history:
            self.history.append(self.smoothed)
            self.since_sample = 0.0
        return level


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
    # Fixed note length (gate) in seconds; 0 = hold each phrase until the
    # next one starts (legato). >0 = engine releases after exactly this long.
    note_length_seconds: float = 0.015
    channel: int = 3
    name: str = "MarkovGenerative"
    enabled: bool = True
    seed: int | None = None

    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(),
        init=False, repr=False,
    )
    _last_idx: int | None = field(default=None, init=False, repr=False)
    # (MIDI channel, note) of the legato notes still sounding: the note_off
    # goes where the note went, even if the channel changed meanwhile.
    _held: list[tuple[int, int]] = field(
        default_factory=list, init=False, repr=False,
    )
    _last_phrase_t: float = field(default=-1.0e9, init=False, repr=False)
    _readings: dict[str, _Reading] = field(
        default_factory=dict, init=False, repr=False,
    )
    # Current value (0..1) of each meta-parameter.
    _levels: dict[str, float] = field(
        default_factory=dict, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        if self.seed is not None:
            self._rng = np.random.default_rng(self.seed)

    def _track(
        self, role: str, band: str, bands: BandPowers, dt: float,
    ) -> None:
        """Update one meta-parameter (0..1, relative to its recent values)."""
        try:
            p = float(np.mean(bands.by_band[band]))
        except KeyError:
            self._levels[role] = 0.0
            return
        x = float(np.log10(max(p, 1e-6)))
        reading = self._readings.get(role)
        if reading is None or reading.band != band:
            # A new band starts a new reading: the old band's values say
            # nothing about this one's.
            reading = _Reading(band=band, smoothed=x)
            self._readings[role] = reading
        self._levels[role] = reading.update(x, dt)

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

        # Self-heal when the clock restarted behind our state (fresh engine
        # re-using this live strategy object): a stale timestamp would mute
        # the voice for the previous run's whole duration.
        if self._last_phrase_t > t:
            self._last_phrase_t = -1.0e9

        # No EEG at all (flat line, lost samples): stay quiet, and keep it
        # out of the ranges, which would otherwise be stretched for minutes.
        try:
            energy_p = float(np.mean(bands.by_band[self.energy_band]))
        except KeyError:
            energy_p = 0.0
        flat = np.log10(max(energy_p, 1e-6)) < FLAT_LOG_POWER
        if not flat:
            # Tracked on every window, not only when a phrase is due, so the
            # range reflects the last tens of seconds.
            dt = ctx.hop_seconds
            self._track("energy", self.energy_band, bands, dt)
            self._track("darkness", self.darkness_band, bands, dt)
            self._track("spread", self.spread_band, bands, dt)

        # Only generate a new phrase once the interval has elapsed.
        if t - self._last_phrase_t < max(0.0, self.min_interval_seconds):
            return events

        # Release the previous phrase before starting a new one.
        for ch, n in self._held:
            events.append(MappingEvent(kind="note_off", channel=ch, note=n))
        self._held.clear()

        if flat or "energy" not in self._levels:
            # Nothing to play from; keep the timer so we re-check next window.
            self._last_phrase_t = t
            return events
        energy = self._levels["energy"]
        darkness = self._levels.get("darkness", 0.0)
        spread = self._levels.get("spread", 0.0)

        n_notes = 1 + int(round(energy * (self.max_notes_per_window - 1)))
        velocity = max(1, min(127, int(40 + energy * 80)))
        gate_len = max(0.0, self.note_length_seconds)
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
                duration=gate_len,
            ))
            # Gated notes are released by the engine; only legato notes
            # are held until the next phrase.
            if gate_len == 0.0:
                self._held.append((self.channel, note))
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

    def reset_runtime(self) -> None:
        """Forget per-run state so a fresh engine clock starts clean.

        The ranges go too: a new engine may read another source, whose
        levels have nothing to do with the old one.
        """
        self._held.clear()
        self._last_phrase_t = -1.0e9
        self._last_idx = None
        self._readings.clear()
        self._levels.clear()

    def shutdown(self) -> list[MappingEvent]:
        events = [
            MappingEvent(kind="note_off", channel=ch, note=n)
            for ch, n in self._held
        ]
        self._held.clear()
        return events
