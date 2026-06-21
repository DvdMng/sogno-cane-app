"""Musical scales and helpers for EEG -> note mapping.

Scales are expressed as a sorted tuple of pitch classes (semitones from the
root, 0..11) or, for microtonal scales, as a tuple of cents from the root
followed by an octave size in cents. The default is 12-EDO; a custom
microtonal scale can be provided at runtime by the user.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Canonical 12-EDO scales by pitch-class set.
SCALES: dict[str, tuple[int, ...]] = {
    "chromatic":         (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    "major":             (0, 2, 4, 5, 7, 9, 11),
    "natural_minor":     (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor":    (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor":     (0, 2, 3, 5, 7, 9, 11),
    "dorian":            (0, 2, 3, 5, 7, 9, 10),
    "phrygian":          (0, 1, 3, 5, 7, 8, 10),
    "lydian":            (0, 2, 4, 6, 7, 9, 11),
    "mixolydian":        (0, 2, 4, 5, 7, 9, 10),
    "aeolian":           (0, 2, 3, 5, 7, 8, 10),
    "locrian":           (0, 1, 3, 5, 6, 8, 10),
    "major_pentatonic":  (0, 2, 4, 7, 9),
    "minor_pentatonic":  (0, 3, 5, 7, 10),
    "blues":             (0, 3, 5, 6, 7, 10),
    "whole_tone":        (0, 2, 4, 6, 8, 10),
    "diminished":        (0, 2, 3, 5, 6, 8, 9, 11),
    "augmented":         (0, 3, 4, 7, 8, 11),
    "japanese_hirajoshi":(0, 2, 3, 7, 8),
    "egyptian_suspended":(0, 2, 5, 7, 10),
    "hijaz":             (0, 1, 4, 5, 7, 8, 11),
    "raga_bhairav":      (0, 1, 4, 5, 7, 8, 11),
}

# Note name -> root pitch class (C=0..B=11).
NOTE_NAME_TO_PC: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11,
}


@dataclass(frozen=True)
class MidiNoteRange:
    """Allowed pitch range for output notes, inclusive."""

    lo: int = 24   # C1
    hi: int = 96   # C7

    def __post_init__(self) -> None:
        if not 0 <= self.lo <= 127:
            raise ValueError("lo out of MIDI range")
        if not 0 <= self.hi <= 127:
            raise ValueError("hi out of MIDI range")
        if self.lo > self.hi:
            raise ValueError("lo must be <= hi")


def build_scale(
    scale_name: str,
    root_note_name: str = "C",
    note_range: MidiNoteRange | None = None,
) -> tuple[int, ...]:
    """Build a sorted tuple of MIDI note numbers for a given scale.

    Parameters
    ----------
    scale_name
        Key in :data:`SCALES`.
    root_note_name
        Note name e.g. ``"C"``, ``"F#"``, ``"Bb"``.
    note_range
        Inclusive ``[lo, hi]`` MIDI range to populate. Defaults to C1..C7.
    """
    if scale_name not in SCALES:
        raise KeyError(
            f"Unknown scale {scale_name!r}; "
            f"choose one of {sorted(SCALES)}"
        )
    if root_note_name not in NOTE_NAME_TO_PC:
        raise KeyError(
            f"Unknown root {root_note_name!r}; "
            f"valid: {sorted(NOTE_NAME_TO_PC)}"
        )
    rng = note_range or MidiNoteRange()
    pcs = SCALES[scale_name]
    root_pc = NOTE_NAME_TO_PC[root_note_name]
    notes: list[int] = []
    for octave in range(-1, 11):
        base = (octave + 1) * 12 + root_pc
        for pc in pcs:
            n = base + pc
            if rng.lo <= n <= rng.hi:
                notes.append(n)
    return tuple(sorted(set(notes)))


def quantize_to_scale(
    value_0_1: float,
    scale_notes: tuple[int, ...],
) -> int:
    """Map a normalized 0..1 value to a note inside the given scale."""
    if not scale_notes:
        raise ValueError("scale_notes must be non-empty")
    v = float(np.clip(value_0_1, 0.0, 1.0))
    idx = int(round(v * (len(scale_notes) - 1)))
    return int(scale_notes[idx])
