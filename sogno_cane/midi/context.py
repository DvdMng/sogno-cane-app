"""Per-window context passed from the mapping engine to each strategy.

Carrying an explicit, deterministic ``t_seconds`` (simulation time, derived
from the sample counter rather than the wall clock) is what lets strategies
implement musical *pacing* — minimum interval between notes, minimum hold
time, rhythmic quantisation — in a way that is fully reproducible under a
fixed RNG seed and therefore unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowContext:
    """Immutable metadata describing the window handed to a strategy.

    Attributes
    ----------
    t_seconds
        Simulation time (seconds) at the *end* of the current window,
        accumulated from sample counts — independent of wall-clock jitter.
    hop_seconds
        Nominal interval between two consecutive window evaluations.
    sample_rate_hz
        EEG sampling rate the window was computed at.
    window_samples
        Number of samples in the analysis window.
    mains_hz
        Power-line frequency the band powers exclude (0 = none), for
        strategies that run their own spectral analysis.
    """

    t_seconds: float
    hop_seconds: float
    sample_rate_hz: int
    window_samples: int
    mains_hz: float = 50.0
