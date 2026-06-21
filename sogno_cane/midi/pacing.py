"""Musical pacing helpers shared by every note-emitting strategy.

The original build exposed "min interval", "min hold" and "change threshold"
controls in the UI, but no strategy ever read them, so the controls were
silent no-ops and the output retriggered on every 100 ms hop (~140 MIDI
events per second — unmusically dense). These helpers are the missing piece:
strategies compose a :class:`NoteGate` per voice and the engine feeds them a
deterministic timestamp via :class:`~sogno_cane.midi.context.WindowContext`.

Two independent concerns:

* :class:`NoteGate` — *when* a voice is allowed to change/release a note
  (rate limiting + hysteresis on the driving value + minimum hold time).
* :class:`AdaptiveNorm` — *how* a raw log-power is mapped to 0..1, with a
  slowly-adapting floor/ceiling (automatic gain control) so the full musical
  range is used regardless of the signal's absolute scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Adaptive normalisation (AGC)                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class AdaptiveNorm:
    """Map a streaming scalar to 0..1 using a self-calibrating window.

    The floor/ceiling expand quickly toward new extremes (``attack``) and
    contract slowly back toward the running centre (``decay``) so a quiet
    passage does not permanently widen the range. A minimum span keeps the
    mapping from exploding sensitivity when the signal is nearly constant.
    """

    attack: float = 0.25
    decay: float = 0.02
    min_span: float = 0.5
    floor: Optional[float] = None
    ceil: Optional[float] = None

    def reset(self) -> None:
        self.floor = None
        self.ceil = None

    def normalize(self, x: float) -> float:
        if self.floor is None or self.ceil is None:
            self.floor = x - self.min_span * 0.5
            self.ceil = x + self.min_span * 0.5
        # Expand toward new extremes quickly.
        if x < self.floor:
            self.floor += (x - self.floor) * self.attack
        if x > self.ceil:
            self.ceil += (x - self.ceil) * self.attack
        # Contract slowly toward the current value (re-centre over time).
        self.floor += (x - self.floor) * self.decay * 0.5
        self.ceil += (x - self.ceil) * self.decay * 0.5
        # Enforce a minimum span around the midpoint.
        span = self.ceil - self.floor
        if span < self.min_span:
            mid = 0.5 * (self.floor + self.ceil)
            self.floor = mid - self.min_span * 0.5
            self.ceil = mid + self.min_span * 0.5
            span = self.min_span
        v = (x - self.floor) / span
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


# --------------------------------------------------------------------------- #
# Note gate (rate limiting + hysteresis)                                       #
# --------------------------------------------------------------------------- #
@dataclass
class NoteGate:
    """Decide when a single voice may speak, change, or release.

    All thresholds are read live every call so the UI can tweak them in real
    time. State is timestamp-based (seconds), driven by the deterministic
    simulation clock rather than wall-clock time.
    """

    _last_change_t: float = field(default=-1.0e9, repr=False)
    _last_norm: Optional[float] = field(default=None, repr=False)
    _note_on_t: Optional[float] = field(default=None, repr=False)

    def reset(self) -> None:
        self._last_change_t = -1.0e9
        self._last_norm = None
        self._note_on_t = None

    def may_change(
        self,
        t: float,
        norm: float,
        *,
        min_interval: float,
        change_threshold: float,
    ) -> bool:
        """True if a new note is allowed *now* given interval + hysteresis."""
        if t - self._last_change_t < max(0.0, min_interval):
            return False
        if (
            self._last_norm is not None
            and abs(norm - self._last_norm) < max(0.0, change_threshold)
        ):
            return False
        return True

    def may_release(self, t: float, *, min_hold: float) -> bool:
        """True if the currently-held note has met its minimum hold time."""
        if self._note_on_t is None:
            return True
        return (t - self._note_on_t) >= max(0.0, min_hold)

    def commit_on(self, t: float, norm: float) -> None:
        self._last_change_t = t
        self._last_norm = norm
        self._note_on_t = t

    def commit_off(self) -> None:
        self._note_on_t = None
