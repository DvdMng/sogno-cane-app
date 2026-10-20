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
# Burst detection (scale-free event thresholds)                               #
# --------------------------------------------------------------------------- #
@dataclass
class BurstDetector:
    """How unusual is this value, relative to the signal's own recent past?

    Tracks a slow baseline of a streaming scalar (a log band power) and its
    typical deviation, and reports each new value as a deviation score:
    0 = ordinary, 2 = two typical deviations above normal. Event strategies
    fire on that score, so a threshold means the same thing on the
    simulator, on a real Unicorn with dry electrodes, or with gel — unlike a
    fixed uV^2 level, which was tuned on the simulator and never fired on
    real data (measured: 0 events/min).

    :class:`AdaptiveNorm` is the wrong tool here: it stretches the recent
    range to fill 0..1 for pitch mapping, so every new local maximum reads
    as 1.0 and an "event" fires many times a minute whatever the threshold.
    """

    tau_seconds: float = 20.0
    # Windows (0.1 s each) spent learning before any score is reported.
    warmup_windows: int = 20
    _mean: Optional[float] = field(default=None, repr=False)
    _dev: Optional[float] = field(default=None, repr=False)
    _n: int = field(default=0, repr=False)

    def reset(self) -> None:
        self._mean = None
        self._dev = None
        self._n = 0

    def score(self, x: float, dt: float) -> float:
        """Deviation score of ``x`` against the past, then learn from it.

        Scoring happens *before* learning, so a burst is measured against
        the quiet baseline that preceded it rather than against itself.

        Start-up: the baseline is an exact running average until it has as
        much history as the moving average would weigh (``1/n`` beats the
        EMA rate), and nothing fires during a short warm-up. Without this a
        guessed initial deviation made events fire ~3x too often for the
        first ~20 s of every session (measured on real recordings).
        """
        if self._mean is None or self._dev is None:
            self._mean = x
            self._dev = 0.0
            self._n = 1
            return 0.0
        z = 0.0
        if self._n >= self.warmup_windows:
            z = (x - self._mean) / max(self._dev, 1e-3)
        self._n += 1
        ema = min(1.0, max(0.0, dt) / max(1e-6, self.tau_seconds))
        a = max(ema, 1.0 / self._n)
        self._mean += a * (x - self._mean)
        self._dev += a * (abs(x - self._mean) - self._dev)
        return z


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

    def _heal_clock(self, t: float) -> None:
        """Recover from a simulation clock that restarted behind our state.

        Strategy objects outlive engines (the UI re-adds the same live
        instances to every new engine, whose clock starts at zero), so a
        timestamp from a previous run would otherwise keep the voice mute
        for up to the previous session's whole duration.
        """
        if self._last_change_t > t:
            self._last_change_t = -1.0e9
        if self._note_on_t is not None and self._note_on_t > t:
            self._note_on_t = None

    def is_starved(self, t: float, *, max_silence: float) -> bool:
        """True if the voice has been quiet for at least ``max_silence`` s.

        ``max_silence <= 0`` disables the check (never starved).
        """
        if max_silence <= 0.0:
            return False
        self._heal_clock(t)
        return (t - self._last_change_t) >= max_silence

    def may_change(
        self,
        t: float,
        norm: float,
        *,
        min_interval: float,
        change_threshold: float,
        max_silence: float = 0.0,
    ) -> bool:
        """True if a new note is allowed *now* given interval + hysteresis.

        ``max_silence`` (seconds, 0 = off) is a liveness failsafe: once the
        voice has been quiet that long, the hysteresis check is bypassed so
        a stable signal can no longer starve the voice forever. The rate
        limit (``min_interval``) still applies.
        """
        self._heal_clock(t)
        if t - self._last_change_t < max(0.0, min_interval):
            return False
        if self.is_starved(t, max_silence=max_silence):
            return True
        if (
            self._last_norm is not None
            and abs(norm - self._last_norm) < max(0.0, change_threshold)
        ):
            return False
        return True

    def may_release(self, t: float, *, min_hold: float) -> bool:
        """True if the currently-held note has met its minimum hold time."""
        self._heal_clock(t)
        if self._note_on_t is None:
            return True
        return (t - self._note_on_t) >= max(0.0, min_hold)

    def commit_on(self, t: float, norm: float) -> None:
        self._last_change_t = t
        self._last_norm = norm
        self._note_on_t = t

    def commit_off(self) -> None:
        self._note_on_t = None
