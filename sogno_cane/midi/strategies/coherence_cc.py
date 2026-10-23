"""Inter-channel coherence -> MIDI Control Change.

Measures how closely two groups of electrodes oscillate together inside a
band (default: the front half of the headset against the back half) and
sends it as a CC value: 0 = independent, 127 = in lockstep.

This is the "coupling" strategy: when the two streams synchronize, the CC
rises, allowing macro modulation in Ableton.

Estimator: magnitude-squared coherence over the last 4 s, from 1 s Hann
segments overlapping by half (7 segments, 1 Hz resolution). Each segment
loses its offset, its drift and the mains sinusoids by least squares, the
bins around the mains harmonics are left out, and the cross-spectrum is
pooled over the band before squaring, so a strong oscillation on a few
bins is not diluted by the rest of the band. Independent signals still
score 0.11-0.15 (estimator bias with 7 segments, measured on independent
pink noise): that bias is subtracted and the rest rescaled, so 0 means
independent.

The previous estimator worked on the 1 s window alone, in 0.25 s segments
(4 Hz resolution). Measured on synthetic signals: delta had no bins left
and always read 0; a shared oscillation also showed in the neighbouring
bands (6 Hz theta read 117 in alpha); independent signals read about 18,
and with mains hum 85 in gamma.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from sogno_cane.eeg.bands import BandPowers
from sogno_cane.midi.context import WindowContext
from sogno_cane.midi.mapper import MappingEvent

HISTORY_SECONDS = 4.0
SEGMENT_SECONDS = 1.0
# Half-width of the slot left out around each mains harmonic.
MAINS_EXCLUSION_HZ = 3.0
# Mean estimate for independent signals (pink noise, 7 segments), per band:
# what "no coherence" reads before the correction.
_BIAS = {
    "delta": 0.1485,
    "theta": 0.1263,
    "alpha": 0.1259,
    "beta": 0.1192,
    "gamma": 0.1143,
}


@dataclass
class CoherenceCCStrategy:
    """Coherence between two channel sets, sent as a smoothed CC value."""

    band: str = "alpha"
    channels_a: tuple[int, ...] = (0, 1, 2, 3)
    channels_b: tuple[int, ...] = (4, 5, 6, 7)
    cc_number: int = 1  # mod wheel
    channel: int = 2
    smoothing: float = 0.4   # 0 = no smoothing, ~1 = very slow
    min_interval_seconds: float = 0.04
    name: str = "CoherenceCC"
    enabled: bool = True

    _last_cc: int = field(default=-1, init=False, repr=False)
    _smoothed: float = field(default=0.0, init=False, repr=False)
    _last_t: float = field(default=-1.0e9, init=False, repr=False)
    # Raw samples of the last HISTORY_SECONDS (the engine hands over only
    # the last second) and the clock at their end.
    _history: np.ndarray | None = field(default=None, init=False, repr=False)
    _history_t: float = field(default=-1.0e9, init=False, repr=False)

    def on_window(
        self,
        eeg_window: np.ndarray,
        bands: BandPowers,
        ctx: WindowContext,
    ) -> list[MappingEvent]:
        if eeg_window.shape[0] < 32:
            return []
        fs = int(bands.sample_rate_hz)
        history = self._remember(eeg_window, ctx.t_seconds, fs)
        if not self.channels_a or not self.channels_b:
            return []

        n_ch = history.shape[1]
        a_idx = [c for c in self.channels_a if 0 <= c < n_ch]
        b_idx = [c for c in self.channels_b if 0 <= c < n_ch]
        if not a_idx or not b_idx:
            return []
        if history.shape[0] < int(round(HISTORY_SECONDS * fs)):
            return []   # the first 4 s of a run

        coh = _band_coherence(
            history[:, a_idx].mean(axis=1),
            history[:, b_idx].mean(axis=1),
            fs,
            self._band_range_hz(),
            float(getattr(ctx, "mains_hz", 50.0)),
        )
        if coh is None:
            return []   # flat line: nothing to measure
        bias = _BIAS.get(self.band, _BIAS["alpha"])
        coh = (coh - bias) / (1.0 - bias)

        # Smooth.
        s = float(np.clip(coh, 0.0, 1.0))
        self._smoothed = (
            self.smoothing * self._smoothed + (1.0 - self.smoothing) * s
        )
        cc_val = int(round(self._smoothed * 127.0))
        if self._last_t > ctx.t_seconds:
            # Clock restarted behind our state (fresh engine re-using this
            # live strategy object): drop the stale throttle timestamp.
            self._last_t = -1.0e9
        if cc_val == self._last_cc:
            return []
        if ctx.t_seconds - self._last_t < max(0.0, self.min_interval_seconds):
            return []
        self._last_cc = cc_val
        self._last_t = ctx.t_seconds
        return [MappingEvent(
            kind="cc",
            channel=self.channel,
            control=self.cc_number,
            value=cc_val,
        )]

    def _remember(
        self, eeg_window: np.ndarray, t: float, fs: int,
    ) -> np.ndarray:
        """Append the window's new samples to the history and return it."""
        history = self._history
        new = int(round((t - self._history_t) * fs))
        if (
            history is None
            or history.shape[1] != eeg_window.shape[1]
            or new < 0
            or new > eeg_window.shape[0]
        ):
            # First window, a new run (clock restarted) or a gap.
            history = np.array(eeg_window, dtype=np.float64)
        elif new > 0:
            history = np.concatenate([history, eeg_window[-new:]], axis=0)
        self._history = history[-int(round(HISTORY_SECONDS * fs)):]
        self._history_t = t
        return self._history

    def _band_range_hz(self) -> tuple[float, float]:
        from sogno_cane.eeg.bands import BAND_RANGES_HZ
        return BAND_RANGES_HZ.get(self.band, (8.0, 13.0))

    def reset_runtime(self) -> None:
        """Forget the emission throttle and the history so a fresh engine
        starts clean."""
        self._last_cc = -1
        self._last_t = -1.0e9
        self._history = None
        self._history_t = -1.0e9

    def shutdown(self) -> list[MappingEvent]:
        return []


@lru_cache(maxsize=8)
def _segment_tools(seg: int, fs: int, mains_hz: float):
    """Window, least-squares cleaner and kept bins for one segment length."""
    t = np.arange(seg, dtype=np.float64) / fs
    columns = [np.ones(seg), t - t.mean()]
    freqs = np.fft.rfftfreq(seg, d=1.0 / fs)
    keep = np.ones(freqs.shape, dtype=bool)
    if mains_hz > 0.0:
        harmonic = mains_hz
        while harmonic - MAINS_EXCLUSION_HZ <= freqs[-1]:
            if harmonic < fs / 2.0:
                columns += [
                    np.sin(2.0 * np.pi * harmonic * t),
                    np.cos(2.0 * np.pi * harmonic * t),
                ]
            keep &= ~(
                (freqs >= harmonic - MAINS_EXCLUSION_HZ)
                & (freqs <= harmonic + MAINS_EXCLUSION_HZ)
            )
            harmonic += mains_hz
    basis = np.stack(columns, axis=1)
    return np.hanning(seg), basis, np.linalg.pinv(basis), freqs, keep


def _band_coherence(
    a: np.ndarray,
    b: np.ndarray,
    sample_rate_hz: int,
    band_hz: tuple[float, float],
    mains_hz: float = 50.0,
) -> float | None:
    """Band coherence of two 1-D signals, in ``[0, 1]`` (biased, see top).

    ``None`` when there is nothing to measure: too short, no bins in the
    band, or a flat line.
    """
    seg = int(round(SEGMENT_SECONDS * sample_rate_hz))
    n = min(len(a), len(b))
    if seg < 8 or n < 2 * seg:
        return None
    window, basis, pinv, freqs, keep = _segment_tools(
        seg, int(sample_rate_hz), float(mains_hz),
    )
    mask = keep & (freqs >= band_hz[0]) & (freqs < band_hz[1])
    if not np.any(mask):
        return None
    starts = range(0, n - seg + 1, seg // 2)
    k = len(starts)
    x = np.stack(
        [a[s:s + seg] for s in starts] + [b[s:s + seg] for s in starts],
        axis=1,
    ).astype(np.float64)
    x -= basis @ (pinv @ x)          # offset, drift and mains removed
    spec = np.fft.rfft(x * window[:, None], axis=0)[mask]
    fa, fb = spec[:, :k], spec[:, k:]
    paa = float(np.sum(np.abs(fa) ** 2))
    pbb = float(np.sum(np.abs(fb) ** 2))
    if paa < 1e-6 or pbb < 1e-6:
        return None
    pab = float(np.sum(np.abs(np.sum(fa * np.conj(fb), axis=1))))
    return pab * pab / (paa * pbb)
