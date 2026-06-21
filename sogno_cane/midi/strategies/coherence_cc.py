"""Inter-channel coherence -> MIDI Control Change.

For a given EEG window, computes the magnitude-squared coherence between two
channel sets (typical use: left vs right, or human vs dog when two engines
are linked) inside a target band, then maps it to a CC value 0..127.

This is the "coupling" strategy: when the two streams synchronize, the CC
rises, allowing macro modulation in Ableton.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sogno_cane.eeg.bands import BandPowers
from sogno_cane.midi.context import WindowContext
from sogno_cane.midi.mapper import MappingEvent


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

    def on_window(
        self,
        eeg_window: np.ndarray,
        bands: BandPowers,
        ctx: WindowContext,
    ) -> list[MappingEvent]:
        if eeg_window.shape[0] < 32:
            return []
        if not self.channels_a or not self.channels_b:
            return []

        n_ch = eeg_window.shape[1]
        a_idx = [c for c in self.channels_a if 0 <= c < n_ch]
        b_idx = [c for c in self.channels_b if 0 <= c < n_ch]
        if not a_idx or not b_idx:
            return []

        a = eeg_window[:, a_idx].mean(axis=1)
        b = eeg_window[:, b_idx].mean(axis=1)

        coh = _band_coherence(
            a, b,
            bands.sample_rate_hz,
            self._band_range_hz(),
        )

        # Smooth.
        s = float(np.clip(coh, 0.0, 1.0))
        self._smoothed = (
            self.smoothing * self._smoothed + (1.0 - self.smoothing) * s
        )
        cc_val = int(round(self._smoothed * 127.0))
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

    def _band_range_hz(self) -> tuple[float, float]:
        from sogno_cane.eeg.bands import BAND_RANGES_HZ
        return BAND_RANGES_HZ.get(self.band, (8.0, 13.0))

    def shutdown(self) -> list[MappingEvent]:
        return []


def _band_coherence(
    a: np.ndarray,
    b: np.ndarray,
    sample_rate_hz: int,
    band_hz: tuple[float, float],
) -> float:
    """Mean magnitude-squared coherence inside ``band_hz`` for two 1-D signals.

    Implementation: Welch-style averaging with overlapping segments. Returns
    a value in ``[0, 1]``. At least two segments are required for the estimate
    to be meaningful (a single segment trivially yields coherence 1).
    """
    n = len(a)
    if n < 16:
        return 0.0
    seg = max(16, n // 4)
    step = max(1, seg // 2)
    f_axis = np.fft.rfftfreq(seg, d=1.0 / sample_rate_hz)
    band_mask = (f_axis >= band_hz[0]) & (f_axis < band_hz[1])
    if not np.any(band_mask):
        return 0.0

    window = np.hanning(seg)
    paa = np.zeros_like(f_axis, dtype=np.complex128)
    pbb = np.zeros_like(f_axis, dtype=np.complex128)
    pab = np.zeros_like(f_axis, dtype=np.complex128)
    count = 0
    for start in range(0, n - seg + 1, step):
        sa = (a[start:start + seg] - a[start:start + seg].mean()) * window
        sb = (b[start:start + seg] - b[start:start + seg].mean()) * window
        fa = np.fft.rfft(sa)
        fb = np.fft.rfft(sb)
        paa += fa * np.conj(fa)
        pbb += fb * np.conj(fb)
        pab += fa * np.conj(fb)
        count += 1
    if count < 2:
        # Single segment -> coherence is identically 1 and meaningless.
        return 0.0
    paa /= count
    pbb /= count
    pab /= count

    denom = np.real(paa) * np.real(pbb)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    coh = np.abs(pab) ** 2 / denom
    return float(np.mean(np.real(coh[band_mask])))
