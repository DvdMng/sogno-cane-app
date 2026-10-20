"""Band-power extraction from windowed EEG samples."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Canonical EEG bands (Hz).
BAND_RANGES_HZ: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


@dataclass
class BandPowers:
    """Per-channel mean power in each canonical band, in uV^2."""

    by_band: dict[str, np.ndarray]      # band_name -> (n_channels,) array
    sample_rate_hz: int
    window_samples: int

    def channel_mean(self, band: str) -> float:
        """Mean across channels for a given band, in uV^2."""
        return float(np.mean(self.by_band[band]))

    def total_power(self) -> np.ndarray:
        """Sum of all band powers per channel."""
        return np.sum(np.stack(list(self.by_band.values()), axis=0), axis=0)

    def relative(self, band: str) -> np.ndarray:
        """Relative band power per channel (0..1)."""
        total = self.total_power()
        total = np.where(total < 1e-9, 1.0, total)
        return self.by_band[band] / total


def _detrend_linear(x: np.ndarray) -> np.ndarray:
    """Remove each channel's mean and least-squares slope.

    Real Unicorn amplifiers are DC-coupled: every electrode sits on an
    offset of tens of thousands of microvolts that slowly drifts. Left in,
    that offset leaks through the window into the lowest bins and swamps
    every band (measured on real BCI Core recordings: delta came out at
    100% of the power, the brain signal at 0%).
    """
    n = x.shape[0]
    t = np.arange(n, dtype=np.float64)
    t -= t.mean()
    centred = x - x.mean(axis=0)
    denom = float(np.dot(t, t))
    if denom <= 0.0:
        return centred
    slope = (t @ centred) / denom
    return centred - t[:, None] * slope[None, :]


def compute_band_powers(
    eeg: np.ndarray,
    sample_rate_hz: int,
    bands: dict[str, tuple[float, float]] | None = None,
    *,
    detrend: bool = True,
    mains_hz: float = 50.0,
    notch_half_width_hz: float = 2.0,
) -> BandPowers:
    """Compute mean band power per channel using a Hann-windowed FFT.

    Parameters
    ----------
    eeg
        Array of shape ``(n_samples, n_channels)`` in microvolts.
    sample_rate_hz
        EEG sampling rate.
    bands
        Optional override of band ranges. Defaults to canonical EEG bands.
    detrend
        Remove each channel's offset and linear drift before the FFT
        (required for DC-coupled hardware such as the Unicorn BCI Core).
    mains_hz
        Power-line frequency to exclude from every band, with all its
        harmonics up to Nyquist (50 Hz in Europe). With dry electrodes the
        mains hum is otherwise the whole of the gamma band. 0 disables.
    notch_half_width_hz
        Half-width of each excluded mains slot.

    Returns
    -------
    BandPowers
        Per-channel band powers in uV^2.
    """
    if eeg.ndim != 2:
        raise ValueError(f"eeg must be 2-D, got shape {eeg.shape}")
    n_samples, n_channels = eeg.shape
    if n_samples < 8:
        raise ValueError("Need at least 8 samples for band power")

    bands = bands or BAND_RANGES_HZ

    signal = eeg.astype(np.float64)
    if detrend:
        signal = _detrend_linear(signal)

    # Apply a Hann window for spectral leakage suppression.
    window = np.hanning(n_samples).astype(np.float64)
    window_norm = np.sum(window ** 2)
    windowed = signal * window[:, None]

    # Real FFT.
    fft = np.fft.rfft(windowed, axis=0)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)
    # Power spectral density (Welch-style normalization).
    psd = (np.abs(fft) ** 2) / (sample_rate_hz * window_norm)
    # One-sided spectrum: double every positive-frequency bin except DC. The
    # true Nyquist bin (present only for an EVEN window length) is counted
    # once, so it must NOT be doubled; for an ODD window the final rfft bin is
    # an ordinary positive-frequency bin and is doubled like the rest.
    psd[1:] *= 2.0
    if n_samples % 2 == 0:
        psd[-1] /= 2.0

    # Spectral notch: drop the bins around the mains frequency and its
    # harmonics. Excluding bins (rather than filtering) has no state and no
    # start-up transient, so it is exact from the very first window.
    keep = np.ones_like(freqs, dtype=bool)
    if mains_hz > 0.0 and len(freqs) > 1:
        harmonic = float(mains_hz)
        while harmonic - notch_half_width_hz <= freqs[-1]:
            keep &= ~(
                (freqs >= harmonic - notch_half_width_hz)
                & (freqs <= harmonic + notch_half_width_hz)
            )
            harmonic += float(mains_hz)

    out: dict[str, np.ndarray] = {}
    for band_name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi) & keep
        if not np.any(mask):
            out[band_name] = np.zeros(n_channels, dtype=np.float64)
        else:
            # Integrate PSD across the band -> uV^2.
            df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
            out[band_name] = np.sum(psd[mask], axis=0) * df
    return BandPowers(
        by_band=out,
        sample_rate_hz=sample_rate_hz,
        window_samples=n_samples,
    )
