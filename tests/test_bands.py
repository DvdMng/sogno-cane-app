import numpy as np

from sogno_cane.eeg.bands import BAND_RANGES_HZ, compute_band_powers


def _sine(freq, fs, n, amp=10.0, n_ch=4):
    t = np.arange(n) / fs
    col = amp * np.sin(2 * np.pi * freq * t)
    return np.tile(col[:, None], (1, n_ch))


def test_band_power_localises_frequency():
    fs = 250
    n = fs  # 1 s
    # A 10 Hz sine should put its energy in alpha (8-13 Hz).
    eeg = _sine(10.0, fs, n)
    bp = compute_band_powers(eeg, fs)
    powers = {b: bp.channel_mean(b) for b in BAND_RANGES_HZ}
    assert max(powers, key=powers.get) == "alpha"


def test_band_power_beta():
    fs = 250
    eeg = _sine(20.0, fs, fs)
    bp = compute_band_powers(eeg, fs)
    powers = {b: bp.channel_mean(b) for b in BAND_RANGES_HZ}
    assert max(powers, key=powers.get) == "beta"


def test_relative_power_sums_to_one():
    fs = 250
    eeg = _sine(10.0, fs, fs) + _sine(20.0, fs, fs)
    bp = compute_band_powers(eeg, fs)
    rel = np.stack([bp.relative(b) for b in BAND_RANGES_HZ], axis=0)
    assert np.allclose(rel.sum(axis=0), 1.0, atol=1e-6)


def test_parseval_amplitude_scaling():
    """A 10 uV sine has variance 50 uV^2; integrated PSD should be close."""
    fs = 250
    eeg = _sine(10.0, fs, fs * 4, amp=10.0, n_ch=1)
    bp = compute_band_powers(eeg, fs)
    total = float(bp.total_power()[0])
    assert 35.0 < total < 65.0  # ~50 uV^2 within windowing tolerance


def test_requires_2d():
    import pytest
    with pytest.raises(ValueError):
        compute_band_powers(np.zeros(100), 250)
