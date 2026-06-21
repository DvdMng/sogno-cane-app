"""Two distinct subject profiles for the EEG simulator: human and dog.

Both profiles produce a Unicorn-compatible 8-channel stream, but the spectral
content, baseline state, artifact rates and motion patterns differ.

Human profile is calibrated against typical adult resting EEG literature:
prominent posterior alpha (8-12 Hz, ~30 uV), moderate frontal beta, lower
theta/delta, occasional eye-blink artifacts at ~0.3 Hz.

Dog profile is informed by veterinary EEG studies (e.g. Pellegrino & Sica
2004, Itoh et al. 2018): canine resting EEG shows weaker alpha equivalents,
stronger theta (4-8 Hz) and gamma activity, more frequent motion artifacts,
and a different dominant frequency band.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BandSpec:
    """Spectral content of one frequency band.

    amplitude_uv is the standard-deviation contribution to each channel in
    microvolts; bandwidth_hz is the gaussian sigma of the band-pass envelope
    centered at center_hz.
    """

    name: str
    center_hz: float
    bandwidth_hz: float
    amplitude_uv: float


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    channel_names: tuple[str, ...]
    sample_rate_hz: int
    pink_noise_amplitude_uv: float
    line_noise_amplitude_uv: float
    line_noise_hz: float
    bands: tuple[BandSpec, ...]
    blink_rate_hz: float
    blink_amplitude_uv: float
    emg_burst_rate_hz: float
    emg_burst_amplitude_uv: float
    motion_drift_amplitude_uv: float
    accelerometer_baseline_g: tuple[float, float, float]
    accelerometer_noise_g: float
    gyroscope_noise_dps: float
    battery_start_percent: float
    battery_drain_percent_per_hour: float
    # Channels that get blink artifact (frontal). Indices into channel_names.
    blink_channels: tuple[int, ...] = field(default_factory=tuple)
    # Optional per-channel amplitude multiplier (e.g. posterior alpha boost).
    band_channel_weights: dict[str, tuple[float, ...]] = field(
        default_factory=dict
    )


HUMAN_PROFILE = DeviceProfile(
    name="HUMAN",
    channel_names=("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"),
    sample_rate_hz=250,
    pink_noise_amplitude_uv=8.0,
    line_noise_amplitude_uv=2.0,
    line_noise_hz=50.0,
    bands=(
        BandSpec("delta", 2.0, 1.5, 6.0),
        BandSpec("theta", 6.0, 2.0, 5.0),
        BandSpec("alpha", 10.0, 1.5, 18.0),
        BandSpec("beta", 20.0, 6.0, 6.0),
        BandSpec("gamma", 45.0, 10.0, 2.0),
    ),
    blink_rate_hz=0.30,
    blink_amplitude_uv=80.0,
    emg_burst_rate_hz=0.15,
    emg_burst_amplitude_uv=20.0,
    motion_drift_amplitude_uv=4.0,
    accelerometer_baseline_g=(0.0, 0.0, 1.0),
    accelerometer_noise_g=0.005,
    gyroscope_noise_dps=0.5,
    battery_start_percent=98.0,
    battery_drain_percent_per_hour=20.0,
    blink_channels=(0,),  # Fz catches eye blinks
    band_channel_weights={
        # Posterior alpha boost (Pz/PO7/Oz/PO8 dominate).
        "alpha": (0.4, 0.5, 0.7, 0.5, 1.1, 1.3, 1.4, 1.3),
        # Frontal beta slight boost (Fz).
        "beta":  (1.2, 1.0, 1.1, 1.0, 0.9, 0.8, 0.8, 0.8),
    },
)


DOG_PROFILE = DeviceProfile(
    name="DOG",
    channel_names=("Fp", "F3", "F4", "Cz", "T3", "T4", "O1", "O2"),
    sample_rate_hz=250,
    pink_noise_amplitude_uv=10.0,
    line_noise_amplitude_uv=2.0,
    line_noise_hz=50.0,
    bands=(
        BandSpec("delta", 2.5, 2.0, 8.0),
        BandSpec("theta", 6.5, 2.5, 12.0),  # dogs: stronger theta
        BandSpec("alpha", 11.0, 2.0, 6.0),   # weaker than humans
        BandSpec("beta", 22.0, 7.0, 7.0),
        BandSpec("gamma", 50.0, 15.0, 5.0),  # more gamma
    ),
    blink_rate_hz=0.10,
    blink_amplitude_uv=60.0,
    emg_burst_rate_hz=0.50,  # dogs move more
    emg_burst_amplitude_uv=35.0,
    motion_drift_amplitude_uv=12.0,
    accelerometer_baseline_g=(0.1, 0.0, 0.9),
    accelerometer_noise_g=0.03,  # restless
    gyroscope_noise_dps=4.0,
    battery_start_percent=95.0,
    battery_drain_percent_per_hour=22.0,
    blink_channels=(0, 1, 2),
    band_channel_weights={
        "theta": (1.1, 1.2, 1.2, 1.4, 1.0, 1.0, 0.9, 0.9),
        "gamma": (0.8, 0.9, 0.9, 1.0, 1.1, 1.1, 1.2, 1.2),
    },
)


ALL_PROFILES: dict[str, DeviceProfile] = {
    "HUMAN": HUMAN_PROFILE,
    "DOG": DOG_PROFILE,
}
