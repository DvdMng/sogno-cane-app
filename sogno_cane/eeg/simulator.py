"""Hyperrealistic EEG simulator producing Unicorn Hybrid Black packets.

Each call to :py:meth:`EEGSimulator.next_packet` returns a deterministic chunk
of N samples advanced by ``N / sample_rate_hz`` seconds of simulation time.

Signal model per channel:
    s(t) = pink_noise(t)
         + sum over bands of band_oscillation(t, center, sigma)
         + sin(2 pi f_line t) * line_amp
         + blink_artifact(t)    (frontal channels only)
         + emg_burst(t)
         + slow_drift(t)

All amplitudes are in microvolts; sampling resolution mirrors the Unicorn's
24-bit ADC over its full +/- 750 mV input range (~0.09 uV per LSB), which is
applied as a quantization step so the output is byte-faithful to a real
acquisition.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from sogno_cane.eeg.profiles import DeviceProfile, HUMAN_PROFILE
from sogno_cane.eeg.unicorn_packet import (
    ACC_SLICE,
    BATTERY_INDEX,
    COUNTER_INDEX,
    EEG_SLICE,
    GYR_SLICE,
    PACKET_VALUES,
    VALIDATION_INDEX,
    UnicornPacket,
)

# Unicorn full-scale: +/-750 mV, 24-bit signed -> 1500e3 uV / 2^24 LSBs.
UNICORN_LSB_UV: float = 1500_000.0 / (1 << 24)

# Default packet size: 25 samples per packet ~= 100 ms cadence at 250 Hz.
DEFAULT_PACKET_SAMPLES: int = 25


@dataclass
class _BandState:
    """Running phase state for one narrow-band oscillator per channel."""

    phase: np.ndarray            # shape (n_channels,)
    amplitude_env: np.ndarray    # shape (n_channels,)


class EEGSimulator:
    """Stateful generator of Unicorn-shaped EEG packets.

    Parameters
    ----------
    profile
        Subject profile (human or dog) describing spectrum and artifacts.
    packet_samples
        Number of samples returned by :py:meth:`next_packet`.
    seed
        Reproducibility seed; pass ``None`` for non-deterministic output.
    """

    def __init__(
        self,
        profile: DeviceProfile = HUMAN_PROFILE,
        packet_samples: int = DEFAULT_PACKET_SAMPLES,
        seed: Optional[int] = None,
    ) -> None:
        if packet_samples <= 0:
            raise ValueError("packet_samples must be positive")
        self._profile = profile
        self._packet_samples = packet_samples
        self._rng = np.random.default_rng(seed)
        self._n_channels = len(profile.channel_names)

        # Sample-time counter (monotonic, wraps at 2^32 like the device).
        self._sample_counter: int = 0
        # Wall-clock anchor for packet timestamps.
        self._t0: float = time.monotonic()
        # Cumulative simulation time in seconds.
        self._elapsed_s: float = 0.0
        # Current battery percentage (drains slowly).
        self._battery: float = profile.battery_start_percent

        # Pink-noise state via simple "Voss-McCartney" filter
        # implemented with a single-pole AR(1) per channel for stability.
        self._pink_state: np.ndarray = np.zeros(
            self._n_channels, dtype=np.float64
        )

        # Per-band oscillator phase and slowly-varying amplitude envelope.
        self._band_states: list[_BandState] = []
        for band in profile.bands:
            self._band_states.append(
                _BandState(
                    phase=self._rng.uniform(
                        0.0, 2 * np.pi, size=self._n_channels
                    ),
                    amplitude_env=np.ones(self._n_channels, dtype=np.float64),
                )
            )

        # State for slow motion drift (random walk per channel).
        self._drift_state: np.ndarray = np.zeros(
            self._n_channels, dtype=np.float64
        )

        # Pending EMG burst envelope (decays exponentially).
        self._emg_envelope: float = 0.0
        # Pending blink envelope.
        self._blink_envelope: float = 0.0
        # Phase of mains line frequency.
        self._line_phase: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def profile(self) -> DeviceProfile:
        return self._profile

    @property
    def sample_rate_hz(self) -> int:
        return self._profile.sample_rate_hz

    @property
    def packet_samples(self) -> int:
        return self._packet_samples

    @property
    def elapsed_seconds(self) -> float:
        return self._elapsed_s

    def reset(self) -> None:
        self._sample_counter = 0
        self._elapsed_s = 0.0
        self._battery = self._profile.battery_start_percent
        self._t0 = time.monotonic()
        self._pink_state.fill(0.0)
        self._drift_state.fill(0.0)
        self._emg_envelope = 0.0
        self._blink_envelope = 0.0
        self._line_phase = 0.0

    def next_packet(
        self,
        n_samples: Optional[int] = None,
        wall_clock: Optional[float] = None,
    ) -> UnicornPacket:
        """Return the next packet of EEG + IMU samples.

        Parameters
        ----------
        n_samples
            Override the default packet size for this call.
        wall_clock
            Override the timestamp; otherwise ``time.monotonic()`` is used.
        """
        n = n_samples if n_samples is not None else self._packet_samples
        if n <= 0:
            raise ValueError("n_samples must be positive")

        fs = self._profile.sample_rate_hz
        dt = 1.0 / fs
        out = np.zeros((n, PACKET_VALUES), dtype=np.float32)

        # ---- 1. EEG channels --------------------------------------------
        eeg = self._generate_eeg(n)
        out[:, EEG_SLICE] = eeg.astype(np.float32)

        # ---- 2. Accelerometer -------------------------------------------
        baseline = np.asarray(
            self._profile.accelerometer_baseline_g, dtype=np.float64
        )
        acc_noise = self._rng.normal(
            0.0, self._profile.accelerometer_noise_g, size=(n, 3)
        )
        out[:, ACC_SLICE] = (baseline[None, :] + acc_noise).astype(np.float32)

        # ---- 3. Gyroscope -----------------------------------------------
        gyr_noise = self._rng.normal(
            0.0, self._profile.gyroscope_noise_dps, size=(n, 3)
        )
        out[:, GYR_SLICE] = gyr_noise.astype(np.float32)

        # ---- 4. Battery -------------------------------------------------
        drain = self._profile.battery_drain_percent_per_hour * (n * dt) / 3600.0
        self._battery = max(0.0, self._battery - drain)
        out[:, BATTERY_INDEX] = np.float32(self._battery)

        # ---- 5. Counter -------------------------------------------------
        counter_vals = (
            self._sample_counter
            + np.arange(n, dtype=np.int64)
        ) % (1 << 32)
        out[:, COUNTER_INDEX] = counter_vals.astype(np.float32)
        self._sample_counter = int(
            (self._sample_counter + n) % (1 << 32)
        )

        # ---- 6. Validation ----------------------------------------------
        out[:, VALIDATION_INDEX] = np.float32(1.0)

        # ---- 7. Time bookkeeping ----------------------------------------
        self._elapsed_s += n * dt
        timestamp = (
            wall_clock if wall_clock is not None else time.monotonic()
        )

        return UnicornPacket(
            samples=out,
            timestamp=float(timestamp),
            sample_rate=fs,
        )

    # ------------------------------------------------------------------ #
    # Internal signal generators                                         #
    # ------------------------------------------------------------------ #

    def _generate_eeg(self, n: int) -> np.ndarray:
        fs = self._profile.sample_rate_hz
        dt = 1.0 / fs
        nc = self._n_channels
        out = np.zeros((n, nc), dtype=np.float64)

        # ----- pink noise via leaky integrator on white noise -----
        # AR(1) coefficient gives an approximately 1/f spectrum below fs/2.
        a = 0.97
        gain = self._profile.pink_noise_amplitude_uv * np.sqrt(1 - a * a)
        white = self._rng.normal(0.0, 1.0, size=(n, nc))
        for i in range(n):
            self._pink_state = a * self._pink_state + white[i]
            out[i] += gain * self._pink_state

        # ----- narrow-band oscillators -----
        t_axis = np.arange(n, dtype=np.float64) * dt
        for band, state in zip(self._profile.bands, self._band_states):
            # Slow amplitude modulation (1 Hz Gaussian random walk, clipped).
            mod = self._rng.normal(0.0, 0.05, size=nc)
            state.amplitude_env = np.clip(
                state.amplitude_env + mod, 0.4, 1.6
            )
            # Tiny frequency jitter so it never sounds perfectly periodic.
            f_jitter = self._rng.normal(
                0.0, band.bandwidth_hz * 0.05, size=nc
            )
            f = band.center_hz + f_jitter
            phase_inc = 2.0 * np.pi * f * dt
            # Per-channel weights (e.g. posterior alpha boost).
            w = np.asarray(
                self._profile.band_channel_weights.get(
                    band.name, (1.0,) * nc
                ),
                dtype=np.float64,
            )
            amp = band.amplitude_uv * w * state.amplitude_env  # (nc,)
            phases = (
                state.phase[None, :]
                + np.cumsum(np.broadcast_to(phase_inc, (n, nc)), axis=0)
            )
            out += amp[None, :] * np.sin(phases)
            state.phase = (phases[-1] + phase_inc) % (2.0 * np.pi)

        # ----- mains line noise -----
        phase_inc = 2.0 * np.pi * self._profile.line_noise_hz * dt
        line = self._profile.line_noise_amplitude_uv * np.sin(
            self._line_phase + np.arange(n) * phase_inc
        )
        self._line_phase = (self._line_phase + n * phase_inc) % (2 * np.pi)
        out += line[:, None]

        # ----- slow motion drift (per-channel random walk + decay) -----
        drift_amp = self._profile.motion_drift_amplitude_uv
        if drift_amp > 0.0:
            step = self._rng.normal(0.0, drift_amp * 0.05, size=(n, nc))
            for i in range(n):
                self._drift_state = 0.995 * self._drift_state + step[i]
                out[i] += self._drift_state

        # ----- eye blinks (Poisson-triggered exponential bumps) -----
        blink_prob = self._profile.blink_rate_hz * dt
        for i in range(n):
            if self._rng.random() < blink_prob:
                self._blink_envelope = self._profile.blink_amplitude_uv
            if self._blink_envelope > 0.01:
                self._blink_envelope *= 0.92  # ~250 ms decay
                for ch in self._profile.blink_channels:
                    out[i, ch] += self._blink_envelope

        # ----- EMG bursts (Poisson-triggered noisy bumps) -----
        emg_prob = self._profile.emg_burst_rate_hz * dt
        for i in range(n):
            if self._rng.random() < emg_prob:
                self._emg_envelope = self._profile.emg_burst_amplitude_uv
            if self._emg_envelope > 0.01:
                self._emg_envelope *= 0.85
                # EMG = high-freq noise scaled by envelope, on all channels
                out[i] += self._rng.normal(
                    0.0, self._emg_envelope * 0.6, size=nc
                )

        # ----- quantize to Unicorn ADC LSB -----
        return np.round(out / UNICORN_LSB_UV) * UNICORN_LSB_UV
