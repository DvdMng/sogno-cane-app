"""g.tec Unicorn Hybrid Black packet layout.

The Unicorn Python API delivers samples as a 2-D float32 array of shape
``(n_samples, 17)``. We mirror that structure exactly so downstream code can
be ported to a real device without changes.

Column layout (matches Unicorn Suite Hybrid Black Python API v1.0):
    0..7   EEG channels 1..8 in microvolts
    8..10  Accelerometer X, Y, Z in g
    11..13 Gyroscope X, Y, Z in deg/s
    14     Battery level in percent (0..100)
    15     Sample counter (monotonic, 32-bit modulo)
    16     Validation indicator (1.0 valid, 0.0 invalid)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EEG_CHANNELS: int = 8
ACC_CHANNELS: int = 3
GYR_CHANNELS: int = 3
PACKET_VALUES: int = 17

EEG_SLICE = slice(0, 8)
ACC_SLICE = slice(8, 11)
GYR_SLICE = slice(11, 14)
BATTERY_INDEX: int = 14
COUNTER_INDEX: int = 15
VALIDATION_INDEX: int = 16

DEFAULT_SAMPLE_RATE_HZ: int = 250
DEFAULT_CHANNEL_NAMES: tuple[str, ...] = (
    "Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8",
)


@dataclass(frozen=True)
class UnicornPacket:
    """A chunk of samples emitted by the simulator.

    Attributes
    ----------
    samples
        Array of shape ``(n_samples, 17)`` and dtype ``float32``.
    timestamp
        Wall-clock seconds of the first sample (monotonic clock).
    sample_rate
        Sample rate the packet was generated at (Hz).
    """

    samples: np.ndarray
    timestamp: float
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim != 2 or self.samples.shape[1] != PACKET_VALUES:
            raise ValueError(
                f"Unicorn packet must have shape (N, {PACKET_VALUES}), "
                f"got {self.samples.shape}"
            )
        if self.samples.dtype != np.float32:
            raise ValueError(
                f"Unicorn packet must be float32, got {self.samples.dtype}"
            )

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def eeg(self) -> np.ndarray:
        return self.samples[:, EEG_SLICE]

    @property
    def accelerometer(self) -> np.ndarray:
        return self.samples[:, ACC_SLICE]

    @property
    def gyroscope(self) -> np.ndarray:
        return self.samples[:, GYR_SLICE]

    @property
    def battery(self) -> np.ndarray:
        return self.samples[:, BATTERY_INDEX]

    @property
    def counter(self) -> np.ndarray:
        return self.samples[:, COUNTER_INDEX]

    @property
    def validation(self) -> np.ndarray:
        return self.samples[:, VALIDATION_INDEX]
