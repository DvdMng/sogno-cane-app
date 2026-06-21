"""Packet sources: the engine clocks any object that yields Unicorn packets.

This abstraction lets the realtime engine run from either the synthetic
:class:`~sogno_cane.eeg.simulator.EEGSimulator` or a real recording loaded
from disk (CSV/EDF/BDF/npz) via :class:`ArrayPlaybackSource`, with no change
to the MIDI pipeline downstream.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from sogno_cane.eeg.unicorn_packet import (
    BATTERY_INDEX,
    COUNTER_INDEX,
    EEG_CHANNELS,
    EEG_SLICE,
    PACKET_VALUES,
    VALIDATION_INDEX,
    UnicornPacket,
)


@runtime_checkable
class PacketSource(Protocol):
    """Anything the engine can clock to obtain packets."""

    sample_rate_hz: int
    packet_samples: int

    def next_packet(self) -> Optional[UnicornPacket]:
        ...

    def reset(self) -> None:
        ...


class ArrayPlaybackSource:
    """Stream a fixed ``(n_samples, n_channels)`` EEG matrix as packets.

    Channels are padded or truncated to the Unicorn's 8 EEG columns; IMU,
    battery, counter and validation columns are synthesised so downstream
    code sees a normal :class:`UnicornPacket`. Playback can loop.
    """

    def __init__(
        self,
        eeg: np.ndarray,
        sample_rate_hz: float,
        packet_samples: int = 25,
        loop: bool = True,
    ) -> None:
        eeg = np.asarray(eeg, dtype=np.float32)
        if eeg.ndim == 1:
            eeg = eeg[:, None]
        n, nc = eeg.shape
        fitted = np.zeros((n, EEG_CHANNELS), dtype=np.float32)
        k = min(nc, EEG_CHANNELS)
        fitted[:, :k] = eeg[:, :k]
        self._eeg = fitted
        self._n = n
        # Keep a float rate for sub-Hz precision in timing math; only fall
        # back to 250 when the input is genuinely non-positive (not merely
        # a small fractional rate that would round to 0).
        self.sample_rate_hz = (
            float(sample_rate_hz) if sample_rate_hz and sample_rate_hz > 0
            else 250.0
        )
        self.packet_samples = max(1, int(packet_samples))
        self.loop = loop
        self._pos = 0
        self._counter = 0
        self._finished = False

    @property
    def n_samples(self) -> int:
        return self._n

    @property
    def position_seconds(self) -> float:
        return self._pos / float(self.sample_rate_hz)

    @property
    def duration_seconds(self) -> float:
        return self._n / float(self.sample_rate_hz)

    @property
    def is_finished(self) -> bool:
        return self._finished

    def reset(self) -> None:
        self._pos = 0
        self._counter = 0
        self._finished = False

    def next_packet(self) -> Optional[UnicornPacket]:
        if self._n == 0:
            return None
        if self._pos >= self._n:
            if not self.loop:
                self._finished = True
                return None
            self._pos = 0

        n = min(self.packet_samples, self._n - self._pos)
        block = self._eeg[self._pos:self._pos + n]
        self._pos += n

        out = np.zeros((n, PACKET_VALUES), dtype=np.float32)
        out[:, EEG_SLICE] = block
        out[:, BATTERY_INDEX] = np.float32(100.0)
        counter_vals = (self._counter + np.arange(n)) % (1 << 32)
        out[:, COUNTER_INDEX] = counter_vals.astype(np.float32)
        self._counter = int((self._counter + n) % (1 << 32))
        out[:, VALIDATION_INDEX] = np.float32(1.0)

        return UnicornPacket(
            samples=out,
            timestamp=self._pos / float(self.sample_rate_hz),
            sample_rate=self.sample_rate_hz,
        )
