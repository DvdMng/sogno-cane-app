"""Scrolling multi-channel waveform display backed by pyqtgraph.

Uses a calm two-color alternation (pink / cyan) across the 8 EEG channels
with thin anti-aliased traces.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget
import pyqtgraph as pg

from sogno_cane.eeg.profiles import DeviceProfile
from sogno_cane.eeg.unicorn_packet import UnicornPacket
from sogno_cane.ui.theme import ACCENT_CYAN, ACCENT_PINK, BG_DEEP, TEXT_DIM


class _DisplayCleaner:
    """Causal per-channel DC blocker + mains notch, for the display only.

    A real Unicorn is DC-coupled: each electrode rides on an offset of tens
    of thousands of microvolts, so raw traces land far outside the plot (it
    showed nothing at all). Dry electrodes also pick up 100-170 uV of mains
    hum. This shows what the analysis actually looks at: offset removed,
    50/100 Hz notched. Stateful and causal, so there is no edge effect; the
    blocker is seeded with the first sample, so there is no start-up spike.
    """

    def __init__(self, n_channels: int, fs: float, mains_hz: float = 50.0):
        self._r_dc = 0.995                     # ~0.2 Hz high-pass at 250 Hz
        self._x_prev: np.ndarray | None = None
        self._y_prev = np.zeros(n_channels)
        self._notches: list[tuple[float, float, float, float, float]] = []
        self._states: list[np.ndarray] = []
        if mains_hz > 0.0:
            harmonic = mains_hz
            while harmonic < fs / 2.0:
                self._notches.append(self._design_notch(harmonic, fs))
                self._states.append(np.zeros((2, n_channels)))
                harmonic += mains_hz

    @staticmethod
    def _design_notch(f0: float, fs: float, bandwidth_hz: float = 4.0):
        w0 = 2.0 * np.pi * f0 / fs
        r = 1.0 - np.pi * bandwidth_hz / fs
        c = np.cos(w0)
        gain = (1.0 - 2.0 * r * c + r * r) / (2.0 - 2.0 * c)   # unity at DC
        return (gain, -2.0 * c * gain, gain, -2.0 * r * c, r * r)

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float64)
        for i in range(x.shape[0]):
            xi = x[i].astype(np.float64)
            if self._x_prev is None:
                self._x_prev = xi.copy()
            v = xi - self._x_prev + self._r_dc * self._y_prev
            self._x_prev = xi
            self._y_prev = v
            for (b0, b1, b2, a1, a2), z in zip(self._notches, self._states):
                o = b0 * v + z[0]
                z[0] = b1 * v - a1 * o + z[1]
                z[1] = b2 * v - a2 * o
                v = o
            out[i] = v
        return out


def _palette(n: int) -> list[str]:
    # Alternate pink / cyan with subtle brightness variations.
    out: list[str] = []
    for i in range(n):
        if i % 2 == 0:
            out.append("#FF5FA0" if i // 2 % 2 == 0 else "#FF3D8A")
        else:
            out.append("#82F1FF" if i // 2 % 2 == 0 else "#5BE9FF")
    return out


class WaveformWidget(QWidget):
    def __init__(
        self,
        profile: DeviceProfile,
        seconds_visible: float = 5.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._profile = profile
        self._fs = profile.sample_rate_hz
        self._n_visible = int(seconds_visible * self._fs)

        pg.setConfigOption("background", (10, 4, 24, 0))
        pg.setConfigOption("foreground", TEXT_DIM)
        pg.setConfigOption("antialias", True)

        self._plot = pg.PlotWidget()
        self._plot.setBackground(None)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.showGrid(x=False, y=False)
        self._plot.getAxis("left").setStyle(showValues=False)
        self._plot.getAxis("bottom").setStyle(showValues=False)
        self._plot.getAxis("left").setPen(pg.mkPen(TEXT_DIM, width=0))
        self._plot.getAxis("bottom").setPen(pg.mkPen(TEXT_DIM, width=0))
        self._plot.setXRange(0, self._n_visible)
        self._stride_uv = 260.0
        self._plot.setYRange(
            -self._stride_uv,
            self._stride_uv * len(profile.channel_names),
        )

        palette = _palette(len(profile.channel_names))
        self._curves: list[pg.PlotDataItem] = []
        for i, ch_name in enumerate(profile.channel_names):
            pen = pg.mkPen(palette[i], width=1.4)
            curve = self._plot.plot([], [], pen=pen)
            self._curves.append(curve)
            label = pg.TextItem(
                text=ch_name, color=TEXT_DIM, anchor=(0, 0.5),
            )
            label.setPos(2, i * self._stride_uv)
            self._plot.addItem(label)

        self._buffers: list[deque[float]] = [
            deque([0.0] * self._n_visible, maxlen=self._n_visible)
            for _ in profile.channel_names
        ]
        self._x_axis = np.arange(self._n_visible)
        self._cleaner = _DisplayCleaner(len(profile.channel_names), self._fs)

        self.setMinimumHeight(160)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)

    def push_packet(self, packet: UnicornPacket) -> None:
        eeg = self._cleaner.process(packet.eeg)
        for ch, buf in enumerate(self._buffers):
            buf.extend(eeg[:, ch].tolist())

    def repaint_traces(self) -> None:
        for ch, buf in enumerate(self._buffers):
            data = np.asarray(buf, dtype=np.float32) + ch * self._stride_uv
            self._curves[ch].setData(self._x_axis, data)
