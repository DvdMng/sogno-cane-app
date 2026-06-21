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

        self.setMinimumHeight(160)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)

    def push_packet(self, packet: UnicornPacket) -> None:
        eeg = packet.eeg
        for ch, buf in enumerate(self._buffers):
            buf.extend(eeg[:, ch].astype(float).tolist())

    def repaint_traces(self) -> None:
        for ch, buf in enumerate(self._buffers):
            data = np.asarray(buf, dtype=np.float32) + ch * self._stride_uv
            self._curves[ch].setData(self._x_axis, data)
