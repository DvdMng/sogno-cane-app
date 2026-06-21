"""Band power bar display."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget
import pyqtgraph as pg

from sogno_cane.eeg.bands import BAND_RANGES_HZ
from sogno_cane.ui.theme import ACCENT_CYAN, ACCENT_PINK, TEXT_DIM

# Gradient between pink and cyan across the bands - calm and informative.
_BAND_COLORS = {
    "delta": "#FF3D8A",
    "theta": "#D44CB7",
    "alpha": "#A65AE2",
    "beta":  "#7B82FF",
    "gamma": "#5BE9FF",
}


class SpectrumWidget(QWidget):
    """Vertical bars showing mean log-power per band."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        pg.setConfigOption("background", (10, 4, 24, 0))
        pg.setConfigOption("foreground", TEXT_DIM)

        self._plot = pg.PlotWidget()
        self._plot.setBackground(None)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.setYRange(-1, 5)
        self._band_names = list(BAND_RANGES_HZ.keys())

        ax_bottom = self._plot.getAxis("bottom")
        ax_bottom.setTicks([
            [(i, name.upper()) for i, name in enumerate(self._band_names)]
        ])
        ax_bottom.setStyle(tickFont=None)
        self._plot.getAxis("left").setStyle(showValues=True)

        brushes = [_BAND_COLORS.get(b, ACCENT_CYAN) for b in self._band_names]
        self._bars = pg.BarGraphItem(
            x=np.arange(len(self._band_names)),
            height=np.zeros(len(self._band_names)),
            width=0.7,
            brushes=brushes,
        )
        self._plot.addItem(self._bars)

        # Keep the spectrum a compact strip so the waveform stays dominant.
        self.setMinimumHeight(70)
        self.setMaximumHeight(104)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)

    def update_powers(self, mean_log_powers: dict[str, float]) -> None:
        heights = np.array(
            [mean_log_powers.get(b, -1.0) for b in self._band_names],
            dtype=np.float32,
        )
        self._bars.setOpts(height=heights)
