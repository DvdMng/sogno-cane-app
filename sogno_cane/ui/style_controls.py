"""REGOLAZIONI of the 3D view: contrast, thickness of the lines, seconds
in view, height of the waves, fabric.

One set for STUDIO and for the projection (the projector shows what is set
on the PC, as with the point of view), applied at the next frame, kept at
the next start.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from sogno_cane.ui.landscape_view import ViewStyle
from sogno_cane.ui.theme import PANEL_FILL, TEXT, TEXT_DIM


def _number(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


# name, caption, slider units per unit of value, how the value reads, tip
_ROWS = (
    ("contrast", "contrasto", 100,
     lambda v: f"{v * 100:.0f} %",
     "Quanto le linee risaltano sul fondo."),
    ("thickness", "spessore linee", 10,
     lambda v: _number(v, 1) + " px",
     "Lo spessore di onde, fili e sezioni."),
    ("seconds", "campo visibile", 1,
     lambda v: f"{v:.0f} s",
     "Quanti secondi di EEG si vedono, dal presente in primo piano al\n"
     "passato in fondo. Cambiandolo la vista riparte da capo."),
    ("gain", "ampiezza", 100,
     lambda v: _number(v, 2) + " ×",
     "L'altezza delle onde."),
    ("fabric", "tessuto", 100,
     lambda v: "solo le onde" if v <= 0 else f"{v * 100:.0f} %",
     "Quanto si vedono i fili fra le onde (forma MESH); a zero\n"
     "restano solo le onde."),
)


class ViewStyleControls(QDialog):
    """The REGOLAZIONI panel: every change applies at once."""

    changed = Signal()

    def __init__(self, style: ViewStyle, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Regolazioni della vista 3D")
        self.setModal(False)
        self.setObjectName("ViewStyleControls")
        self.setStyleSheet(
            f"#ViewStyleControls {{ background: {PANEL_FILL}; }}"
            f" QLabel {{ color: {TEXT}; }}"
        )
        self._style = style
        v = QVBoxLayout(self)
        intro = QLabel("Valgono per la vista 3D in STUDIO e in proiezione; "
                       "ogni modifica si vede subito.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {TEXT_DIM};")
        v.addWidget(intro)
        form = QFormLayout()
        self._rows: dict[str, tuple] = {}
        for name, caption, units, show, tip in _ROWS:
            lo, hi = ViewStyle.RANGES[name]
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(round(lo * units), round(hi * units))
            slider.setMinimumWidth(220)
            value = QLabel()
            value.setMinimumWidth(96)
            for w in (slider, value):
                w.setToolTip(tip)
            row = QHBoxLayout()
            row.addWidget(slider, 1)
            row.addWidget(value)
            form.addRow(caption, row)
            self._rows[name] = (slider, value, units, show)
            slider.valueChanged.connect(
                lambda pos, n=name: self._moved(n, pos))
        v.addLayout(form)
        reset = QPushButton("VALORI INIZIALI")
        reset.setToolTip("Contrasto, spessore, campo visibile, ampiezza e "
                         "tessuto come all'inizio.")
        reset.clicked.connect(self.reset)
        v.addWidget(reset, 0, Qt.AlignmentFlag.AlignRight)
        self._load()

    def set_style(self, style: ViewStyle) -> None:
        self._style = style
        self._load()

    def _load(self) -> None:
        for name, (slider, value, units, show) in self._rows.items():
            current = getattr(self._style, name)
            slider.blockSignals(True)
            slider.setValue(round(current * units))
            slider.blockSignals(False)
            value.setText(show(current))

    def _moved(self, name: str, pos: int) -> None:
        slider, value, units, show = self._rows[name]
        self._style.set(name, pos / units)
        value.setText(show(getattr(self._style, name)))
        self.changed.emit()

    def reset(self) -> None:
        self._style.reset()
        self._load()
        self.changed.emit()


__all__ = ["ViewStyleControls"]
