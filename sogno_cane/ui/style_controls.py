"""REGOLAZIONI of the 3D view: contrast, thickness of the lines, seconds
in view, height of the waves, fabric.

One set for STUDIO and for the projection (the projector shows what is set
on the PC, as with the point of view), applied at the next frame, kept at
the next start. The rows are sliders with their value in words; STUDIO
shows them in a panel of their own, the projection's panel among its other
settings. Only what acts is shown: the fabric only for the MESH shape.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
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


def number(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


class SliderRow(QObject):
    """A setting as a slider with its value in words beside it: the field
    of a form row (``field``). ``moved`` gives the value, not the slider
    position."""

    moved = Signal(float)

    def __init__(self, lo: float, hi: float, units: int, show, tip: str,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._units = units
        self._show = show
        self.field = QWidget()
        row = QHBoxLayout(self.field)
        row.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(round(lo * units), round(hi * units))
        self.slider.setMinimumWidth(180)
        self.label = QLabel()
        self.label.setMinimumWidth(92)
        for w in (self.slider, self.label):
            w.setToolTip(tip)
        row.addWidget(self.slider, 1)
        row.addWidget(self.label)
        self.slider.valueChanged.connect(self._on_slider)

    def _on_slider(self, pos: int) -> None:
        self.show_value(pos / self._units)
        self.moved.emit(pos / self._units)

    def value(self) -> float:
        return self.slider.value() / self._units

    def set_value(self, value: float) -> None:
        """Show ``value`` (within the range) without reporting it as a
        change."""
        self.slider.blockSignals(True)
        self.slider.setValue(round(value * self._units))
        self.slider.blockSignals(False)
        self.show_value(self.value())

    def show_value(self, value: float) -> None:
        self.label.setText(self._show(value))


# name, caption, slider units per unit of value, how the value reads, tip
_ROWS = (
    ("contrast", "contrasto", 100,
     lambda v: f"{v * 100:.0f} %",
     "Quanto le linee risaltano sul fondo."),
    ("thickness", "spessore linee", 10,
     lambda v: number(v, 1) + " px",
     "Lo spessore di onde, fili e sezioni."),
    ("seconds", "campo visibile", 1,
     lambda v: f"{v:.0f} s",
     "Quanti secondi di EEG si vedono, dal presente in primo piano al\n"
     "passato in fondo. Ciò che è già arrivato resta."),
    ("gain", "ampiezza", 100,
     lambda v: number(v, 2) + " ×",
     "L'altezza delle onde."),
    ("fabric", "tessuto", 100,
     lambda v: "solo le onde" if v <= 0 else f"{v * 100:.0f} %",
     "Quanto si vedono i fili fra le onde; a zero restano solo le onde."),
)


class StyleRows(QObject):
    """The REGOLAZIONI as rows of a form, on a ViewStyle: every move
    applies at once (``changed``)."""

    changed = Signal()

    def __init__(self, style: ViewStyle, form: QFormLayout,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._style = style
        self._form = form
        self.rows: dict[str, SliderRow] = {}
        for name, caption, units, show, tip in _ROWS:
            lo, hi = ViewStyle.RANGES[name]
            row = SliderRow(lo, hi, units, show, tip, self)
            row.moved.connect(lambda v, n=name: self._moved(n, v))
            form.addRow(caption, row.field)
            self.rows[name] = row
        self._shown = True
        self._mesh = True
        self.load()

    def set_style(self, style: ViewStyle) -> None:
        self._style = style
        self.load()

    def load(self) -> None:
        """The sliders to the style's values (changed elsewhere)."""
        for name, row in self.rows.items():
            row.set_value(getattr(self._style, name))

    def _moved(self, name: str, value: float) -> None:
        self._style.set(name, value)
        self.rows[name].show_value(getattr(self._style, name))
        self.changed.emit()

    def reset(self) -> None:
        self._style.reset()
        self.load()
        self.changed.emit()

    def show(self, shown: bool, mesh: bool) -> None:
        """Rows on or off the form; the fabric only for the MESH shape."""
        self._shown, self._mesh = bool(shown), bool(mesh)
        for name, row in self.rows.items():
            self._form.setRowVisible(
                row.field, self._shown and (name != "fabric" or self._mesh))


class ViewStyleControls(QDialog):
    """STUDIO's REGOLAZIONI panel: every change applies at once."""

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
        v = QVBoxLayout(self)
        intro = QLabel("Valgono per la vista 3D in STUDIO e in proiezione; "
                       "ogni modifica si vede subito.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {TEXT_DIM};")
        v.addWidget(intro)
        form = QFormLayout()
        self._style_rows = StyleRows(style, form, self)
        self._style_rows.changed.connect(self.changed)
        v.addLayout(form)
        reset = QPushButton("VALORI INIZIALI")
        reset.setToolTip("Contrasto, spessore, campo visibile, ampiezza e "
                         "tessuto come all'inizio.")
        reset.clicked.connect(self.reset)
        v.addWidget(reset, 0, Qt.AlignmentFlag.AlignRight)

    @property
    def rows(self) -> dict[str, SliderRow]:
        return self._style_rows.rows

    def set_style(self, style: ViewStyle) -> None:
        self._style_rows.set_style(style)

    def load_style(self) -> None:
        self._style_rows.load()

    def set_shape(self, shape: str) -> None:
        """STUDIO's shape: the fabric only shows on MESH."""
        self._style_rows.show(True, shape == "mesh")
        self.adjustSize()

    def reset(self) -> None:
        self._style_rows.reset()


__all__ = ["SliderRow", "StyleRows", "ViewStyleControls", "number"]
