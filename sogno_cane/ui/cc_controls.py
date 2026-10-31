"""Controls for how a CC moves: its morbidezza and its output range.

Both were spin boxes over raw numbers, a smoothing coefficient from 0 to
0,99 and two bounds from 0 to 127: correct, but mechanical, and the
coefficient said nothing about what one hears. Here the morbidezza is set
as the time the CC takes to follow the signal, on a slider with a
logarithmic scale, and the output range as a span on a bar from 0 to 127;
the exact values sit beside them and can be typed. The voices keep their
own fields (``smoothing``, ``out_min``, ``out_max``), so saved
configurations read as before.

None of these controls takes the mouse wheel unless it has been clicked:
scrolling the page over them must never change a CC by accident.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QLocale, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QWidget,
)

from sogno_cane.ui.theme import (
    ACCENT_CYAN,
    ACCENT_PINK,
    BORDER_DIM,
    TEXT,
    TEXT_DIM,
)

# The engine re-reads the EEG every 0.1 s: the smoothing applies per step.
HOP_S = 0.1
TAU_MIN = 0.1        # s: the shortest morbidezza on the slider
TAU_MAX = 30.0       # s: the longest
_STEPS = 1000        # slider positions: about 0,6 % of the time per step
_ITALIAN = QLocale(QLocale.Language.Italian, QLocale.Country.Italy)

SMOOTH_TIP = (
    "Il tempo con cui il CC segue il segnale.\n"
    "Breve: il parametro reagisce subito e segue ogni oscillazione\n"
    "dell'EEG. Lungo: si muove lento e continuo, senza scatti.\n"
    "È il tempo in cui il CC compie circa due terzi di uno spostamento.\n"
    "Si trascina il cursore oppure si scrive il valore."
)

RANGE_TIP = (
    "Il tratto del parametro che il CC percorre, da 0 a 127.\n"
    "Si trascinano le due estremità della barra oppure si scrivono i\n"
    "valori: con 30–90 il filtro o la manopola del synth si muove solo\n"
    "fra 30 e 90. Per invertire il verso c'è «inverti»."
)


def tau_from_smoothing(smoothing: float) -> float:
    """Seconds for about two thirds of a change (0 = none)."""
    s = float(smoothing)
    if s <= 0.0:
        return 0.0
    return -HOP_S / math.log(min(s, 0.9999))


def smoothing_from_tau(tau: float) -> float:
    """The voice's coefficient for a time in seconds."""
    return 0.0 if tau <= 0.0 else math.exp(-HOP_S / float(tau))


def _position(tau: float) -> int:
    """Slider position: 0 = none, then a logarithmic scale of times."""
    if tau <= 0.0:
        return 0
    f = math.log(max(tau, TAU_MIN) / TAU_MIN) / math.log(TAU_MAX / TAU_MIN)
    return 1 + round(min(1.0, f) * (_STEPS - 1))


def _tau_at(position: int) -> float:
    if position <= 0:
        return 0.0
    return TAU_MIN * (TAU_MAX / TAU_MIN) ** ((position - 1) / (_STEPS - 1))


class _Slider(QSlider):
    """Takes the wheel only once clicked."""

    def __init__(self) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _SecondsBox(QDoubleSpinBox):
    """The morbidezza in seconds, without arrows: typed, or stepped with
    the wheel and the arrow keys by about 6 %, as fine at 0,2 s as at
    20 s."""

    def __init__(self) -> None:
        super().__init__()
        self.setLocale(_ITALIAN)
        self.setRange(0.0, TAU_MAX)
        self.setDecimals(2)
        self.setSuffix(" s")
        self.setSpecialValueText("nessuna")
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignmentFlag.AlignRight
                          | Qt.AlignmentFlag.AlignVCenter)
        self.setKeyboardTracking(False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt name
        digits = 2 if value < 1.0 else 1 if value < 10.0 else 0
        return self.locale().toString(float(value), "f", digits)

    def stepBy(self, steps: int) -> None:  # noqa: N802 - Qt name
        value = self.value()
        if value < TAU_MIN:
            value = TAU_MIN if steps > 0 else 0.0
        else:
            value *= 1.06 ** steps
            if value < TAU_MIN * 0.999:
                value = 0.0
        self.setValue(min(TAU_MAX, value))

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _NumberBox(QSpinBox):
    """An end of the range, 0–127, without arrows; the wheel once
    clicked."""

    def __init__(self, value: int) -> None:
        super().__init__()
        self.setRange(0, 127)
        self.setValue(int(value))
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignmentFlag.AlignRight
                          | Qt.AlignmentFlag.AlignVCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedWidth(46)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class SmoothingControl(QWidget):
    """«morbidezza»: a slider with a logarithmic scale of times (none, then
    0,1 s to 30 s) and the time itself beside it."""

    def __init__(self, obj, set_safe, default: float, note: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._obj, self._set = obj, set_safe
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.slider = _Slider()
        self.slider.setRange(0, _STEPS)
        self.slider.setPageStep(50)
        self.slider.setFixedSize(190, 22)
        self.field = _SecondsBox()
        self.field.setFixedWidth(78)
        tau = tau_from_smoothing(getattr(obj, "smoothing", default))
        self.slider.setValue(_position(tau))
        self.field.setValue(tau)
        tip = SMOOTH_TIP + ("\n\n" + note if note else "")
        for w in (self.slider, self.field):
            w.setToolTip(tip)
        self.slider.valueChanged.connect(self._from_slider)
        self.field.valueChanged.connect(self._from_field)
        row.addWidget(self.slider)
        row.addWidget(self.field)

    def seconds(self) -> float:
        return tau_from_smoothing(getattr(self._obj, "smoothing", 0.0))

    def _from_slider(self, position: int) -> None:
        tau = _tau_at(position)
        self.field.blockSignals(True)
        self.field.setValue(tau)
        self.field.blockSignals(False)
        self._set(self._obj, "smoothing", smoothing_from_tau(tau))

    def _from_field(self, tau: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(_position(tau))
        self.slider.blockSignals(False)
        self._set(self._obj, "smoothing", smoothing_from_tau(tau))


class RangeBar(QWidget):
    """A span on a bar from 0 to 127. Drag either end, or click on the bar
    to bring the nearer end there; the arrow keys and, once clicked, the
    wheel move the last end touched by one (with Shift by ten). The ends
    never cross."""

    changed = Signal(int, int)

    _PAD = 6.0             # px: room for the handles at the ends
    _HANDLE = (8.0, 16.0)  # px

    def __init__(self, lo: int = 0, hi: int = 127,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lo, self._hi = 0, 127
        self._active = 1                 # the end last touched: 0 low, 1 high
        self._dragging = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedSize(190, 22)
        self.set_values(lo, hi)

    def values(self) -> tuple[int, int]:
        return self._lo, self._hi

    def set_values(self, lo: int, hi: int) -> None:
        """From outside (the number boxes): no signal back."""
        lo = max(0, min(127, int(lo)))
        hi = max(lo, min(127, int(hi)))
        if (lo, hi) != (self._lo, self._hi):
            self._lo, self._hi = lo, hi
            self.update()

    def _x(self, value: float) -> float:
        span = max(1.0, self.width() - 2 * self._PAD)
        return self._PAD + span * value / 127.0

    def _value_at(self, x: float) -> int:
        span = max(1.0, self.width() - 2 * self._PAD)
        return max(0, min(127, round((x - self._PAD) / span * 127.0)))

    def _move(self, end: int, value: int) -> None:
        value = max(0, min(127, int(value)))
        if end == 0:
            lo, hi = min(value, self._hi), self._hi
        else:
            lo, hi = self._lo, max(value, self._lo)
        if (lo, hi) != (self._lo, self._hi):
            self._lo, self._hi = lo, hi
            self.update()
            self.changed.emit(lo, hi)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt name
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        value = self._value_at(event.position().x())
        if self._lo == self._hi:
            end = 0 if value < self._lo else 1 if value > self._hi \
                else self._active
        elif value <= self._lo:
            end = 0
        elif value >= self._hi:
            end = 1
        else:
            end = 0 if value - self._lo < self._hi - value else 1
        self._active = end
        self._dragging = True
        self._move(end, value)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self._dragging:
            self._move(self._active, self._value_at(event.position().x()))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt name
        self._dragging = False

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt name
        step = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier \
            else 1
        now = self._lo if self._active == 0 else self._hi
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Down):
            self._move(self._active, now - step)
        elif event.key() in (Qt.Key.Key_Right, Qt.Key.Key_Up):
            self._move(self._active, now + step)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt name
        if not self.hasFocus():
            event.ignore()
            return
        notches = event.angleDelta().y() // 120
        if notches:
            now = self._lo if self._active == 0 else self._hi
            self._move(self._active, now + notches)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt name
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        mid = self.height() / 2.0
        on = self.isEnabled()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(BORDER_DIM))
        x0, x1 = self._x(0), self._x(127)
        p.drawRoundedRect(QRectF(x0, mid - 2.0, x1 - x0, 4.0), 2.0, 2.0)
        a, b = self._x(self._lo), self._x(self._hi)
        p.setBrush(QColor(ACCENT_CYAN if on else TEXT_DIM))
        p.drawRoundedRect(QRectF(a, mid - 2.0, max(0.0, b - a), 4.0),
                          2.0, 2.0)
        w, h = self._HANDLE
        for end, x in ((0, a), (1, b)):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(ACCENT_PINK if on else TEXT_DIM))
            p.drawRoundedRect(QRectF(x - w / 2, mid - h / 2, w, h), 2.0, 2.0)
            if self.hasFocus() and end == self._active:
                p.setPen(QPen(QColor(TEXT), 1.0))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(
                    QRectF(x - w / 2 - 2, mid - h / 2 - 2, w + 4, h + 4),
                    3.0, 3.0)
        p.end()


class RangeControl(QWidget):
    """«uscita»: the bar and, beside it, its two ends as numbers."""

    def __init__(self, obj, set_safe, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._obj, self._set = obj, set_safe
        lo = max(0, min(127, int(getattr(obj, "out_min", 0))))
        hi = max(lo, min(127, int(getattr(obj, "out_max", 127))))
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.bar = RangeBar(lo, hi)
        self.low = _NumberBox(lo)
        self.high = _NumberBox(hi)
        # Never crossed: the low end stays at or below the high one.
        self.low.setMaximum(hi)
        self.high.setMinimum(lo)
        for w in (self.bar, self.low, self.high):
            w.setToolTip(RANGE_TIP)
        dash = QLabel("–")
        dash.setStyleSheet(f"color: {TEXT_DIM};")
        self.bar.changed.connect(self._from_bar)
        self.low.valueChanged.connect(self._from_low)
        self.high.valueChanged.connect(self._from_high)
        row.addWidget(self.bar)
        row.addSpacing(2)
        row.addWidget(self.low)
        row.addWidget(dash)
        row.addWidget(self.high)

    def _from_bar(self, lo: int, hi: int) -> None:
        for box in (self.low, self.high):
            box.blockSignals(True)
        self.low.setMaximum(127)
        self.high.setMinimum(0)
        self.low.setValue(lo)
        self.high.setValue(hi)
        self.low.setMaximum(hi)
        self.high.setMinimum(lo)
        for box in (self.low, self.high):
            box.blockSignals(False)
        self._set(self._obj, "out_min", lo)
        self._set(self._obj, "out_max", hi)

    def _from_low(self, value: int) -> None:
        self._set(self._obj, "out_min", int(value))
        self.high.setMinimum(int(value))
        self.bar.set_values(int(value), self.high.value())

    def _from_high(self, value: int) -> None:
        self._set(self._obj, "out_max", int(value))
        self.low.setMaximum(int(value))
        self.bar.set_values(self.low.value(), int(value))


__all__ = [
    "RANGE_TIP",
    "SMOOTH_TIP",
    "RangeBar",
    "RangeControl",
    "SmoothingControl",
    "smoothing_from_tau",
    "tau_from_smoothing",
]
