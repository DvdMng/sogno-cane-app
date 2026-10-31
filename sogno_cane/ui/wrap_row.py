"""A row of controls that wraps instead of squeezing.

A QHBoxLayout given less room than its minimum (a maximized window on a
small or scaled screen, a window snapped to half the screen) narrows its
widest items below their text: «● REGISTRA UOMO+CANE» came out cut on both
sides. This row never narrows an item: what does not fit on a line moves to
the next one.
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QWidget


class WrapRow(QLayout):
    """Items left to right at their natural size, wrapping onto a new line
    when the row is too short. The last item, usually a group of controls
    kept together, sits at the right end of whichever line it lands on."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 10,
                 line_spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._gap = spacing
        self._line_gap = line_spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt name
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802 - Qt name
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):  # noqa: N802 - Qt name
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):  # noqa: N802 - Qt name
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt name
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt name
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt name
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt name
        """Everything on one line."""
        items = self._shown()
        m = self.contentsMargins()
        width = sum(i.sizeHint().width() for i in items)
        width += self._gap * max(0, len(items) - 1)
        height = max((i.sizeHint().height() for i in items), default=0)
        return QSize(width + m.left() + m.right(),
                     height + m.top() + m.bottom())

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt name
        """The widest item: everything else can go to another line."""
        items = self._shown()
        m = self.contentsMargins()
        width = max((i.minimumSize().width() for i in items), default=0)
        height = max((i.sizeHint().height() for i in items), default=0)
        return QSize(width + m.left() + m.right(),
                     height + m.top() + m.bottom())

    def lines(self) -> int:
        """How many lines the row takes at its current width."""
        return self._arrange(self.geometry(), apply=False, count=True)

    def _shown(self) -> list[QLayoutItem]:
        return [i for i in self._items if not i.isEmpty()]

    def _arrange(self, rect: QRect, apply: bool, count: bool = False) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        items = self._shown()
        rows: list[list[QLayoutItem]] = []
        x = 0
        for item in items:
            w = item.sizeHint().width()
            if rows and x + self._gap + w <= area.width():
                rows[-1].append(item)
                x += self._gap + w
            else:
                rows.append([item])
                x = w
        if count:
            return len(rows)
        y = area.y()
        for n, row in enumerate(rows):
            height = max(i.sizeHint().height() for i in row)
            if apply:
                x = area.x()
                for item in row:
                    size = item.sizeHint()
                    if item is items[-1]:
                        x = max(x, area.x() + area.width() - size.width())
                    top = y + (height - size.height()) // 2
                    item.setGeometry(QRect(QPoint(x, top), size))
                    x += size.width() + self._gap
            y += height
            if n < len(rows) - 1:
                y += self._line_gap
        return y - area.y() + m.top() + m.bottom()


__all__ = ["WrapRow"]
