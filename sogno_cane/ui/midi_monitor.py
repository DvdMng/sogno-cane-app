"""Scrolling MIDI activity log."""
from __future__ import annotations

from collections import deque

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from sogno_cane.midi.mapper import MappingEvent
from sogno_cane.ui.theme import FG_CYAN, FG_LIME, FG_MAGENTA, FG_YELLOW


class MidiMonitor(QWidget):
    """Rolling list of the most recent MIDI events."""

    MAX_LINES = 200

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._list = QListWidget()
        self._list.setUniformItemSizes(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._list)

        self._pending: deque[MappingEvent] = deque(maxlen=2000)
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._drain)
        self._timer.start()

    def push_event(self, event: MappingEvent) -> None:
        self._pending.append(event)

    def _drain(self) -> None:
        if not self._pending:
            return
        for ev in list(self._pending):
            self._add_line(ev)
        self._pending.clear()
        while self._list.count() > self.MAX_LINES:
            self._list.takeItem(0)
        self._list.scrollToBottom()

    def _add_line(self, ev: MappingEvent) -> None:
        # Display channels 1-16 to match the DAW (Ableton); the wire stays 0-15.
        ch = ev.channel + 1
        if ev.kind == "note_on":
            text = (
                f"NOTE_ON   ch{ch:02d}  "
                f"n={ev.note:3d}  v={ev.velocity:3d}"
            )
            color = FG_LIME
        elif ev.kind == "note_off":
            text = f"NOTE_OFF  ch{ch:02d}  n={ev.note:3d}"
            color = FG_MAGENTA
        elif ev.kind == "cc":
            text = (
                f"CC        ch{ch:02d}  "
                f"cc={ev.control:3d}  v={ev.value:3d}"
            )
            color = FG_CYAN
        elif ev.kind == "pitchbend":
            text = f"PITCHBEND ch{ch:02d}  p={ev.pitch:+d}"
            color = FG_YELLOW
        else:
            text = f"{ev.kind.upper():9s} ch{ch:02d}"
            color = FG_LIME
        item = QListWidgetItem(text)
        item.setForeground(_qcolor(color))
        self._list.addItem(item)


def _qcolor(hex_str: str):
    from PySide6.QtGui import QColor
    return QColor(hex_str)
