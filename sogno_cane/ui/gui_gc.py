"""Run Python's cyclic garbage collector in the GUI thread only.

Left automatic, the collector runs in whichever thread happens to trip its
allocation threshold. With live EEG that is usually the UDP receiver or the
engine thread, each busy ~60 times a second. If a collection there frees a
Qt object (a dialog, a replaced plot), Qt's destructor runs in that thread:
it takes a Qt lock and then waits for the GIL, while the GUI thread, holding
the GIL, waits for that same lock — the window freezes for good. That exact
deadlock showed up in the test suite while 0.9.3 was being written;
pyqtgraph documents the same hazard.

So automatic collection is switched off and a GUI-thread timer collects
instead, generation by generation, with the interpreter's own thresholds.
Reference counting still frees everything that is not in a cycle at once.
"""
from __future__ import annotations

import gc

from PySide6.QtCore import QObject, QTimer


class GuiThreadGarbageCollector(QObject):
    """Create once, in the GUI thread, and keep a reference."""

    def __init__(self, parent: QObject | None = None,
                 interval_ms: int = 1000) -> None:
        super().__init__(parent)
        self._threshold = gc.get_threshold()
        gc.disable()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.check)
        self._timer.start(interval_ms)

    def check(self) -> None:
        """Collect the generations that are due, as the automatic GC would."""
        l0, l1, l2 = gc.get_count()
        if l0 > self._threshold[0]:
            gc.collect(0)
            if l1 > self._threshold[1]:
                gc.collect(1)
                if l2 > self._threshold[2]:
                    gc.collect(2)
