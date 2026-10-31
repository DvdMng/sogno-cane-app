"""Projection: both EEGs on a second screen (the projector).

A window of its own, by default full screen on the projector set up as an
extended desktop (Windows: Win+P, Extend), while the app keeps working on
the PC's screen: it opens without taking the focus. Everything about it is
set from a control panel that stays open in the app and applies each
change at once, also while projecting: screen, full screen or window,
content (the combined perspective view, as lines or as one surface, or
the trace views), layout, seconds shown, amplitude, labels and the drift of
the point of view.
With a single screen it opens as an ordinary window.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, fields

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QScreen
from PySide6.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sogno_cane.eeg.profiles import DeviceProfile
from sogno_cane.eeg.unicorn_packet import UnicornPacket
from sogno_cane.ui.landscape_view import FRAME_MS as LANDSCAPE_FRAME_MS
from sogno_cane.ui.landscape_view import LandscapeView
from sogno_cane.ui.theme import PANEL_FILL, TEXT, TEXT_DIM
from sogno_cane.ui.waveform_widget import WaveformWidget

_FRAME_MS = 16
_FRAME_MS_SLOW = 33
_BLACK = "#000000"

CONTENTS = {"3d": "Vista 3D", "traces": "Tracce"}
SHAPES = {"lines": "Linee", "mesh": "Mesh (onde unite)"}
LAYOUTS = {
    "3d": {"grouped": "Uomo e cane affiancati",
           "interleaved": "Uomo e cane intrecciati"},
    "traces": {"stacked": "Uomo sopra, cane sotto",
               "side": "Uomo a sinistra, cane a destra",
               "human": "Solo uomo",
               "dog": "Solo cane"},
}
_TITLES = {"human": "UOMO", "dog": "CANE"}


@dataclass
class ProjectionSettings:
    screen: str = ""             # QScreen.name() of the projector
    fullscreen: bool = True
    content: str = "3d"
    layout: str = "grouped"
    shape: str = "lines"
    seconds: float = 6.0
    gain: float = 1.0
    labels: bool = True
    drift: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data) -> "ProjectionSettings":
        out = cls()
        if not isinstance(data, dict):
            return out
        for f in fields(cls):
            if f.name in data:
                try:
                    setattr(out, f.name, type(getattr(out, f.name))(
                        data[f.name]))
                except (TypeError, ValueError):
                    pass
        if out.content not in CONTENTS:
            out.content = "3d"
        if out.layout not in LAYOUTS[out.content]:
            out.layout = next(iter(LAYOUTS[out.content]))
        if out.shape not in SHAPES:
            out.shape = "lines"
        return out


def screen_label(screen: QScreen, primary: QScreen | None) -> str:
    size = screen.geometry()
    text = f"{screen.name()}  {size.width()}×{size.height()}"
    return text + ("  (schermo del PC)" if screen is primary else "")


def default_screen(screens: list[QScreen], primary: QScreen | None,
                   preferred: str = "") -> QScreen | None:
    """The remembered projector, else the first screen that is not the
    PC's own, else the PC's."""
    for s in screens:
        if preferred and s.name() == preferred:
            return s
    for s in screens:
        if s is not primary:
            return s
    return primary or (screens[0] if screens else None)


def find_screen(name: str) -> QScreen | None:
    return default_screen(list(QGuiApplication.screens()),
                          QGuiApplication.primaryScreen(), name)


class ProjectionControls(QDialog):
    """The projection's settings; every change applies at once."""

    settings_changed = Signal(object)     # ProjectionSettings
    toggle_requested = Signal()           # open / close the projection

    def __init__(self, settings: ProjectionSettings,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Proiezione")
        self.setModal(False)
        self.setObjectName("ProjectionControls")
        self.setStyleSheet(
            f"#ProjectionControls {{ background: {PANEL_FILL}; }}"
            f"QLabel, QCheckBox {{ color: {TEXT}; }}"
        )
        self._loading = True
        v = QVBoxLayout(self)
        form = QFormLayout()

        screen_row = QHBoxLayout()
        self._screen_cb = QComboBox()
        screen_row.addWidget(self._screen_cb, 1)
        rescan = QPushButton("RICERCA")
        rescan.setToolTip("Rileggi gli schermi collegati (dopo aver "
                          "collegato il proiettore).")
        rescan.clicked.connect(lambda: self._fill_screens(
            self.settings().screen))
        screen_row.addWidget(rescan)
        form.addRow("schermo", screen_row)

        self._full_cb = QCheckBox("a tutto schermo")
        self._full_cb.setToolTip(
            "Acceso: la proiezione copre tutto lo schermo scelto (senza "
            "bordi né cursore).\nSpento: è una finestra normale, da "
            "spostare e ridimensionare."
        )
        form.addRow("", self._full_cb)

        self._content_cb = QComboBox()
        for key, text in CONTENTS.items():
            self._content_cb.addItem(text, key)
        form.addRow("contenuto", self._content_cb)
        self._layout_cb = QComboBox()
        form.addRow("disposizione", self._layout_cb)
        self._shape_cb = QComboBox()
        for key, text in SHAPES.items():
            self._shape_cb.addItem(text, key)
        self._shape_cb.setToolTip(
            "Linee: ogni onda per sé. Mesh: le onde unite in un'unica\n"
            "superficie a trama fine; le onde restano le linee più\n"
            "luminose. Solo per la vista 3D."
        )
        form.addRow("forma", self._shape_cb)

        self._seconds_sb = QDoubleSpinBox()
        self._seconds_sb.setRange(2.0, 20.0)
        self._seconds_sb.setSingleStep(1.0)
        self._seconds_sb.setDecimals(0)
        self._seconds_sb.setSuffix(" s")
        self._seconds_sb.setToolTip(
            "Quanto passato si vede: la lunghezza delle tracce, o quanto\n"
            "lontano arrivano le onde nella vista 3D prima di dissolversi."
        )
        form.addRow("secondi", self._seconds_sb)
        self._gain_sb = QDoubleSpinBox()
        self._gain_sb.setRange(0.25, 4.0)
        self._gain_sb.setSingleStep(0.25)
        self._gain_sb.setDecimals(2)
        self._gain_sb.setSuffix(" ×")
        self._gain_sb.setToolTip("Altezza delle onde.")
        form.addRow("ampiezza", self._gain_sb)
        self._labels_cb = QCheckBox("UOMO / CANE e nomi degli elettrodi")
        form.addRow("scritte", self._labels_cb)
        self._drift_cb = QCheckBox("movimento lento del punto di vista")
        self._drift_cb.setToolTip("Il punto di vista oscilla lentamente, "
                                  "così la profondità si legge (vista 3D).")
        form.addRow("", self._drift_cb)
        v.addLayout(form)

        self._note = QLabel("")
        self._note.setWordWrap(True)
        self._note.setStyleSheet(f"color: {TEXT_DIM};")
        v.addWidget(self._note)
        self._toggle_btn = QPushButton("PROIETTA")
        self._toggle_btn.setObjectName("PrimaryButton")
        self._toggle_btn.clicked.connect(self.toggle_requested.emit)
        v.addWidget(self._toggle_btn)

        self._layout_wanted = settings.layout
        self._fill_screens(settings.screen)
        self._full_cb.setChecked(settings.fullscreen)
        self._content_cb.setCurrentIndex(list(CONTENTS).index(
            settings.content))
        self._fill_layouts()
        self._shape_cb.setCurrentIndex(list(SHAPES).index(settings.shape))
        self._seconds_sb.setValue(settings.seconds)
        self._gain_sb.setValue(settings.gain)
        self._labels_cb.setChecked(settings.labels)
        self._drift_cb.setChecked(settings.drift)
        self._sync_enabled()
        self._loading = False

        self._screen_cb.currentIndexChanged.connect(self._changed)
        self._content_cb.currentIndexChanged.connect(self._on_content)
        for w in (self._full_cb, self._labels_cb, self._drift_cb):
            w.toggled.connect(self._changed)
        self._layout_cb.currentIndexChanged.connect(self._changed)
        self._shape_cb.currentIndexChanged.connect(self._changed)
        self._seconds_sb.valueChanged.connect(self._changed)
        self._gain_sb.valueChanged.connect(self._changed)
        self._update_note()

    # ------------------------------------------------------------------ #

    def _fill_screens(self, preferred: str) -> None:
        loading, self._loading = self._loading, True
        self._screens = list(QGuiApplication.screens())
        primary = QGuiApplication.primaryScreen()
        self._screen_cb.clear()
        for s in self._screens:
            self._screen_cb.addItem(screen_label(s, primary), s.name())
        chosen = default_screen(self._screens, primary, preferred)
        if chosen in self._screens:
            self._screen_cb.setCurrentIndex(self._screens.index(chosen))
        self._loading = loading
        self._update_note()
        if not loading:
            self._changed()

    def _fill_layouts(self) -> None:
        content = self._content_cb.currentData()
        wanted = self._layout_cb.currentData() or self._layout_wanted
        self._layout_cb.blockSignals(True)
        self._layout_cb.clear()
        for key, text in LAYOUTS[content].items():
            self._layout_cb.addItem(text, key)
        keys = list(LAYOUTS[content])
        self._layout_cb.setCurrentIndex(
            keys.index(wanted) if wanted in keys else 0
        )
        self._layout_cb.blockSignals(False)

    def _on_content(self) -> None:
        self._fill_layouts()
        self._sync_enabled()
        self._changed()

    def _sync_enabled(self) -> None:
        is_3d = self._content_cb.currentData() == "3d"
        self._drift_cb.setEnabled(is_3d)
        self._shape_cb.setEnabled(is_3d)

    def _changed(self, *_args) -> None:
        if self._loading:
            return
        self._update_note()
        self.settings_changed.emit(self.settings())

    def _update_note(self) -> None:
        if not hasattr(self, "_screens"):
            return
        primary = QGuiApplication.primaryScreen()
        i = self._screen_cb.currentIndex()
        screen = self._screens[i] if 0 <= i < len(self._screens) else None
        if len(self._screens) < 2:
            text = ("C'è un solo schermo: la proiezione si apre in una "
                    "finestra. Per il proiettore collegalo in modalità "
                    "Estendi (Windows: Win+P, Estendi), poi premi RICERCA.")
        elif screen is primary and self._full_cb.isChecked():
            text = ("È lo schermo del PC: a tutto schermo lo coprirebbe. "
                    "Esc o doppio clic sulla proiezione per chiuderla.")
        else:
            text = ("La proiezione si apre sullo schermo scelto; l'app resta "
                    "qui e ogni modifica vale subito.")
        self._note.setText(text)

    # ------------------------------------------------------------------ #

    def settings(self) -> ProjectionSettings:
        return ProjectionSettings(
            screen=self._screen_cb.currentData() or "",
            fullscreen=self._full_cb.isChecked(),
            content=self._content_cb.currentData(),
            layout=self._layout_cb.currentData(),
            shape=self._shape_cb.currentData(),
            seconds=float(self._seconds_sb.value()),
            gain=float(self._gain_sb.value()),
            labels=self._labels_cb.isChecked(),
            drift=self._drift_cb.isChecked(),
        )

    def set_projecting(self, on: bool) -> None:
        self._toggle_btn.setText("CHIUDI PROIEZIONE" if on else "PROIETTA")


class ProjectionWindow(QWidget):
    """The projected picture, on a screen of its own."""

    closed = Signal()

    def __init__(
        self,
        profiles: dict[str, DeviceProfile],
        settings: ProjectionSettings,
        visible: dict[str, list[int]] | None = None,
    ) -> None:
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle("SOGNO_CANE — proiezione")
        # The app keeps the focus: the projector is not where one works.
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        # Black, painted by this window only (the app's stylesheet would win
        # over a palette); the content on top is transparent.
        self.setObjectName("ProjectionWindow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"#ProjectionWindow {{ background: {_BLACK}; }}")
        self._profiles = dict(profiles)
        self._settings = settings
        self._visible = {k: list(v) for k, v in (visible or {}).items()}
        self._view: LandscapeView | None = None
        self._traces: dict[str, WaveformWidget] = {}
        self._box = QVBoxLayout(self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._build()

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(
            LANDSCAPE_FRAME_MS if self._view is not None else _FRAME_MS
        )
        self._timer.timeout.connect(self._on_frame)
        self._frame_prev: float | None = None
        self._frame_gaps: list[float] = []
        self._timer.start()

    # ------------------------------------------------------------------ #
    # Content                                                            #
    # ------------------------------------------------------------------ #

    @property
    def settings(self) -> ProjectionSettings:
        return self._settings

    def _clear(self) -> None:
        while self._box.count():
            item = self._box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            elif item.layout() is not None:
                lay = item.layout()
                while lay.count():
                    sub = lay.takeAt(0).widget()
                    if sub is not None:
                        sub.deleteLater()
        self._view = None
        self._traces = {}

    def _build(self) -> None:
        self._clear()
        s = self._settings
        if s.content == "3d":
            view = LandscapeView(self._profiles, seconds=s.seconds,
                                 background=_BLACK)
            for kind, chans in self._visible.items():
                view.set_visible_channels(kind, chans)
            self._view = view
            self._apply_live(s)
            self._box.addWidget(view)
            self._set_frame_ms(LANDSCAPE_FRAME_MS)
            return
        kinds = {"human": ["human"], "dog": ["dog"]}.get(
            s.layout, ["human", "dog"])
        row = QBoxLayout(QBoxLayout.Direction.LeftToRight
                         if s.layout == "side"
                         else QBoxLayout.Direction.TopToBottom)
        row.setContentsMargins(24, 16, 24, 16)
        row.setSpacing(18)
        for kind in kinds:
            if kind not in self._profiles:
                continue
            col = QWidget()
            lay = QVBoxLayout(col)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(2)
            if s.labels:
                title = QLabel(_TITLES[kind])
                title.setStyleSheet(
                    f"color: {TEXT_DIM}; letter-spacing: 3px; "
                    "font-size: 9pt; background: transparent;"
                )
                lay.addWidget(title)
            wave = WaveformWidget(self._profiles[kind],
                                  seconds_visible=s.seconds)
            wave.set_axes_visible(False)
            if kind in self._visible:
                wave.set_visible_channels(self._visible[kind])
            lay.addWidget(wave, 1)
            self._traces[kind] = wave
            row.addWidget(col, 1)
        self._box.addLayout(row)
        self._apply_live(s)
        self._set_frame_ms(_FRAME_MS)

    def _set_frame_ms(self, ms: int) -> None:
        timer = getattr(self, "_timer", None)
        if timer is not None:
            timer.setInterval(ms)
            self._frame_gaps = []

    def _apply_live(self, s: ProjectionSettings) -> None:
        """What changes without rebuilding the picture."""
        if self._view is not None:
            self._view.order = s.layout
            self._view.shape = s.shape
            self._view.show_labels = s.labels
            self._view.gain = s.gain
            self._view.drift = s.drift
        for wave in self._traces.values():
            wave.gain = s.gain
            wave.set_labels_visible(s.labels)

    def apply(self, settings: ProjectionSettings) -> None:
        """Take new settings: rebuild only if the picture itself changed."""
        old, self._settings = self._settings, settings
        rebuild = (old.content, old.seconds) != \
            (settings.content, settings.seconds) or (
            settings.content == "traces"
            and (old.layout, old.labels) != (settings.layout,
                                             settings.labels))
        if rebuild:
            self._build()
            self.update()            # the margins too: no trace of before
        else:
            self._apply_live(settings)
        if (old.screen, old.fullscreen) != (settings.screen,
                                            settings.fullscreen):
            self.place(find_screen(settings.screen))

    # ------------------------------------------------------------------ #
    # Feeding                                                            #
    # ------------------------------------------------------------------ #

    def push(self, kind: str, packet: UnicornPacket) -> None:
        if self._view is not None:
            self._view.push(kind, packet)
        wave = self._traces.get(kind)
        if wave is not None:
            wave.push_packet(packet)

    def set_visible_channels(self, kind: str, indices) -> None:
        self._visible[kind] = list(indices)
        if self._view is not None:
            self._view.set_visible_channels(kind, indices)
        wave = self._traces.get(kind)
        if wave is not None:
            wave.set_visible_channels(indices)

    def set_profile(self, kind: str, profile: DeviceProfile) -> None:
        self._profiles[kind] = profile
        self._build()

    # ------------------------------------------------------------------ #
    # Placement                                                          #
    # ------------------------------------------------------------------ #

    def place(self, screen: QScreen | None) -> None:
        """On ``screen``: full screen if asked and if it is not the only
        screen there is, else an ordinary window."""
        full = self._settings.fullscreen and screen is not None \
            and len(QGuiApplication.screens()) >= 2
        if self.isVisible():
            self.hide()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, full)
        if screen is not None:
            self.setScreen(screen)
            geo = screen.availableGeometry()
            if full:
                self.setGeometry(screen.geometry())
            else:
                self.setGeometry(
                    geo.x() + geo.width() // 6, geo.y() + geo.height() // 6,
                    geo.width() * 2 // 3, geo.height() * 2 // 3,
                )
        if full:
            self.showFullScreen()
            self.setCursor(Qt.CursorShape.BlankCursor)
        else:
            self.showNormal()
            self.unsetCursor()

    # ------------------------------------------------------------------ #
    # Frames and closing                                                 #
    # ------------------------------------------------------------------ #

    def _on_frame(self) -> None:
        """One frame; settle for 30 fps if 60 cannot be held."""
        now = time.perf_counter()
        if self._frame_prev is not None and \
                self._timer.interval() == _FRAME_MS:
            gap = now - self._frame_prev
            if gap < 0.25:
                self._frame_gaps.append(gap)
            if len(self._frame_gaps) >= 120:
                if sorted(self._frame_gaps)[60] > _FRAME_MS * 1.4 / 1000.0:
                    self._timer.setInterval(_FRAME_MS_SLOW)
                self._frame_gaps.clear()
        self._frame_prev = now
        if self._view is not None:
            self._view.advance()
        for wave in self._traces.values():
            wave.repaint_traces()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt name
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, _event) -> None:  # noqa: N802
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self._timer.stop()
        self.closed.emit()
        super().closeEvent(event)
