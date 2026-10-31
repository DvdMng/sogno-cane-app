"""Main application window for SOGNO_CANE."""
from __future__ import annotations

import html
import threading
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
import os

from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sogno_cane import __version__, config as appconfig, update as updater
from sogno_cane.eeg.bands import compute_band_powers
from sogno_cane.eeg.profiles import DOG_PROFILE, HUMAN_PROFILE
from sogno_cane.eeg.unicorn_packet import UnicornPacket
from sogno_cane.io.archive import Archive
from sogno_cane.settings import Settings
from sogno_cane.ui.archive_panel import ArchivePanel
from sogno_cane.ui.device_panel import DevicePanel
from sogno_cane.ui.landscape_gl import make_landscape_view
from sogno_cane.ui.landscape_view import (
    NAVIGATION_HINT,
    Navigation,
    ViewStyle,
)
from sogno_cane.ui.mapping_panel import MappingPanel
from sogno_cane.ui.midi_monitor import MidiMonitor
from sogno_cane.ui.projection import (
    ProjectionControls,
    ProjectionSettings,
    ProjectionWindow,
    find_screen,
)
from sogno_cane.ui.theme import (
    ACCENT_CYAN,
    ACCENT_PINK,
    TEXT_DIM,
    BackgroundWidget,
    ChromeTitle,
    app_icon,
)
from sogno_cane.ui.style_controls import ViewStyleControls
from sogno_cane.ui.wrap_row import WrapRow

# Room for the window frame and title bar (Windows), in logical pixels.
_FRAME_W = 16
_FRAME_H = 40


class MainWindow(QMainWindow):
    # Cross-thread signals for the online updater.
    _update_available = Signal(object)   # updater.UpdateInfo
    _update_staged = Signal(str)         # version
    _update_failed = Signal(str)         # message
    _update_none = Signal()              # manual check found nothing

    def __init__(self, archive: Archive | None = None) -> None:
        super().__init__()
        # The version belongs where the user can always see it: not knowing
        # which build is running makes every update look like a no-op.
        self.setWindowTitle(f"SOGNO_CANE {__version__}")
        try:
            self.setWindowIcon(app_icon())
        except Exception:
            pass
        self._settings = Settings.load()
        w = int(self._settings.get("window", "w", default=1600))
        h = int(self._settings.get("window", "h", default=1020))
        self.resize(w, h)

        self._archive = archive or Archive()

        bg = BackgroundWidget()
        self.setCentralWidget(bg)
        self._bg = bg

        outer = QVBoxLayout(bg)
        outer.setContentsMargins(28, 10, 28, 12)
        outer.setSpacing(8)

        outer.addWidget(ChromeTitle("SOGNO_CANE"))

        # Top-level tabs: STUDIO (live) / MAPPING (config) / ARCHIVE.
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_studio(), "STUDIO")
        self._tabs.addTab(self._build_mapping(), "MAPPATURA")
        self._archive_panel = ArchivePanel(self._archive)
        self._tabs.addTab(self._archive_panel, "ARCHIVIO")
        outer.addWidget(self._tabs, 1)

        # Wire archive <-> devices.
        self._human.recording_saved.connect(
            lambda _id: self._archive_panel.refresh()
        )
        self._dog.recording_saved.connect(
            lambda _id: self._archive_panel.refresh()
        )
        self._archive_panel.load_to_human.connect(self._human.play_loaded)
        self._archive_panel.load_to_dog.connect(self._dog.play_loaded)
        self._archive_panel.load_session.connect(self._play_session)
        self._human.recording_changed.connect(self._sync_record_button)
        self._dog.recording_changed.connect(self._sync_record_button)
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Projection on a second screen (window and its control panel).
        self._projection: ProjectionWindow | None = None
        self._projection_controls: ProjectionControls | None = None
        self._style_controls: ViewStyleControls | None = None
        # Esc here too: a full-screen projection on this screen covers the
        # app, and the keys may still come to the app.
        esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc.activated.connect(self._on_escape)
        self._projection_settings = ProjectionSettings.from_dict(
            self._settings.get("projection", "settings", default={})
        )
        app = QGuiApplication.instance()
        if app is not None:
            app.screenRemoved.connect(self._on_screen_removed)

        # Spectrum refresh timer.
        self._human_window_buf: list[np.ndarray] = []
        self._dog_window_buf: list[np.ndarray] = []
        self._spectrum_timer = QTimer(self)
        self._spectrum_timer.setInterval(250)
        self._spectrum_timer.timeout.connect(self._refresh_spectra)
        self._spectrum_timer.start()

        self._restore_settings()

        # The configuration in use: which file (None = never saved) and
        # what it held when last saved or loaded, to tell "modified".
        self._config_path: str | None = None
        self._config_snapshot = self._devices_state()
        self._shown_config = None
        self._restore_session()
        self._refresh_config_view()
        self._config_timer = QTimer(self)
        self._config_timer.setInterval(1000)
        self._config_timer.timeout.connect(self._refresh_config_view)
        self._config_timer.start()

        self.statusBar().showMessage(
            "pronto — le registrazioni stanno in " + self._archive.root
        )

        # Online auto-update wiring.
        self._manual_update_check = False
        self._update_available.connect(self._on_update_available)
        self._update_staged.connect(self._on_update_staged)
        self._update_failed.connect(self._on_update_failed)
        self._update_none.connect(self._on_update_none)
        self._announce_applied_update()
        self._notify_pending_update()
        if updater.update_url() and updater.auto_update_enabled():
            self._start_update_check(manual=False)
        self._fit_to_screen()

    # ------------------------------------------------------------------ #
    # Layout                                                             #
    # ------------------------------------------------------------------ #

    def _fit_to_screen(self) -> None:
        """A saved size larger than the screen (the 1600 px default on a
        scaled screen, a size saved on a bigger monitor) is reduced to fit,
        frame included, and the window centred: where the screen can hold
        the window at all."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        self.ensurePolished()               # sizes with the theme applied
        least = self.minimumSizeHint()
        w, h = self.width(), self.height()
        if w + _FRAME_W > avail.width() >= least.width() + _FRAME_W:
            w = avail.width() - _FRAME_W
        if h + _FRAME_H > avail.height() >= least.height() + _FRAME_H:
            h = avail.height() - _FRAME_H
        if (w, h) == (self.width(), self.height()):
            return
        self.resize(w, h)
        self.move(avail.x() + (avail.width() - w - _FRAME_W) // 2,
                  avail.y() + (avail.height() - h - _FRAME_H) // 2)

    def _build_studio(self) -> QWidget:
        studio = QWidget()
        v = QVBoxLayout(studio)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        # Global transport bar.
        v.addWidget(self._build_transport())

        # Device panels — these get the lion's share of the height.
        self._human = DevicePanel("UMANO", HUMAN_PROFILE, archive=self._archive)
        self._dog = DevicePanel("CANE", DOG_PROFILE, archive=self._archive)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.setHandleWidth(8)
        top.addWidget(self._human)
        top.addWidget(self._dog)
        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 1)

        # MIDI activity monitor. It used to eat a fixed 120 px of the panel
        # area whether or not anything was happening; now it shares a
        # draggable splitter with the devices and starts small, so the
        # waveforms (the actual centrepiece) get the room.
        mon_wrap = QWidget()
        mon_v = QVBoxLayout(mon_wrap)
        mon_v.setContentsMargins(0, 0, 0, 0)
        mon_v.setSpacing(2)
        mon_label = QLabel("ATTIVITÀ MIDI")
        mon_label.setStyleSheet(
            "color: #7F7298; font-size: 8pt; letter-spacing: 2px;"
        )
        mon_v.addWidget(mon_label)
        self._monitor = MidiMonitor()
        self._monitor.setMinimumHeight(48)
        mon_v.addWidget(self._monitor, 1)

        # The combined perspective view of both EEGs: hidden until VISTA 3D,
        # then it takes the place of the two trace views. On the graphics
        # card where OpenGL works.
        self._landscape = make_landscape_view(
            {"human": self._human.profile, "dog": self._dog.profile},
        )
        self._landscape_box = QWidget()
        box = QVBoxLayout(self._landscape_box)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        opts = QHBoxLayout()
        opts.setSpacing(6)
        self._shape_btns = self._segmented(
            opts, "FORMA", (("lines", "LINEE"), ("mesh", "MESH")),
            "Linee: ogni onda per sé. Mesh: le onde unite in un'unica\n"
            "superficie a trama fine, che nasconde ciò che sta dietro;\n"
            "le onde restano le linee più luminose.",
            lambda key: self._set_landscape("shape", key),
        )
        opts.addSpacing(18)
        self._order_btns = self._segmented(
            opts, "DISPOSIZIONE",
            (("grouped", "AFFIANCATI"), ("interleaved", "INTRECCIATI")),
            "Affiancati: le onde dell'uomo a sinistra, quelle del cane a\n"
            "destra. Intrecciati: un canale dell'uomo e uno del cane, a turno.",
            lambda key: self._set_landscape("order", key),
        )
        # The view is navigable with the mouse; say how, quietly, on the
        # right. The note wraps on a narrow window instead of widening it
        # (a long line held the whole window wider than the screen).
        hint = QLabel(NAVIGATION_HINT)
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignmentFlag.AlignRight
                          | Qt.AlignmentFlag.AlignVCenter)
        hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 8pt;")
        opts.addSpacing(18)
        style_btn = QPushButton("REGOLAZIONI")
        style_btn.setToolTip(
            "Contrasto, spessore delle linee, campo visibile, ampiezza e\n"
            "tessuto della vista 3D (anche in proiezione)."
        )
        style_btn.clicked.connect(self._show_style_controls)
        opts.addWidget(style_btn)
        opts.addSpacing(18)
        opts.addWidget(hint, 1)
        box.addLayout(opts)
        box.addWidget(self._landscape, 1)
        self._landscape_box.hide()

        self._studio_split = QSplitter(Qt.Orientation.Vertical)
        self._studio_split.setHandleWidth(8)
        self._studio_split.addWidget(top)
        self._studio_split.addWidget(self._landscape_box)
        self._studio_split.addWidget(mon_wrap)
        self._studio_split.setStretchFactor(0, 5)
        self._studio_split.setStretchFactor(1, 6)
        self._studio_split.setStretchFactor(2, 1)
        self._studio_split.setSizes([760, 0, 110])
        v.addWidget(self._studio_split, 1)

        self._landscape_timer = QTimer(self)
        self._landscape_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._landscape_timer.setInterval(self._landscape.FRAME_MS)
        self._landscape_timer.timeout.connect(self._landscape.advance)

        # Wire device events.
        self._human.packet_received.connect(self._on_human_packet)
        self._dog.packet_received.connect(self._on_dog_packet)
        self._human.event_emitted.connect(self._monitor.push_event)
        self._dog.event_emitted.connect(self._monitor.push_event)
        for kind, panel in (("human", self._human), ("dog", self._dog)):
            panel.visible_channels_changed.connect(
                lambda chans, k=kind: self._on_visible_channels(k, chans)
            )
            panel.profile_changed.connect(
                lambda prof, k=kind: self._on_panel_profile(k, prof)
            )
        return studio

    def _build_mapping(self) -> QWidget:
        """Mapping configuration for both devices, in its own tab."""
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)

        # The configuration in use, and what can be done with it.
        bar = QHBoxLayout()
        bar.setSpacing(8)
        cap = QLabel("CONFIGURAZIONE ATTUALE")
        cap.setStyleSheet("font-weight: 700; letter-spacing: 2px;")
        bar.addWidget(cap)
        self._cfg_name_lbl = QLabel("")
        self._cfg_name_lbl.setStyleSheet(
            f"color: {ACCENT_CYAN}; font-weight: 700; font-size: 11pt;"
        )
        bar.addWidget(self._cfg_name_lbl)
        self._cfg_state_lbl = QLabel("")
        bar.addWidget(self._cfg_state_lbl)
        bar.addStretch(1)
        save_btn = QPushButton("SALVA")
        save_btn.setToolTip(
            "Sovrascrive la configurazione attuale con le impostazioni di "
            "adesso\n(entrambi i dispositivi). Se non ha ancora un nome, te "
            "lo chiede."
        )
        save_btn.clicked.connect(self._save_config)
        bar.addWidget(save_btn)
        save_as_btn = QPushButton("SALVA COME…")
        save_as_btn.setToolTip(
            "Salva le impostazioni di adesso come una nuova configurazione."
        )
        save_as_btn.clicked.connect(self._save_config_as)
        bar.addWidget(save_as_btn)
        load_btn = QPushButton("CARICA…")
        load_btn.setToolTip("Carica una configurazione salvata.")
        load_btn.clicked.connect(self._load_config)
        bar.addWidget(load_btn)
        v.addLayout(bar)

        hint = QLabel(
            "Come l'EEG di ogni dispositivo pilota il MIDI. I canali MIDI sono "
            "1–16 (come in Ableton). Ogni modifica vale subito; la "
            "configurazione salva tutto, per entrambi i dispositivi, e alla "
            "riapertura ritrovi quella in uso."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #7F7298; padding: 0 2px 4px;")
        v.addWidget(hint)

        self._mapping_tabs = QTabWidget()
        self._rebuild_mapping_tabs()
        v.addWidget(self._mapping_tabs, 1)
        return wrap

    def _rebuild_mapping_tabs(self) -> None:
        """(Re)build the per-device mapping editors bound to the live bundles."""
        tabs = self._mapping_tabs
        tabs.clear()
        tabs.addTab(MappingPanel(self._human.bundle, "MAPPATURA UMANO"), "UMANO")
        tabs.addTab(MappingPanel(self._dog.bundle, "MAPPATURA CANE"), "CANE")

    def _build_transport(self) -> QWidget:
        """The transport row. On a window too narrow for all of it (a
        maximized window on a small or scaled screen) what does not fit goes
        to a second line: no button is ever narrower than its text. Gaps
        and padding are a little tighter than elsewhere, so that the row
        stays on one line on a 1536 px wide screen (1920 at 125 %)."""
        self._transport = QWidget()
        self._transport.setObjectName("TransportBar")
        bar = WrapRow(self._transport, spacing=6, line_spacing=6)
        lbl = QLabel("TRASPORTO")
        lbl.setStyleSheet("font-weight: 700; letter-spacing: 3px;")
        bar.addWidget(lbl)
        start_all = QPushButton("AVVIA TUTTO")
        start_all.clicked.connect(self._start_all)
        bar.addWidget(start_all)
        stop_all = QPushButton("FERMA TUTTO")
        stop_all.clicked.connect(self._stop_all)
        bar.addWidget(stop_all)
        self._rec_both_btn = QPushButton("● REGISTRA UOMO+CANE")
        self._rec_both_btn.setToolTip(
            "Avvia la registrazione dei due pannelli nello stesso istante\n"
            "(e fa partire i pannelli fermi). In ARCHIVIO le due registrazioni\n"
            "hanno lo stesso nome di sessione e si ricaricano insieme con\n"
            "CARICA SESSIONE. Premi di nuovo per fermarle entrambe."
        )
        self._rec_both_btn.clicked.connect(self._toggle_record_both)
        bar.addWidget(self._rec_both_btn)
        panic = QPushButton("PANIC")
        panic.setToolTip(
            "Spegne ogni nota su tutte le porte MIDI collegate: un note-off\n"
            "per ogni nota di ogni canale, poi Sustain off e All Notes Off.\n"
            "Spegne anche le note rimaste accese da una sessione precedente."
        )
        panic.clicked.connect(self._panic)
        bar.addWidget(panic)
        swap = QPushButton("SCAMBIA EEG")
        swap.setToolTip(
            "Se il segnale dell'uomo è finito sul pannello CANE (e viceversa)\n"
            "scambia i due EEG dal vivo tra i pannelli, senza ricollegare."
        )
        swap.clicked.connect(self._swap_live_eeg)
        bar.addWidget(swap)
        self._view3d_btn = QPushButton("VISTA 3D")
        self._view3d_btn.setCheckable(True)
        self._view3d_btn.setToolTip(
            "Mostra le onde dell'uomo e del cane insieme, in prospettiva,\n"
            "al posto delle due tracce. Premi di nuovo per tornare alle\n"
            "tracce."
        )
        self._view3d_btn.toggled.connect(self._set_view_3d)
        bar.addWidget(self._view3d_btn)
        self._projection_btn = QPushButton("PROIEZIONE")
        self._projection_btn.setToolTip(
            "Proietta le onde su un secondo schermo (il proiettore, collegato\n"
            "in modalità Estendi) mentre l'app resta qui. Apre il pannello\n"
            "della proiezione: schermo, tutto schermo, contenuto e resa."
        )
        self._projection_btn.clicked.connect(self._show_projection_controls)
        bar.addWidget(self._projection_btn)
        # Configuration, version, updates and help stay together, at the
        # right end of the row (or of the second line).
        tail = QWidget()
        bar.addWidget(tail)
        tail_row = QHBoxLayout(tail)
        tail_row.setContentsMargins(0, 0, 0, 0)
        tail_row.setSpacing(6)
        self._cfg_studio_lbl = QLabel("")
        self._cfg_studio_lbl.setToolTip(
            "Configurazione attuale (la mappatura di entrambi i dispositivi).\n"
            "Il pallino rosa vuol dire: ci sono modifiche non salvate.\n"
            "SALVA la sovrascrive; per salvarla con un altro nome o caricarne\n"
            "un'altra vai nella scheda MAPPATURA."
        )
        tail_row.addWidget(self._cfg_studio_lbl)
        cfg_save = QPushButton("SALVA")
        cfg_save.setToolTip("Sovrascrive la configurazione attuale.")
        cfg_save.clicked.connect(self._save_config)
        tail_row.addWidget(cfg_save)
        tail_row.addSpacing(8)
        self._version_lbl = QLabel(f"v{__version__}")
        self._version_lbl.setObjectName("VersionTag")
        self._version_lbl.setToolTip(
            "Versione installata. Premi AGGIORNAMENTI per cercarne una nuova."
        )
        tail_row.addWidget(self._version_lbl)
        self._update_btn = QPushButton("AGGIORNAMENTI")
        self._update_btn.setToolTip("Cerca online una versione più recente")
        self._update_btn.clicked.connect(lambda: self._start_update_check(True))
        tail_row.addWidget(self._update_btn)
        help_btn = QPushButton("?")
        help_btn.setFixedWidth(36)
        help_btn.setToolTip("Aiuto rapido")
        help_btn.clicked.connect(self._show_help)
        tail_row.addWidget(help_btn)
        return self._transport

    # ------------------------------------------------------------------ #
    # Transport                                                          #
    # ------------------------------------------------------------------ #

    def _start_all(self) -> None:
        for d in (self._human, self._dog):
            if not d.is_streaming:
                d._toggle_stream()

    def _stop_all(self) -> None:
        for d in (self._human, self._dog):
            if d.is_streaming:
                d._toggle_stream()

    def _toggle_record_both(self) -> None:
        """Record UOMO and CANE together, or stop whatever is recording.

        Both panels are running before either recorder starts, then the two
        recorders start in the same instant: the pair lines up to within one
        packet (~16 ms with the headsets, 100 ms with simulator or file).
        """
        panels = (self._human, self._dog)
        if any(p.is_recording for p in panels):
            for p in panels:
                p.stop_recording(save=True)
            self.statusBar().showMessage(
                "Registrazione UOMO+CANE salvata in ARCHIVIO", 6000,
            )
            return
        for p in panels:
            if not p.is_streaming:
                p._toggle_stream()
        session = time.strftime("sess-%Y%m%d-%H%M%S")
        started = self._archive.now_iso()
        for p in panels:
            p.start_recording(session=session, started=started)
        self.statusBar().showMessage(
            f"Registrazione UOMO+CANE in corso — sessione {started}", 6000,
        )

    def _sync_record_button(self, *_args) -> None:
        # While anything records (a pair or a single panel) the button
        # stops every recording, and says so.
        busy = self._human.is_recording or self._dog.is_recording
        self._rec_both_btn.setText(
            "■ FERMA REGISTRAZIONI" if busy else "● REGISTRA UOMO+CANE"
        )

    def _play_session(self, human_loaded, dog_loaded) -> None:
        """Play a recorded UOMO+CANE pair on its two panels, in step."""
        pairs = [
            (p, ld) for p, ld in ((self._human, human_loaded),
                                  (self._dog, dog_loaded))
            if ld is not None
        ]
        for p, _ld in pairs:
            if p.is_streaming:
                p._toggle_stream()
        for p, ld in pairs:
            p.play_loaded(ld, autostart=False)
        for p, _ld in pairs:             # start together
            p._toggle_stream()
        self._tabs.setCurrentIndex(0)
        self.statusBar().showMessage(
            "Sessione UOMO+CANE in riproduzione sui due pannelli", 6000,
        )

    def _swap_live_eeg(self) -> None:
        """Exchange the two panels' live headsets.

        With two Recorders the app can tell the streams apart but not which
        one is the dog: that only the user knows (tap an electrode, watch
        which panel jumps). Swapping keeps both streams' claims intact.
        """
        h, d = self._human, self._dog
        if h.live_source is None or d.live_source is None:
            QMessageBox.information(
                self, "Scambia EEG",
                "Per scambiare servono due EEG dal vivo, uno per pannello.",
            )
            return
        h_live = h.detach_live_source()
        d_live = d.detach_live_source()
        h.adopt_live_source(d_live)
        d.adopt_live_source(h_live)
        self.statusBar().showMessage(
            "EEG scambiati tra UMANO e CANE", 4000,
        )

    def _panic(self) -> None:
        self._human.panic()
        self._dog.panic()
        self.statusBar().showMessage("PANIC — tutte le note spente", 2000)

    def _show_help(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, "SOGNO_CANE — guida rapida",
            "L'ORDINE GIUSTO\n"
            "  1. Scegli il PROFILO (umano / cane) su ogni pannello.\n"
            "  2. Scegli la porta MIDI e premi COLLEGA (su Windows serve\n"
            "     loopMIDI). Finché il pulsante è giallo il suono non esce.\n"
            "  3. Scegli la SORGENTE: SIMULATORE, FILE o EEG DAL VIVO.\n"
            "  4. Premi AVVIA.\n\n"
            "SORGENTI\n"
            "  • SIMULATORE — segnale sintetico, per provare senza cuffie.\n"
            "  • FILE — riproduce una registrazione (CSV / EDF / BDF / npz)\n"
            "    attraverso la stessa mappatura musicale.\n"
            "  • EEG DAL VIVO — collega una cuffia Unicorn BCI Core vera.\n"
            "    Nel BCI Core Recorder: Connect, poi Settings → Network →\n"
            "    'Sending data via UDP' (127.0.0.1, porta 1001), poi Play.\n"
            "    Poi premi EEG DAL VIVO: porta e formato si riconoscono da\n"
            "    soli. Con due cuffie: due Recorder, anche sulla stessa\n"
            "    porta — SOGNO_CANE li distingue e a ogni pannello dà il\n"
            "    suo. Se uomo e cane risultano invertiti: SCAMBIA EEG.\n"
            "    Se nel Recorder è attivo solo LSL, attiva anche UDP.\n"
            "    Filtri del Recorder: meglio spenti — SOGNO_CANE toglie da\n"
            "    solo l'offset degli elettrodi e il ronzio a 50 Hz. Guida\n"
            "    completa: EEG_COLLEGAMENTO.txt nella cartella dell'app.\n\n"
            "LA RIGA DI STATO IN BASSO AL PANNELLO\n"
            "  Dice sempre la verità su cosa entra e cosa esce: verde tutto\n"
            "  a posto, rosa qualcosa non va (nessun dato EEG, MIDI non\n"
            "  collegato, invii falliti).\n\n"
            "VEDERE LE ONDE\n"
            "  • ONDE accanto al nome del pannello sceglie quali onde\n"
            "    mostrare (per esempio le 4 collegate all'amplificatore del\n"
            "    cane). Vale per la visualizzazione, non per il MIDI.\n"
            "  • VISTA 3D, in alto, mostra le onde dell'uomo e del cane\n"
            "    insieme, in prospettiva: il presente in primo piano, il\n"
            "    passato che si allontana e svanisce. FORMA: LINEE, ogni\n"
            "    onda per sé, o MESH, le onde unite in un'unica superficie.\n"
            "    DISPOSIZIONE: affiancati (uomo a sinistra, cane a destra)\n"
            "    o intrecciati. Si naviga con il mouse: trascina per\n"
            "    ruotare, tasto destro per spostare, rotella per avvicinare,\n"
            "    doppio clic per l'inquadratura iniziale. REGOLAZIONI:\n"
            "    contrasto, spessore delle linee, campo visibile, ampiezza e,\n"
            "    con MESH, tessuto. La proiezione mostra lo stesso punto di\n"
            "    vista e le stesse regolazioni.\n"
            "  • PROIEZIONE porta le onde sul proiettore collegato in\n"
            "    modalità Estendi (Windows: Win+P, Estendi) mentre l'app\n"
            "    resta qui. Dal suo pannello: schermo, tutto schermo,\n"
            "    vista 3D o tracce, disposizione; per la vista 3D forma e\n"
            "    regolazioni (le stesse di STUDIO), per le tracce campo\n"
            "    visibile e ampiezza. A tutto schermo non c'è la barra del\n"
            "    titolo: Esc riporta la finestra, un secondo Esc la chiude,\n"
            "    F11 torna a tutto schermo.\n\n"
            "NOTE CHE RESTANO ACCESE\n"
            "  PANIC manda un note-off per ogni nota di ogni canale: spegne\n"
            "  anche le note rimaste appese in Ableton da una sessione\n"
            "  precedente, senza riavviarlo.\n\n"
            "REGISTRARE E RIASCOLTARE\n"
            "  • ● REGISTRA su un pannello registra quel pannello.\n"
            "  • ● REGISTRA UOMO+CANE, in alto, avvia le due registrazioni\n"
            "    nello stesso istante. In ARCHIVIO la coppia ha lo stesso\n"
            "    nome di sessione e CARICA SESSIONE la riproduce sui due\n"
            "    pannelli insieme.\n"
            "  • Dalla scheda ARCHIVIO puoi rinominare, tagliare un\n"
            "    intervallo, esportare in CSV, eliminare o ricaricare una\n"
            "    registrazione in un dispositivo.\n\n"
            "CONFIGURAZIONE\n"
            "  • In alto e nella scheda MAPPATURA è indicata la\n"
            "    configurazione in uso; il pallino rosa segnala modifiche\n"
            "    non salvate. SALVA la sovrascrive, SALVA COME… ne crea\n"
            "    una nuova, CARICA… ne apre un'altra. Alla riapertura\n"
            "    l'app riparte dalla configurazione in uso, comprese le\n"
            "    modifiche non ancora salvate.\n\n"
            "MUSICALITÀ (scheda MAPPATURA)\n"
            "  • ogni scheda si apre con la descrizione della voce: cosa\n"
            "    ascolta, cosa invia e come si usa in Ableton;\n"
            "  • intervallo / durata minima / soglia di cambiamento\n"
            "    regolano quanto è rada ogni voce;\n"
            "  • 'note length' quanto dura ogni nota (15 ms = trigger per\n"
            "    gli inviluppi, 0 = HOLD legato);\n"
            "  • nei CC, «morbidezza» è il tempo con cui il CC segue il\n"
            "    segnale (da «nessuna» a 30 s) e «uscita» il tratto del\n"
            "    parametro percorso, per esempio 30–90;\n"
            "  • 'max silence' fa ripartire una voce rimasta muta troppo a\n"
            "    lungo, così nessun canale MIDI muore.",
        )

    def _on_tab_changed(self, idx: int) -> None:
        if self._tabs.tabText(idx) == "ARCHIVIO":
            self._archive_panel.refresh()

    # ------------------------------------------------------------------ #
    # Online updates                                                     #
    # ------------------------------------------------------------------ #

    def _notify_pending_update(self) -> None:
        pend = updater.has_pending_update()
        if pend:
            self.statusBar().showMessage(
                f"Aggiornamento {pend.get('version','?')} pronto — "
                f"riavvia per installarlo",
                8000,
            )

    def _announce_applied_update(self) -> None:
        """Tell the user, once, that the app just changed under their feet."""
        notice = updater.take_applied_notice()
        if not notice:
            return
        version = str(notice.get("version", "")).strip()
        notes = str(notice.get("notes", "")).strip()
        body = f"SOGNO_CANE è stato aggiornato alla versione {version}."
        if notes:
            body += f"\n\nCosa cambia:\n{notes}"
        QMessageBox.information(self, "Aggiornamento installato", body)
        self.statusBar().showMessage(f"Aggiornato alla versione {version}", 8000)

    def _start_update_check(self, manual: bool = False) -> None:
        if not updater.update_url():
            if manual:
                QMessageBox.information(
                    self, "Updates",
                    "No update URL is configured.\n\n"
                    "Set 'update_url' in settings.json (or the "
                    "SOGNO_CANE_UPDATE_URL environment variable) to your "
                    "release manifest, e.g. a GitHub Releases version.json.",
                )
            return
        self._manual_update_check = manual
        if manual:
            self.statusBar().showMessage("Cerco aggiornamenti…", 4000)
        threading.Thread(target=self._check_thread, daemon=True).start()

    def _check_thread(self) -> None:
        info = updater.check_for_update(current=__version__)
        if info is not None:
            self._update_available.emit(info)
        elif self._manual_update_check:
            self._update_none.emit()

    def _on_update_none(self) -> None:
        QMessageBox.information(
            self, "Aggiornamenti",
            f"Sei aggiornato (versione {__version__}).",
        )

    def _on_update_available(self, info) -> None:
        auto = updater.auto_update_enabled()
        if auto and not self._manual_update_check:
            self.statusBar().showMessage(
                f"Scarico l'aggiornamento {info.version}…", 0
            )
            self._download_update(info)
            return
        notes = f"\n\n{info.notes}" if info.notes else ""
        if QMessageBox.question(
            self, "Aggiornamento disponibile",
            f"È disponibile la versione {info.version} "
            f"(tu hai la {__version__}).{notes}\n\n"
            "La scarico adesso? Verrà installata al prossimo avvio.",
        ) == QMessageBox.StandardButton.Yes:
            self.statusBar().showMessage(
                f"Scarico l'aggiornamento {info.version}…", 0
            )
            self._download_update(info)

    def _download_update(self, info) -> None:
        def work():
            try:
                updater.download_and_stage(info)
                self._update_staged.emit(info.version)
            except Exception as exc:
                self._update_failed.emit(str(exc))
        threading.Thread(target=work, daemon=True).start()

    def _on_update_staged(self, version: str) -> None:
        self.statusBar().showMessage(
            f"Aggiornamento {version} pronto — si installa al prossimo avvio", 0
        )
        QMessageBox.information(
            self, "Aggiornamento pronto",
            f"La versione {version} è stata scaricata.\n\n"
            "Verrà installata automaticamente al prossimo avvio di "
            "SOGNO_CANE (da START.bat).",
        )

    def _on_update_failed(self, msg: str) -> None:
        self.statusBar().showMessage(f"Aggiornamento fallito: {msg}", 8000)

    # ------------------------------------------------------------------ #
    # Configurations (save / load the complete mapping setup)            #
    # ------------------------------------------------------------------ #

    def _collect_devices(self) -> dict:
        return {
            key: {
                "profile": dev.profile.name,
                "port": dev.current_port_name(),
                "loop": dev._loop_cb.isChecked(),
                "bundle": dev.bundle,
            }
            for key, dev in (("human", self._human), ("dog", self._dog))
        }

    def _devices_state(self) -> dict:
        """What a configuration holds, to tell whether it was modified.

        The MIDI port is left out on purpose: it depends on which ports
        exist on this PC right now, not on what the user changed.
        """
        return {
            key: {
                "profile": dev.profile.name,
                "loop": dev._loop_cb.isChecked(),
                "bundle": appconfig.serialize_bundle(dev.bundle),
            }
            for key, dev in (("human", self._human), ("dog", self._dog))
        }

    def _config_name(self) -> str:
        if not self._config_path:
            return "predefinita"
        return os.path.splitext(os.path.basename(self._config_path))[0]

    def _config_dirty(self) -> bool:
        return self._devices_state() != self._config_snapshot

    def _refresh_config_view(self) -> None:
        """Show which configuration is in use and whether it changed."""
        name = self._config_name()
        dirty = self._config_dirty()
        if (name, dirty) == self._shown_config:
            return
        self._shown_config = (name, dirty)
        never_saved = not self._config_path
        if dirty:
            state = ("● non salvata" if never_saved
                     else "● modifiche non salvate")
            colour = ACCENT_PINK
        else:
            state = ("impostazioni di partenza" if never_saved
                     else "salvata")
            colour = TEXT_DIM
        self._cfg_name_lbl.setText(name)
        self._cfg_state_lbl.setText(state)
        self._cfg_state_lbl.setStyleSheet(f"color: {colour};")
        short = name if len(name) <= 24 else name[:23] + "…"
        dot = (f" <span style='color:{ACCENT_PINK}'>●</span>" if dirty
               else "")
        self._cfg_studio_lbl.setText(
            f"<span style='color:{TEXT_DIM}'>CONFIG</span> "
            f"<b>{html.escape(short)}</b>{dot}"
        )
        self.setWindowTitle(
            f"SOGNO_CANE {__version__} — {name}" + (" ●" if dirty else "")
        )

    def _apply_devices(self, devs: dict) -> None:
        for key, dev in (("human", self._human), ("dog", self._dog)):
            dd = devs.get(key) if isinstance(devs, dict) else None
            if not isinstance(dd, dict):
                continue
            appconfig.apply_bundle_config(dev.bundle, dd.get("bundle", {}))
            prof = dd.get("profile")
            if prof:
                dev._profile_cb.setCurrentText(prof)
            loop = dd.get("loop")
            if loop is not None:
                dev._loop_cb.setChecked(bool(loop))
            port = dd.get("port")
            if port:
                dev.set_port_name(port)
        self._rebuild_mapping_tabs()

    def _save_config(self) -> None:
        """SALVA: overwrite the configuration in use, without questions."""
        if not self._config_path:
            self._save_config_as()
            return
        self._write_config(self._config_path)

    def _save_config_as(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Salva configurazione come", "Nome della configurazione:",
            text=self._config_name() if self._config_path
            else "nuova_configurazione",
        )
        if not ok or not name.strip():
            return
        safe = "".join(
            c if (c.isalnum() or c in " -_") else "-" for c in name.strip()
        ).strip() or "config"
        path = os.path.join(appconfig.configs_dir(), f"{safe}.json")
        same = bool(self._config_path) and (
            os.path.normcase(os.path.abspath(path))
            == os.path.normcase(os.path.abspath(self._config_path))
        )
        if os.path.exists(path) and not same:
            if QMessageBox.question(
                self, "Sovrascrivere?",
                f"Esiste già una configurazione «{safe}».\n"
                "Sovrascriverla con le impostazioni di adesso?",
            ) != QMessageBox.StandardButton.Yes:
                return
        self._write_config(path)

    def _write_config(self, path: str) -> None:
        try:
            cfg = appconfig.build_config(self._collect_devices())
            appconfig.save_config(path, cfg)
        except Exception as e:
            QMessageBox.warning(self, "Salvataggio fallito", str(e))
            return
        self._config_path = path
        self._config_snapshot = self._devices_state()
        self._refresh_config_view()
        self._save_session()
        self.statusBar().showMessage(
            f"Configurazione «{self._config_name()}» salvata", 6000
        )

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Carica configurazione",
            appconfig.configs_dir(),
            "Configurazione SOGNO_CANE (*.json);;Tutti i file (*)",
        )
        if not path:
            return
        if self._config_dirty() and QMessageBox.question(
            self, "Modifiche non salvate",
            f"Le modifiche a «{self._config_name()}» non sono salvate e "
            f"andrebbero perse.\nCaricare comunque "
            f"«{os.path.splitext(os.path.basename(path))[0]}»?",
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            data = appconfig.load_config(path)
        except Exception as e:
            QMessageBox.warning(self, "Caricamento fallito", str(e))
            return
        self._apply_devices(data.get("devices", {}))
        self._config_path = path
        self._config_snapshot = self._devices_state()
        self._refresh_config_view()
        self._save_session()
        self.statusBar().showMessage(
            f"Configurazione «{self._config_name()}» caricata", 6000
        )

    def _save_session(self) -> None:
        """Remember the configuration in use and the working state."""
        try:
            appconfig.save_session(
                appconfig.build_config(self._collect_devices()),
                self._config_path,
            )
        except Exception:
            pass

    def _restore_session(self) -> None:
        """Reopen with the configuration that was in use, as it was left.

        The saved file is applied first and becomes the reference for
        "modified"; the working state is applied on top, so changes that
        were never saved come back too, flagged as unsaved.
        """
        data, path = appconfig.load_session()
        if data is None:
            return
        if path and os.path.exists(path):
            try:
                self._apply_devices(
                    appconfig.load_config(path).get("devices", {})
                )
                self._config_snapshot = self._devices_state()
            except Exception:
                path = None
        else:
            path = None
        self._apply_devices(data.get("devices", {}))
        self._config_path = path

    # ------------------------------------------------------------------ #
    # Spectrum                                                           #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Combined 3D view and projection                                    #
    # ------------------------------------------------------------------ #

    def _segmented(self, row, caption, options, tip, on_pick) -> dict:
        """A caption and mutually exclusive buttons, like SORGENTE."""
        lbl = QLabel(caption)
        lbl.setStyleSheet(f"color: {TEXT_DIM}; letter-spacing: 2px;")
        row.addWidget(lbl)
        group = QButtonGroup(self)
        group.setExclusive(True)
        buttons = {}
        for key, text in options:
            btn = QPushButton(text)
            btn.setObjectName("ModeButton")
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _on, k=key: on_pick(k))
            group.addButton(btn)
            row.addWidget(btn)
            buttons[key] = btn
        buttons[options[0][0]].setChecked(True)
        return buttons

    def _set_landscape(self, attr: str, value: str) -> None:
        """FORMA / DISPOSIZIONE of the 3D view in STUDIO."""
        setattr(self._landscape, attr, value)
        buttons = self._shape_btns if attr == "shape" else self._order_btns
        if value in buttons and not buttons[value].isChecked():
            buttons[value].setChecked(True)
        if attr == "shape" and self._style_controls is not None:
            self._style_controls.set_shape(value)     # the fabric: MESH only
        self._landscape.update()

    def _set_view_3d(self, on: bool) -> None:
        """The perspective view in place of the two trace views."""
        on = bool(on)
        if self._view3d_btn.isChecked() != on:
            self._view3d_btn.setChecked(on)        # re-enters with the same on
            return
        self._human.set_waveform_visible(not on)
        self._dog.set_waveform_visible(not on)
        self._landscape_box.setVisible(on)
        if on:
            sizes = self._studio_split.sizes()
            total = sum(sizes)
            self._studio_split.setSizes(
                [int(total * 0.30), int(total * 0.58), int(total * 0.12)]
            )
            self._landscape_timer.start()
        else:
            self._landscape_timer.stop()

    def _feed_views(self, kind: str, packet: UnicornPacket) -> None:
        if self._landscape.isVisible():
            self._landscape.push(kind, packet)
        if self._projection is not None:
            self._projection.push(kind, packet)

    def _on_visible_channels(self, kind: str, chans: list) -> None:
        self._landscape.set_visible_channels(kind, chans)
        if self._projection is not None:
            self._projection.set_visible_channels(kind, chans)

    def _on_panel_profile(self, kind: str, profile) -> None:
        self._landscape.set_profile(kind, profile)
        if self._projection is not None:
            self._projection.set_profile(kind, profile)

    def _show_style_controls(self) -> None:
        """REGOLAZIONI of the 3D view, for STUDIO and the projection."""
        if self._style_controls is None:
            ctl = ViewStyleControls(self._landscape.style, self)
            ctl.changed.connect(lambda: self._on_style_changed(ctl))
            ctl.set_shape(self._landscape.shape)
            self._style_controls = ctl
        self._style_controls.show()
        self._style_controls.raise_()
        self._style_controls.activateWindow()

    def _on_style_changed(self, source=None) -> None:
        """The REGOLAZIONI moved, here or in the projection's panel: both
        views at once, and the other panel shows the same values."""
        self._landscape.update()
        if self._projection is not None:
            self._projection.update()
        for panel in (self._style_controls, self._projection_controls):
            if panel is not None and panel is not source:
                panel.load_style()

    def _on_escape(self) -> None:
        if self._projection is not None and self._projection.isFullScreen():
            self._projection.windowed()

    def _show_projection_controls(self) -> None:
        if self._projection_controls is None:
            ctl = ProjectionControls(self._projection_settings,
                                     self._landscape.style, self)
            ctl.settings_changed.connect(self._on_projection_settings)
            ctl.toggle_requested.connect(self._toggle_projection)
            ctl.style_changed.connect(lambda: self._on_style_changed(ctl))
            self._projection_controls = ctl
        self._projection_controls.set_projecting(self._projection is not None)
        self._projection_controls.show()
        self._projection_controls.raise_()

    def _toggle_projection(self) -> None:
        if self._projection is not None:
            self._projection.close()
            return
        settings = self._projection_controls.settings() \
            if self._projection_controls is not None \
            else self._projection_settings
        self._projection_settings = settings
        win = ProjectionWindow(
            {"human": self._human.profile, "dog": self._dog.profile},
            settings,
            {"human": self._human.visible_channels(),
             "dog": self._dog.visible_channels()},
            # The projector shows the point of view set in STUDIO.
            navigation=self._landscape.navigation,
            style=self._landscape.style,
        )
        win.closed.connect(self._on_projection_closed)
        self._projection = win
        win.place(find_screen(settings.screen))
        self._sync_projection_ui()

    def _on_projection_settings(self, settings: ProjectionSettings) -> None:
        self._projection_settings = settings
        if self._projection is not None:
            self._projection.apply(settings)

    def _on_projection_closed(self) -> None:
        self._projection = None
        self._sync_projection_ui()

    def _sync_projection_ui(self) -> None:
        on = self._projection is not None
        self._projection_btn.setText("PROIEZIONE ●" if on else "PROIEZIONE")
        if self._projection_controls is not None:
            self._projection_controls.set_projecting(on)

    def _on_screen_removed(self, screen) -> None:
        """The projector was unplugged: close what was shown on it."""
        if self._projection is not None and \
                self._projection.screen() is screen:
            self._projection.close()

    def _on_human_packet(self, packet: UnicornPacket) -> None:
        self._human.push_packet_to_waveform(packet)
        self._feed_views("human", packet)
        self._human_window_buf.append(packet.eeg.astype(np.float64))
        total = sum(a.shape[0] for a in self._human_window_buf)
        win = packet.sample_rate
        while (
            self._human_window_buf
            and total - self._human_window_buf[0].shape[0] >= win
        ):
            total -= self._human_window_buf.pop(0).shape[0]

    def _on_dog_packet(self, packet: UnicornPacket) -> None:
        self._dog.push_packet_to_waveform(packet)
        self._feed_views("dog", packet)
        self._dog_window_buf.append(packet.eeg.astype(np.float64))
        total = sum(a.shape[0] for a in self._dog_window_buf)
        win = packet.sample_rate
        while (
            self._dog_window_buf
            and total - self._dog_window_buf[0].shape[0] >= win
        ):
            total -= self._dog_window_buf.pop(0).shape[0]

    def _refresh_spectra(self) -> None:
        if self._human_window_buf:
            arr = np.concatenate(self._human_window_buf, axis=0)
            if arr.shape[0] >= 64:
                bp = compute_band_powers(
                    arr, self._human.profile.sample_rate_hz,
                )
                log_means = {
                    name: float(np.log10(max(np.mean(power), 1e-9)))
                    for name, power in bp.by_band.items()
                }
                self._human.push_spectrum(log_means)
        if self._dog_window_buf:
            arr = np.concatenate(self._dog_window_buf, axis=0)
            if arr.shape[0] >= 64:
                bp = compute_band_powers(arr, self._dog.profile.sample_rate_hz)
                log_means = {
                    name: float(np.log10(max(np.mean(power), 1e-9)))
                    for name, power in bp.by_band.items()
                }
                self._dog.push_spectrum(log_means)

    # ------------------------------------------------------------------ #
    # Settings                                                           #
    # ------------------------------------------------------------------ #

    def _restore_settings(self) -> None:
        for key, dev in (("human", self._human), ("dog", self._dog)):
            prof = self._settings.get(key, "profile", default=dev.profile.name)
            try:
                dev._profile_cb.setCurrentText(prof)
            except Exception:
                pass
            port = self._settings.get(key, "port", default="")
            if port:
                dev.set_port_name(port)
            loop = self._settings.get(key, "loop", default=True)
            try:
                dev._loop_cb.setChecked(bool(loop))
            except Exception:
                pass
            shown = self._settings.get(key, "visible_channels", default=None)
            if isinstance(shown, list) and shown:
                dev.set_visible_channels(shown)
        shape = self._settings.get("display", "shape_3d", default="lines")
        if shape in self._shape_btns:
            self._set_landscape("shape", shape)
        order = self._settings.get("display", "order_3d", default="grouped")
        if order in self._order_btns:
            self._set_landscape("order", order)
        self._landscape.navigation = Navigation.from_dict(
            self._settings.get("display", "camera_3d", default=None))
        self._landscape.style = ViewStyle.from_dict(
            self._settings.get("display", "style_3d", default=None))
        if self._settings.get("display", "view_3d", default=False):
            self._set_view_3d(True)

    def _save_settings(self) -> None:
        self._settings.set("window", "w", self.width())
        self._settings.set("window", "h", self.height())
        for key, dev in (("human", self._human), ("dog", self._dog)):
            self._settings.set(key, "profile", dev.profile.name)
            self._settings.set(key, "port", dev.current_port_name())
            self._settings.set(key, "loop", dev._loop_cb.isChecked())
            self._settings.set(key, "visible_channels",
                               dev.visible_channels())
        self._settings.set("display", "view_3d",
                           self._view3d_btn.isChecked())
        self._settings.set("display", "shape_3d", self._landscape.shape)
        self._settings.set("display", "order_3d", self._landscape.order)
        self._settings.set("display", "camera_3d",
                           self._landscape.navigation.to_dict())
        self._settings.set("display", "style_3d",
                           self._landscape.style.to_dict())
        self._settings.set("projection", "settings",
                           self._projection_settings.to_dict())
        self._settings.save()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self._save_settings()
        self._save_session()
        if self._projection is not None:
            self._projection.close()
        if self._projection_controls is not None:
            self._projection_controls.close()
        self._human.shutdown()
        self._dog.shutdown()
        super().closeEvent(event)
