"""Main application window for SOGNO_CANE."""
from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
import os

from PySide6.QtWidgets import (
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
from sogno_cane.ui.mapping_panel import MappingPanel
from sogno_cane.ui.midi_monitor import MidiMonitor
from sogno_cane.ui.theme import BackgroundWidget, ChromeTitle, app_icon


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
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Spectrum refresh timer.
        self._human_window_buf: list[np.ndarray] = []
        self._dog_window_buf: list[np.ndarray] = []
        self._spectrum_timer = QTimer(self)
        self._spectrum_timer.setInterval(250)
        self._spectrum_timer.timeout.connect(self._refresh_spectra)
        self._spectrum_timer.start()

        self._restore_settings()
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

    # ------------------------------------------------------------------ #
    # Layout                                                             #
    # ------------------------------------------------------------------ #

    def _build_studio(self) -> QWidget:
        studio = QWidget()
        v = QVBoxLayout(studio)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        # Global transport bar.
        v.addLayout(self._build_transport())

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

        self._studio_split = QSplitter(Qt.Orientation.Vertical)
        self._studio_split.setHandleWidth(8)
        self._studio_split.addWidget(top)
        self._studio_split.addWidget(mon_wrap)
        self._studio_split.setStretchFactor(0, 5)
        self._studio_split.setStretchFactor(1, 1)
        self._studio_split.setSizes([760, 110])
        v.addWidget(self._studio_split, 1)

        # Wire device events.
        self._human.packet_received.connect(self._on_human_packet)
        self._dog.packet_received.connect(self._on_dog_packet)
        self._human.event_emitted.connect(self._monitor.push_event)
        self._dog.event_emitted.connect(self._monitor.push_event)
        return studio

    def _build_mapping(self) -> QWidget:
        """Mapping configuration for both devices, in its own tab."""
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)

        # Config save/load bar.
        bar = QHBoxLayout()
        hint = QLabel(
            "Come l'EEG di ogni dispositivo pilota il MIDI. I canali MIDI sono "
            "1–16 (come in Ableton). Puoi salvare tutto come configurazione."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #7F7298; padding: 4px 2px;")
        bar.addWidget(hint, 1)
        save_btn = QPushButton("SALVA CONFIG")
        save_btn.setToolTip("Salva l'intera configurazione (entrambi i dispositivi).")
        save_btn.clicked.connect(self._save_config)
        bar.addWidget(save_btn)
        load_btn = QPushButton("CARICA CONFIG")
        load_btn.setToolTip("Carica una configurazione salvata.")
        load_btn.clicked.connect(self._load_config)
        bar.addWidget(load_btn)
        v.addLayout(bar)

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

    def _build_transport(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(10)
        lbl = QLabel("TRASPORTO")
        lbl.setStyleSheet("font-weight: 700; letter-spacing: 3px;")
        bar.addWidget(lbl)
        start_all = QPushButton("AVVIA TUTTO")
        start_all.clicked.connect(self._start_all)
        bar.addWidget(start_all)
        stop_all = QPushButton("FERMA TUTTO")
        stop_all.clicked.connect(self._stop_all)
        bar.addWidget(stop_all)
        panic = QPushButton("PANIC")
        panic.setToolTip("Manda All-Notes-Off su tutti i canali MIDI.")
        panic.clicked.connect(self._panic)
        bar.addWidget(panic)
        swap = QPushButton("SCAMBIA EEG")
        swap.setToolTip(
            "Se il segnale dell'uomo è finito sul pannello CANE (e viceversa)\n"
            "scambia i due EEG dal vivo tra i pannelli, senza ricollegare."
        )
        swap.clicked.connect(self._swap_live_eeg)
        bar.addWidget(swap)
        bar.addStretch(1)
        self._version_lbl = QLabel(f"v{__version__}")
        self._version_lbl.setObjectName("VersionTag")
        self._version_lbl.setToolTip(
            "Versione installata. Premi AGGIORNAMENTI per cercarne una nuova."
        )
        bar.addWidget(self._version_lbl)
        self._update_btn = QPushButton("AGGIORNAMENTI")
        self._update_btn.setToolTip("Cerca online una versione più recente")
        self._update_btn.clicked.connect(lambda: self._start_update_check(True))
        bar.addWidget(self._update_btn)
        help_btn = QPushButton("?")
        help_btn.setFixedWidth(36)
        help_btn.setToolTip("Aiuto rapido")
        help_btn.clicked.connect(self._show_help)
        bar.addWidget(help_btn)
        return bar

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
            "REGISTRARE E RIASCOLTARE\n"
            "  • ● REGISTRA cattura il segnale in arrivo nell'archivio.\n"
            "  • Dalla scheda ARCHIVIO puoi rinominare, tagliare un\n"
            "    intervallo, esportare in CSV, eliminare o ricaricare una\n"
            "    registrazione in un dispositivo.\n\n"
            "MUSICALITÀ (scheda MAPPATURA)\n"
            "  • intervallo / durata minima / soglia di cambiamento\n"
            "    regolano quanto è rada ogni voce;\n"
            "  • 'note length' quanto dura ogni nota (15 ms = trigger per\n"
            "    gli inviluppi, 0 = HOLD legato);\n"
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

    def _save_config(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Salva configurazione", "Nome della configurazione:",
            text="my_config",
        )
        if not ok or not name.strip():
            return
        safe = "".join(
            c if (c.isalnum() or c in " -_") else "-" for c in name.strip()
        ).strip() or "config"
        path = os.path.join(appconfig.configs_dir(), f"{safe}.json")
        try:
            cfg = appconfig.build_config(self._collect_devices())
            appconfig.save_config(path, cfg)
            self.statusBar().showMessage(f"Configurazione salvata: {path}", 6000)
        except Exception as e:
            QMessageBox.warning(self, "Salvataggio fallito", str(e))

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Carica configurazione",
            appconfig.configs_dir(),
            "Configurazione SOGNO_CANE (*.json);;Tutti i file (*)",
        )
        if not path:
            return
        try:
            data = appconfig.load_config(path)
        except Exception as e:
            QMessageBox.warning(self, "Caricamento fallito", str(e))
            return
        devs = data.get("devices", {})
        for key, dev in (("human", self._human), ("dog", self._dog)):
            dd = devs.get(key)
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
        self.statusBar().showMessage(
            f"Configurazione caricata: {os.path.basename(path)}", 6000
        )

    # ------------------------------------------------------------------ #
    # Spectrum                                                           #
    # ------------------------------------------------------------------ #

    def _on_human_packet(self, packet: UnicornPacket) -> None:
        self._human.push_packet_to_waveform(packet)
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

    def _save_settings(self) -> None:
        self._settings.set("window", "w", self.width())
        self._settings.set("window", "h", self.height())
        for key, dev in (("human", self._human), ("dog", self._dog)):
            self._settings.set(key, "profile", dev.profile.name)
            self._settings.set(key, "port", dev.current_port_name())
            self._settings.set(key, "loop", dev._loop_cb.isChecked())
        self._settings.save()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self._save_settings()
        self._human.shutdown()
        self._dog.shutdown()
        super().closeEvent(event)
