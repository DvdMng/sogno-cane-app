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
        self.setWindowTitle("SOGNO_CANE")
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
        self._tabs.addTab(self._build_mapping(), "MAPPING")
        self._archive_panel = ArchivePanel(self._archive)
        self._tabs.addTab(self._archive_panel, "ARCHIVE")
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
            "ready — recordings are stored in " + self._archive.root
        )

        # Online auto-update wiring.
        self._manual_update_check = False
        self._update_available.connect(self._on_update_available)
        self._update_staged.connect(self._on_update_staged)
        self._update_failed.connect(self._on_update_failed)
        self._update_none.connect(self._on_update_none)
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
        self._human = DevicePanel("HUMAN", HUMAN_PROFILE, archive=self._archive)
        self._dog = DevicePanel("DOG", DOG_PROFILE, archive=self._archive)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.setHandleWidth(8)
        top.addWidget(self._human)
        top.addWidget(self._dog)
        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 1)
        v.addWidget(top, 1)

        # Compact MIDI activity monitor strip along the bottom.
        mon_label = QLabel("MIDI ACTIVITY")
        mon_label.setStyleSheet(
            "color: #7F7298; font-size: 8pt; letter-spacing: 2px;"
        )
        v.addWidget(mon_label)
        self._monitor = MidiMonitor()
        self._monitor.setFixedHeight(120)
        v.addWidget(self._monitor)

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
            "Configure how each device's EEG drives MIDI. Save the whole "
            "setup (both devices) as a named configuration."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #7F7298; padding: 4px 2px;")
        bar.addWidget(hint, 1)
        save_btn = QPushButton("SAVE CONFIG")
        save_btn.setToolTip("Save the complete configuration (both devices).")
        save_btn.clicked.connect(self._save_config)
        bar.addWidget(save_btn)
        load_btn = QPushButton("LOAD CONFIG")
        load_btn.setToolTip("Load a saved configuration.")
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
        tabs.addTab(MappingPanel(self._human.bundle, "HUMAN MAPPING"), "HUMAN")
        tabs.addTab(MappingPanel(self._dog.bundle, "DOG MAPPING"), "DOG")

    def _build_transport(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(10)
        lbl = QLabel("TRANSPORT")
        lbl.setStyleSheet("font-weight: 700; letter-spacing: 3px;")
        bar.addWidget(lbl)
        start_all = QPushButton("▶ START ALL")
        start_all.clicked.connect(self._start_all)
        bar.addWidget(start_all)
        stop_all = QPushButton("■ STOP ALL")
        stop_all.clicked.connect(self._stop_all)
        bar.addWidget(stop_all)
        panic = QPushButton("✸ PANIC")
        panic.setToolTip("Send All-Notes-Off on every MIDI channel.")
        panic.clicked.connect(self._panic)
        bar.addWidget(panic)
        bar.addStretch(1)
        self._update_btn = QPushButton("UPDATES")
        self._update_btn.setToolTip("Check online for a newer version")
        self._update_btn.clicked.connect(lambda: self._start_update_check(True))
        bar.addWidget(self._update_btn)
        help_btn = QPushButton("?")
        help_btn.setFixedWidth(36)
        help_btn.setToolTip("Quick help")
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

    def _panic(self) -> None:
        self._human.panic()
        self._dog.panic()
        self.statusBar().showMessage("PANIC — all notes off", 2000)

    def _show_help(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, "SOGNO_CANE — quick help",
            "STUDIO\n"
            "  • Pick a PROFILE (human/dog) per panel.\n"
            "  • CONNECT a MIDI port (install loopMIDI on Windows).\n"
            "  • START streams the simulator; LOAD FILE plays a recording\n"
            "    (CSV / EDF / BDF / npz) through the same mapping.\n"
            "  • ● REC captures the incoming signal to the archive.\n"
            "  • Tune musicality in the mapping tabs (interval / hold /\n"
            "    change threshold control how sparse each voice is).\n\n"
            "ARCHIVE\n"
            "  • Rename, CUT a time range, EXPORT CSV, DELETE, or\n"
            "    LOAD a recording back into a device for playback.",
        )

    def _on_tab_changed(self, idx: int) -> None:
        if self._tabs.tabText(idx) == "ARCHIVE":
            self._archive_panel.refresh()

    # ------------------------------------------------------------------ #
    # Online updates                                                     #
    # ------------------------------------------------------------------ #

    def _notify_pending_update(self) -> None:
        pend = updater.has_pending_update()
        if pend:
            self.statusBar().showMessage(
                f"Update {pend.get('version','?')} staged — restart to apply",
                8000,
            )

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
            self.statusBar().showMessage("Checking for updates…", 4000)
        threading.Thread(target=self._check_thread, daemon=True).start()

    def _check_thread(self) -> None:
        info = updater.check_for_update(current=__version__)
        if info is not None:
            self._update_available.emit(info)
        elif self._manual_update_check:
            self._update_none.emit()

    def _on_update_none(self) -> None:
        QMessageBox.information(
            self, "Updates",
            f"You are up to date (version {__version__}).",
        )

    def _on_update_available(self, info) -> None:
        auto = updater.auto_update_enabled()
        if auto and not self._manual_update_check:
            self.statusBar().showMessage(
                f"Downloading update {info.version}…", 0
            )
            self._download_update(info)
            return
        notes = f"\n\n{info.notes}" if info.notes else ""
        if QMessageBox.question(
            self, "Update available",
            f"Version {info.version} is available "
            f"(you have {__version__}).{notes}\n\n"
            "Download now? It will be applied at the next launch.",
        ) == QMessageBox.StandardButton.Yes:
            self.statusBar().showMessage(
                f"Downloading update {info.version}…", 0
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
            f"Update {version} ready — applies on next launch", 0
        )
        QMessageBox.information(
            self, "Update ready",
            f"Version {version} has been downloaded.\n\n"
            "It will be installed automatically the next time you start "
            "SOGNO_CANE (via START.bat).",
        )

    def _on_update_failed(self, msg: str) -> None:
        self.statusBar().showMessage(f"Update failed: {msg}", 8000)

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
            self, "Save configuration", "Configuration name:",
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
            self.statusBar().showMessage(f"Saved configuration: {path}", 6000)
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load configuration",
            appconfig.configs_dir(),
            "SOGNO_CANE config (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            data = appconfig.load_config(path)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))
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
            f"Loaded configuration: {os.path.basename(path)}", 6000
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
