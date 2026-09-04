"""Per-device panel: title, waveform, spectrum, transport, recording, source.

A device panel owns one :class:`RealtimeEngine`. Its packet source is either
the synthetic simulator (default) or a recording loaded from disk. Incoming
packets can be captured to the shared archive at any time, regardless of the
source, so the user can record the simulation *or* re-record a loaded file.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sogno_cane.core.engine import RealtimeEngine
from sogno_cane.core.sources import ArrayPlaybackSource
from sogno_cane.eeg.live import (
    ALT_UDP_PORTS,
    DEFAULT_UDP_PORT,
    UnicornLiveSource,
    probe_udp_ports,
)
from sogno_cane.eeg.profiles import ALL_PROFILES, DeviceProfile
from sogno_cane.eeg.unicorn_packet import UnicornPacket
from sogno_cane.io.archive import Archive
from sogno_cane.io.loaders import SUPPORTED_EXTENSIONS, load_eeg
from sogno_cane.io.recording import Recorder
from sogno_cane.midi.mapper import MappingConfig
from sogno_cane.midi.output import MidiOutput, list_output_ports
from sogno_cane.midi.presets import PresetBundle, rich_vocabulary_preset
from sogno_cane.ui.spectrum_widget import SpectrumWidget
from sogno_cane.ui.theme import (
    ACCENT_CYAN as FG_LIME,
    ACCENT_PINK as FG_MAGENTA,
    NEON_LIME as FG_GOOD,
    NEON_YELLOW as FG_YELLOW,
    TEXT_DIM as FG_DIM,
)
from sogno_cane.ui.waveform_widget import WaveformWidget

# Soft cap on a single recording so memory never runs away (minutes).
_MAX_RECORD_SECONDS = 30 * 60


class DevicePanel(QWidget):
    """Single-device control + viz panel."""

    packet_received = Signal(object)         # UnicornPacket
    event_emitted = Signal(object)           # MappingEvent
    recording_saved = Signal(str)            # rec_id
    source_finished = Signal()               # non-looping playback ended

    def __init__(
        self,
        title: str,
        default_profile: DeviceProfile,
        archive: Archive | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._profile = default_profile
        self._archive = archive or Archive()
        self._bundle: PresetBundle = rich_vocabulary_preset()
        self._engine: RealtimeEngine | None = None
        self._midi = MidiOutput()
        self._refresh_lock = False

        # Source state.
        self._playback = None          # LoadedEEG when a file is loaded
        self._loop_playback = True
        # Live acquisition from a real Unicorn (UDP), when connected.
        self._live: UnicornLiveSource | None = None
        self._live_warned_no_data = False

        # Recording state.
        self._recorder: Recorder | None = None
        self._recording = False
        self._last_battery = 100.0

        # MIDI health surfaced by the stats timer.
        self._midi_stall_shown = False
        self._known_reconnects = 0
        self._no_midi_warned = False

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 6, 10, 8)

        # Per-device identity: the two panels are otherwise identical twins,
        # and picking the wrong one mid-performance is an easy mistake.
        is_dog = "DOG" in title.upper() or "CANE" in title.upper()
        self.setObjectName("DevicePanelDog" if is_dog else "DevicePanelHuman")
        # Without this a plain QWidget silently ignores the stylesheet's
        # background and border, and the accent stripe never appears.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        # Header: title + live status on one line.
        header_row = QHBoxLayout()
        header = QLabel(title)
        header.setObjectName("SubHeaderDog" if is_dog else "SubHeaderHuman")
        header_row.addWidget(header)
        header_row.addStretch(1)
        self._status = QLabel("FERMO")
        self._status.setStyleSheet(f"color: {FG_DIM};")
        header_row.addWidget(self._status)
        layout.addLayout(header_row)

        # Row A: PROFILE + MIDI port + REFRESH + CONNECT.
        rowA = QHBoxLayout()
        rowA.setSpacing(6)
        rowA.addWidget(QLabel("PROFILO"))
        self._profile_cb = QComboBox()
        self._profile_cb.addItems(list(ALL_PROFILES.keys()))
        self._profile_cb.setCurrentText(default_profile.name)
        self._profile_cb.setMinimumWidth(86)
        self._profile_cb.currentTextChanged.connect(self._on_profile_changed)
        rowA.addWidget(self._profile_cb)
        rowA.addSpacing(6)
        rowA.addWidget(QLabel("MIDI"))
        self._port_cb = QComboBox()
        self._port_cb.setEditable(False)
        self._port_cb.setMinimumWidth(130)
        rowA.addWidget(self._port_cb, 1)
        self._refresh_btn = QPushButton("RICERCA")
        self._refresh_btn.setToolTip("Rileggi l'elenco delle porte MIDI")
        self._refresh_btn.clicked.connect(self._refresh_ports)
        rowA.addWidget(self._refresh_btn)
        self._connect_btn = QPushButton("COLLEGA")
        # Connecting MIDI is the prerequisite for hearing anything, so it
        # asks for attention until it is done, then steps back.
        self._connect_btn.setObjectName("CallToAction")
        self._connect_btn.clicked.connect(self._toggle_midi)
        rowA.addWidget(self._connect_btn)
        layout.addLayout(rowA)

        # Row B: the three sources are mutually exclusive -> a segmented
        # control, so the active one is visible without reading a label.
        rowB = QHBoxLayout()
        rowB.setSpacing(6)
        rowB.addWidget(QLabel("SORGENTE"))
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)

        self._sim_btn = QPushButton("SIMULATORE")
        self._sim_btn.setObjectName("ModeButton")
        self._sim_btn.setCheckable(True)
        self._sim_btn.setChecked(True)
        self._sim_btn.setToolTip(
            "Segnale EEG sintetico: per provare il set senza indossare nulla."
        )
        self._sim_btn.clicked.connect(self._on_mode_sim)

        self._load_btn = QPushButton("FILE")
        self._load_btn.setObjectName("ModeButton")
        self._load_btn.setCheckable(True)
        self._load_btn.setToolTip(
            "Riproduci una registrazione EEG (CSV / EDF / BDF / npz) "
            "attraverso la stessa mappatura."
        )
        self._load_btn.clicked.connect(self._on_mode_file)

        self._live_btn = QPushButton("EEG DAL VIVO")
        self._live_btn.setObjectName("ModeButton")
        self._live_btn.setCheckable(True)
        self._live_btn.setToolTip(
            "Collega un Unicorn vero. Avvia prima il programma g.tec "
            "(Unicorn BCI Core Recorder -> Settings -> Network -> 'Sending "
            "data via UDP', oppure Unicorn Suite -> DevTools -> "
            "UnicornUDP.exe), poi premi qui: porta e formato dei canali "
            "vengono riconosciuti da soli."
        )
        self._live_btn.clicked.connect(self._on_mode_live)

        for b in (self._sim_btn, self._load_btn, self._live_btn):
            self._mode_group.addButton(b)
            rowB.addWidget(b)

        # Detail line: only says something the buttons cannot (which file,
        # which UDP port). Empty for the simulator, where the lit segment
        # is already the whole story.
        self._source_lbl = QLabel("")
        self._source_lbl.setStyleSheet(f"color: {FG_LIME};")
        rowB.addWidget(self._source_lbl, 1)
        self._loop_cb = QCheckBox("ripeti")
        self._loop_cb.setToolTip(
            "Riparti da capo quando il file finisce (solo sorgente FILE)."
        )
        self._loop_cb.setChecked(True)
        self._loop_cb.toggled.connect(self._on_loop_toggled)
        rowB.addWidget(self._loop_cb)
        layout.addLayout(rowB)

        # Waveform — the centrepiece, gets all the spare vertical space.
        self._waveform = WaveformWidget(self._profile)
        layout.addWidget(self._waveform, 1)

        # Spectrum — compact strip beneath the waveform.
        self._spectrum = SpectrumWidget()
        layout.addWidget(self._spectrum)

        # Transport row: START + REC + recording timer.
        rowT = QHBoxLayout()
        rowT.setSpacing(6)
        self._start_btn = QPushButton("AVVIA")
        self._start_btn.setObjectName("PrimaryButton")
        self._start_btn.clicked.connect(self._toggle_stream)
        rowT.addWidget(self._start_btn, 1)
        self._rec_btn = QPushButton("● REGISTRA")
        self._rec_btn.setToolTip(
            "Registra il segnale in arrivo nell'archivio; poi lo rinomini, "
            "tagli o esporti dalla scheda ARCHIVIO."
        )
        self._rec_btn.clicked.connect(self._toggle_record)
        rowT.addWidget(self._rec_btn)
        self._rec_lbl = QLabel("")
        self._rec_lbl.setStyleSheet(f"color: {FG_MAGENTA};")
        self._rec_lbl.setMinimumWidth(70)
        rowT.addWidget(self._rec_lbl)
        layout.addLayout(rowT)

        # Health strip: on stage this answers "is it actually working?",
        # so it is readable and colour-coded instead of 8pt grey.
        self._stats = QLabel("")
        self._stats.setObjectName("HealthLine")
        self._stats.setStyleSheet(f"color: {FG_DIM};")
        layout.addWidget(self._stats)

        # Waveform repaint timer.
        self._repaint_timer = QTimer(self)
        self._repaint_timer.setInterval(33)   # ~30 fps
        self._repaint_timer.timeout.connect(self._waveform.repaint_traces)
        self._repaint_timer.start()

        # Live stats timer.
        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(500)
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_timer.start()

        # Capture incoming packets for recording (main-thread, queued).
        self.packet_received.connect(self._capture_packet)
        self.source_finished.connect(self._on_source_finished)

        self._refresh_ports()

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def bundle(self) -> PresetBundle:
        return self._bundle

    @property
    def profile(self) -> DeviceProfile:
        return self._profile

    @property
    def is_streaming(self) -> bool:
        return self._engine is not None and self._engine.is_running

    def set_port_name(self, name: str) -> None:
        if name:
            idx = self._port_cb.findText(name)
            if idx >= 0:
                self._port_cb.setCurrentIndex(idx)

    def current_port_name(self) -> str:
        return self._port_cb.currentText()

    def panic(self) -> None:
        """All-notes-off on every channel."""
        try:
            self._midi.all_notes_off()
        except Exception:
            pass

    def shutdown(self) -> None:
        if self._recording:
            self._stop_record(save=True)
        if self._engine is not None and self._engine.is_running:
            self._engine.stop()
        if self._live is not None:
            self._live.close()
            self._live = None
        self._midi.close()

    # ------------------------------------------------------------------ #
    # MIDI                                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _restyle(widget, object_name: str) -> None:
        """Swap a widget's QSS identity and force Qt to repaint it."""
        widget.setObjectName(object_name)
        widget.setStyleSheet("")
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _refresh_ports(self) -> None:
        if self._refresh_lock:
            return
        self._refresh_lock = True
        try:
            current = self._port_cb.currentText()
            self._port_cb.clear()
            ports = list_output_ports()
            if not ports:
                self._port_cb.addItem("(nessuna porta MIDI - installa loopMIDI)")
                self._port_cb.setEnabled(False)
                self._connect_btn.setEnabled(False)
            else:
                self._port_cb.addItems(ports)
                self._port_cb.setEnabled(True)
                self._connect_btn.setEnabled(True)
                if current in ports:
                    self._port_cb.setCurrentText(current)
        finally:
            self._refresh_lock = False

    def _toggle_midi(self) -> None:
        if self._midi.is_open:
            self._midi.close()
            self._connect_btn.setText("COLLEGA")
            self._restyle(self._connect_btn, "CallToAction")
            self._set_status("MIDI scollegato", color=FG_YELLOW)
            return
        name = self._port_cb.currentText()
        if not name or name.startswith("(nessuna porta"):
            return
        try:
            self._midi.open(name)
            self._connect_btn.setText("SCOLLEGA")
            self._restyle(self._connect_btn, "")
            self._set_status(f"MIDI: {name}", color=FG_GOOD)
        except Exception as e:
            self._set_status(f"ERRORE MIDI: {e}", color=FG_MAGENTA)

    # ------------------------------------------------------------------ #
    # Profile / source                                                   #
    # ------------------------------------------------------------------ #

    def _on_profile_changed(self, profile_name: str) -> None:
        was_running = self.is_streaming
        if was_running:
            self._stop_engine()
        self._profile = ALL_PROFILES[profile_name]
        try:
            self._repaint_timer.timeout.disconnect()
        except Exception:
            pass
        idx = self.layout().indexOf(self._waveform)
        self.layout().removeWidget(self._waveform)
        self._waveform.deleteLater()
        self._waveform = WaveformWidget(self._profile)
        self.layout().insertWidget(idx, self._waveform, 1)
        self._repaint_timer.timeout.connect(self._waveform.repaint_traces)
        if was_running:
            self._start_engine()

    def _on_loop_toggled(self, v: bool) -> None:
        self._loop_playback = bool(v)
        if self._playback is not None and self.is_streaming:
            # Apply on the live source if possible.
            src = self._engine.source if self._engine else None
            if isinstance(src, ArrayPlaybackSource):
                src.loop = self._loop_playback

    def _load_file(self) -> None:
        patterns = " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS)
        start_dir = self._archive.root
        path, _ = QFileDialog.getOpenFileName(
            self, "Apri un file EEG",
            start_dir,
            f"File EEG ({patterns});;Tutti i file (*)",
        )
        if not path:
            return
        try:
            loaded = load_eeg(path)
        except Exception as e:
            QMessageBox.warning(
                self, "Apertura fallita",
                f"Non riesco a leggere {os.path.basename(path)}:\n{e}",
            )
            return
        if loaded.n_samples == 0:
            QMessageBox.warning(self, "Apertura fallita", "Il file non contiene campioni.")
            return
        self.play_loaded(loaded)

    def play_loaded(self, loaded, autostart: bool = True) -> None:
        """Use ``loaded`` (a LoadedEEG) as the playback source."""
        if self._live is not None:
            self._disconnect_live(restart=False)
        self._playback = loaded
        self._source_lbl.setText(
            f"{loaded.source_name}  "
            f"({loaded.n_channels}ch {loaded.sample_rate_hz:.0f}Hz "
            f"{loaded.duration_seconds:.0f}s)"
        )
        self._source_lbl.setStyleSheet(f"color: {FG_YELLOW};")
        self._sync_mode_buttons()
        if self.is_streaming:
            self._stop_engine()
            self._start_engine()
        elif autostart:
            self._start_engine()

    def _use_simulator(self) -> None:
        self._playback = None
        self._disconnect_live(restart=False)
        self._source_lbl.setText("")
        self._sync_mode_buttons()
        if self.is_streaming:
            self._stop_engine()
            self._start_engine()

    # ------------------------------------------------------------------ #
    # Source mode (segmented control)                                    #
    # ------------------------------------------------------------------ #

    def current_mode(self) -> str:
        """Which source is actually feeding the engine: sim / file / live."""
        if self._live is not None:
            return "live"
        if self._playback is not None:
            return "file"
        return "sim"

    def _sync_mode_buttons(self) -> None:
        """Make the segmented control mirror reality.

        The buttons express intent; this pulls them back to what actually
        happened, so a cancelled file dialog or a failed live connection
        cannot leave the UI claiming a source that is not running.
        """
        mode = self.current_mode()
        for btn, name in (
            (self._sim_btn, "sim"),
            (self._load_btn, "file"),
            (self._live_btn, "live"),
        ):
            btn.blockSignals(True)
            btn.setChecked(name == mode)
            btn.blockSignals(False)
        self._loop_cb.setEnabled(mode == "file")

    def _on_mode_sim(self) -> None:
        if self.current_mode() != "sim":
            self._use_simulator()
        self._sync_mode_buttons()

    def _on_mode_file(self) -> None:
        self._load_file()          # may be cancelled by the user
        self._sync_mode_buttons()

    def _on_mode_live(self) -> None:
        if self._live is None:
            self._connect_live()   # may be cancelled or fail
        self._sync_mode_buttons()

    # ------------------------------------------------------------------ #
    # Live acquisition (real Unicorn over UDP)                           #
    # ------------------------------------------------------------------ #

    @property
    def live_source(self) -> UnicornLiveSource | None:
        return self._live

    def _toggle_live(self) -> None:
        if self._live is not None:
            self._disconnect_live(restart=True)
        else:
            self._connect_live()

    def _connect_live(self, port: int | None = None) -> None:
        """Find the g.tec sender and start consuming it.

        Plug and play: we listen on the documented ports for a moment and
        adopt whichever one is actually carrying Unicorn datagrams, so the
        user only has to have started the g.tec tool.
        """
        if port is None:
            port = self._discover_port()
            if port is None:
                return
        try:
            live = UnicornLiveSource(port=int(port), packet_samples=25)
            live.open()
        except OSError as exc:
            QMessageBox.warning(
                self, "EEG dal vivo",
                f"Non riesco ad ascoltare sulla porta UDP {port}:\n{exc}\n\n"
                f"Un'altra applicazione (o l'altro pannello) la sta gia' "
                f"usando. Scegli una porta diversa nel programma g.tec "
                f"e riprova.",
            )
            return
        self._live = live
        self._live_warned_no_data = False
        self._playback = None
        self._source_lbl.setText(f"LIVE UNICORN  (UDP {port})")
        self._source_lbl.setStyleSheet(f"color: {FG_GOOD};")
        self._set_status(f"EEG DAL VIVO su UDP {port}", color=FG_GOOD)
        self._sync_mode_buttons()
        if self.is_streaming:
            self._stop_engine()
        self._start_engine()

    def _discover_port(self) -> int | None:
        """Probe the known ports; ask the user only if that is inconclusive."""
        self._set_status("Cerco il segnale dell'Unicorn…", color=FG_YELLOW)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            QApplication.processEvents()
            found = probe_udp_ports(ports=ALT_UDP_PORTS, timeout=1.5)
        finally:
            QApplication.restoreOverrideCursor()

        if len(found) == 1:
            return int(found[0]["port"])
        if len(found) > 1:
            labels = [
                f"UDP {f['port']}  —  {f['channels']} canali ({f['layout']})"
                for f in found
            ]
            choice, ok = QInputDialog.getItem(
                self, "EEG dal vivo",
                "Ho trovato più di un segnale Unicorn.\n"
                "Scegli quello di questo dispositivo:",
                labels, 0, False,
            )
            if not ok:
                self._set_status("FERMO", color=FG_DIM)
                return None
            return int(found[labels.index(choice)]["port"])

        # Nothing heard: let the user name the port, and say why.
        self._set_status("Nessun segnale Unicorn trovato", color=FG_YELLOW)
        port, ok = QInputDialog.getInt(
            self, "EEG dal vivo",
            "Non arrivano dati Unicorn sulle porte solite.\n\n"
            "Avvia prima il programma g.tec:\n"
            "  • Unicorn BCI Core Recorder → Settings → Network →\n"
            "    'Sending data via UDP'  (porta predefinita 1001), oppure\n"
            "  • Unicorn Suite → DevTools → UnicornUDP.exe\n\n"
            "Usa una porta diversa per ogni dispositivo (es. 1001 umano,\n"
            "1002 cane), poi conferma la porta da ascoltare:",
            DEFAULT_UDP_PORT, 1, 65535, 1,
        )
        if not ok:
            self._set_status("FERMO", color=FG_DIM)
            return None
        return int(port)

    def _disconnect_live(self, restart: bool = True) -> None:
        live, self._live = self._live, None
        if live is None:
            return
        was_streaming = self.is_streaming
        if was_streaming:
            self._stop_engine()
        live.close()
        self._source_lbl.setText("")
        self._set_status("EEG dal vivo interrotto", color=FG_YELLOW)
        self._sync_mode_buttons()
        if restart and was_streaming:
            self._start_engine()

    # ------------------------------------------------------------------ #
    # Streaming                                                           #
    # ------------------------------------------------------------------ #

    def _toggle_stream(self) -> None:
        if self.is_streaming:
            self._stop_engine()
        else:
            self._start_engine()

    def _warn_if_no_midi(self) -> None:
        """Starting without a MIDI port produces silence and no error.

        Deliberately NOT a modal dialog: a blocking popup is the last thing
        anyone wants mid-performance. The panel says it loudly instead —
        here, in the pink health strip, and in the still-yellow COLLEGA
        button — which is impossible to miss and never interrupts.
        """
        if self._midi.is_open:
            return
        self._no_midi_warned = True
        self._set_status(
            "NESSUNA PORTA MIDI — non uscirà alcun suono", color=FG_MAGENTA
        )

    def _build_source_and_config(self):
        if self._live is not None:
            # The live socket is bound once and reused across stop/start:
            # rebinding on every START would race with the OS releasing the
            # port and would drop samples for no reason.
            cfg = MappingConfig(
                sample_rate_hz=int(self._live.sample_rate_hz),
                window_seconds=1.0,
                hop_seconds=0.1,
            )
            return self._live, cfg
        if self._playback is not None:
            src = ArrayPlaybackSource(
                self._playback.data,
                self._playback.sample_rate_hz,
                packet_samples=25,
                loop=self._loop_playback,
            )
            cfg = MappingConfig(
                sample_rate_hz=int(round(self._playback.sample_rate_hz)),
                window_seconds=1.0,
                hop_seconds=0.1,
            )
            return src, cfg
        cfg = MappingConfig(
            sample_rate_hz=self._profile.sample_rate_hz,
            window_seconds=1.0,
            hop_seconds=0.1,
        )
        return None, cfg

    def _start_engine(self) -> None:
        source, mapping_cfg = self._build_source_and_config()
        engine = RealtimeEngine(
            profile=self._profile,
            mapping_config=mapping_cfg,
            source=source,
            midi=self._midi,
            on_packet=lambda p: self.packet_received.emit(p),
            on_event=lambda e: self.event_emitted.emit(e),
            on_finished=lambda: self.source_finished.emit(),
        )
        for s in self._bundle.as_list():
            engine.mapper.add_strategy(s)
        self._engine = engine
        engine.start()
        self._start_btn.setText("FERMA")
        self._start_btn.setObjectName("DangerButton")
        self._start_btn.setStyleSheet("")
        self._start_btn.style().unpolish(self._start_btn)
        self._start_btn.style().polish(self._start_btn)
        if self._midi.is_open:
            kind = (
                "RIPRODUZIONE" if self._playback is not None else "IN ONDA"
            )
            self._set_status(kind, color=FG_GOOD)
        else:
            self._warn_if_no_midi()

    def _stop_engine(self) -> None:
        if self._engine is None:
            return
        self._engine.stop()
        self._engine = None
        self._start_btn.setText("AVVIA")
        self._start_btn.setObjectName("PrimaryButton")
        self._start_btn.setStyleSheet("")
        self._start_btn.style().unpolish(self._start_btn)
        self._start_btn.style().polish(self._start_btn)
        self._set_status("FERMO", color=FG_YELLOW)

    def _on_source_finished(self) -> None:
        # The engine already flushed held notes on exhaustion; stop() also
        # parks its note-off scheduler thread so nothing leaks, then send a
        # belt-and-suspenders all-notes-off in case a strategy held state.
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass
        try:
            self._midi.all_notes_off()
        except Exception:
            pass
        self._engine = None
        self._start_btn.setText("AVVIA")
        self._start_btn.setObjectName("PrimaryButton")
        self._start_btn.setStyleSheet("")
        self._start_btn.style().unpolish(self._start_btn)
        self._start_btn.style().polish(self._start_btn)
        self._set_status("FILE TERMINATO", color=FG_YELLOW)
        if self._recording:
            self._stop_record(save=True)

    # ------------------------------------------------------------------ #
    # Recording                                                           #
    # ------------------------------------------------------------------ #

    def _toggle_record(self) -> None:
        if self._recording:
            self._stop_record(save=True)
        else:
            self._start_record()

    def _start_record(self) -> None:
        src = (
            f"file:{self._playback.source_name}"
            if self._playback is not None else "simulator"
        )
        sr = (
            self._playback.sample_rate_hz
            if self._playback is not None
            else self._profile.sample_rate_hz
        )
        self._recorder = Recorder(
            channel_names=list(self._profile.channel_names),
            sample_rate_hz=sr,
            profile=self._profile.name,
            source=src,
        )
        self._recording = True
        self._rec_btn.setText("■ FERMA REG.")
        self._rec_lbl.setText("REG 0.0s")
        if not self.is_streaming:
            self._start_engine()

    def _stop_record(self, save: bool = True) -> None:
        self._recording = False
        self._rec_btn.setText("● REGISTRA")
        rec = self._recorder
        self._recorder = None
        self._rec_lbl.setText("")
        if not save or rec is None or rec.is_empty():
            return
        name = f"{self._profile.name} {self._archive.now_iso()}"
        recording = rec.to_recording(
            self._archive.new_id(), name, self._archive.now_iso()
        )
        try:
            rid = self._archive.save(recording)
            self.recording_saved.emit(rid)
            self._set_status(
                f"SALVATO {rec.duration_seconds:.1f}s in archivio", color=FG_GOOD
            )
        except Exception as e:
            self._set_status(f"ERRORE SALVATAGGIO: {e}", color=FG_MAGENTA)

    def _capture_packet(self, packet: UnicornPacket) -> None:
        # Track battery for the stats line.
        try:
            self._last_battery = float(packet.battery[-1])
        except Exception:
            pass
        if self._recording and self._recorder is not None:
            self._recorder.add_packet(packet)
            dur = self._recorder.duration_seconds
            self._rec_lbl.setText(f"REG {dur:.1f}s")
            if dur >= _MAX_RECORD_SECONDS:
                self._stop_record(save=True)

    # ------------------------------------------------------------------ #
    # Stats / waveform / spectrum                                         #
    # ------------------------------------------------------------------ #

    def _update_stats(self) -> None:
        """Render the health strip: signal in, MIDI out, and any trouble.

        Written as two plain claims ("EEG ..." then "MIDI ...") because
        during a performance the only questions are whether data is coming
        in and whether notes are going out. The colour is the summary: green
        while both are true, yellow while idle, pink when something is wrong.
        """
        parts: list[str] = []
        trouble = False
        idle = not self.is_streaming

        # --- signal in ---
        if self._live is not None:
            lst = self._live.stats
            if self._live.has_signal:
                parts.append(f"EEG dal vivo {lst['samples']} campioni")
            else:
                parts.append("EEG dal vivo NESSUN DATO")
                trouble = True
            if lst["dropped"]:
                parts.append(f"scartati {lst['dropped']}")
            if lst["malformed"]:
                parts.append(f"corrotti {lst['malformed']}")
                trouble = True
        elif self._playback is not None and self._engine is not None:
            src = self._engine.source
            if isinstance(src, ArrayPlaybackSource):
                parts.append(
                    f"FILE {src.position_seconds:.0f}/"
                    f"{src.duration_seconds:.0f}s"
                )
            else:
                parts.append("FILE")
        else:
            parts.append(f"SIMULATORE batteria {self._last_battery:.0f}%")

        # --- MIDI out ---
        ev = self._engine.event_count if self._engine else 0
        if self._midi.is_open:
            st = self._midi.stats
            if st["failed"]:
                parts.append(
                    f"MIDI {st['sent']} inviati, {st['failed']} FALLITI"
                )
                trouble = True
            else:
                parts.append(f"MIDI {st['sent']} inviati")
            if st["reconnects"]:
                parts.append(f"riconnessioni {st['reconnects']}")
        else:
            parts.append(f"MIDI NON COLLEGATO ({ev} eventi non udibili)")
            trouble = True

        self._stats.setText("     ".join(parts))
        if trouble:
            colour = FG_MAGENTA
        elif idle:
            colour = FG_DIM
        else:
            colour = FG_GOOD
        self._stats.setStyleSheet(f"color: {colour};")
        self._surface_live_health()
        self._surface_midi_health()

    def _surface_live_health(self) -> None:
        """Say plainly whether EEG is actually arriving on the wire."""
        live = self._live
        if live is None:
            return
        if live.has_signal:
            if self._live_warned_no_data:
                self._live_warned_no_data = False
                self._set_status(
                    f"EEG DAL VIVO su UDP {live.port}", color=FG_GOOD
                )
            return
        if not self._live_warned_no_data:
            self._live_warned_no_data = True
            age = live.seconds_since_data
            detail = (
                "nessun dato" if age is None
                else f"silenzio da {age:.0f}s"
            )
            self._set_status(
                f"EEG DAL VIVO — {detail} (il programma g.tec sta trasmettendo?)",
                color=FG_MAGENTA,
            )

    def _surface_midi_health(self) -> None:
        """Tell the truth about the wire: sends failing vs recovered.

        Without this the app looks alive (the event monitor keeps moving —
        events are generated regardless) while nothing reaches loopMIDI.
        """
        if not self._midi.is_open:
            self._midi_stall_shown = False
            return
        st = self._midi.stats
        if self._midi.is_stalled:
            self._midi_stall_shown = True
            self._set_status(
                "MIDI BLOCCATO — riconnessione…", color=FG_MAGENTA
            )
            return
        if st["reconnects"] > self._known_reconnects:
            self._known_reconnects = st["reconnects"]
            self._midi_stall_shown = False
            self._set_status(
                f"MIDI: {self._midi.port_name} (riconnesso)", color=FG_GOOD
            )
        elif self._midi_stall_shown:
            # Stall cleared without a full reconnect (transient failure).
            self._midi_stall_shown = False
            self._set_status(f"MIDI: {self._midi.port_name}", color=FG_GOOD)

    def _set_status(self, text: str, color: str = FG_DIM) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(
            f"color: {color}; font-size: 10pt; font-weight: 700;"
        )

    def push_packet_to_waveform(self, packet: UnicornPacket) -> None:
        self._waveform.push_packet(packet)

    def push_spectrum(self, mean_log_powers: dict[str, float]) -> None:
        self._spectrum.update_powers(mean_log_powers)
