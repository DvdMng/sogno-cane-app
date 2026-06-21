"""Per-device panel: title, waveform, spectrum, transport, recording, source.

A device panel owns one :class:`RealtimeEngine`. Its packet source is either
the synthetic simulator (default) or a recording loaded from disk. Incoming
packets can be captured to the shared archive at any time, regardless of the
source, so the user can record the simulation *or* re-record a loaded file.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sogno_cane.core.engine import RealtimeEngine
from sogno_cane.core.sources import ArrayPlaybackSource
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

        # Recording state.
        self._recorder: Recorder | None = None
        self._recording = False
        self._last_battery = 100.0

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 6, 10, 8)

        # Header: title + live status on one line.
        header_row = QHBoxLayout()
        header = QLabel(title)
        header.setObjectName("SubHeader")
        header_row.addWidget(header)
        header_row.addStretch(1)
        self._status = QLabel("IDLE")
        self._status.setStyleSheet(f"color: {FG_DIM};")
        header_row.addWidget(self._status)
        layout.addLayout(header_row)

        # Row A: PROFILE + MIDI port + REFRESH + CONNECT.
        rowA = QHBoxLayout()
        rowA.setSpacing(6)
        rowA.addWidget(QLabel("PROFILE"))
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
        self._refresh_btn = QPushButton("REFRESH")
        self._refresh_btn.setToolTip("Rescan MIDI output ports")
        self._refresh_btn.clicked.connect(self._refresh_ports)
        rowA.addWidget(self._refresh_btn)
        self._connect_btn = QPushButton("CONNECT")
        self._connect_btn.clicked.connect(self._toggle_midi)
        rowA.addWidget(self._connect_btn)
        layout.addLayout(rowA)

        # Row B: SOURCE label + LOAD FILE + USE SIM + loop.
        rowB = QHBoxLayout()
        rowB.setSpacing(6)
        rowB.addWidget(QLabel("SOURCE"))
        self._source_lbl = QLabel("SIMULATOR")
        self._source_lbl.setStyleSheet(f"color: {FG_LIME};")
        rowB.addWidget(self._source_lbl, 1)
        self._load_btn = QPushButton("LOAD FILE")
        self._load_btn.setToolTip(
            "Play back an EEG file (CSV / EDF / BDF / npz) through the mapper."
        )
        self._load_btn.clicked.connect(self._load_file)
        rowB.addWidget(self._load_btn)
        self._sim_btn = QPushButton("USE SIM")
        self._sim_btn.clicked.connect(self._use_simulator)
        self._sim_btn.setEnabled(False)
        rowB.addWidget(self._sim_btn)
        self._loop_cb = QCheckBox("loop")
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
        self._start_btn = QPushButton("START")
        self._start_btn.setObjectName("PrimaryButton")
        self._start_btn.clicked.connect(self._toggle_stream)
        rowT.addWidget(self._start_btn, 1)
        self._rec_btn = QPushButton("● REC")
        self._rec_btn.setToolTip(
            "Record the incoming stream to the archive; rename/trim/export "
            "it from the ARCHIVE tab."
        )
        self._rec_btn.clicked.connect(self._toggle_record)
        rowT.addWidget(self._rec_btn)
        self._rec_lbl = QLabel("")
        self._rec_lbl.setStyleSheet(f"color: {FG_MAGENTA};")
        self._rec_lbl.setMinimumWidth(70)
        rowT.addWidget(self._rec_lbl)
        layout.addLayout(rowT)

        # Live stats line.
        self._stats = QLabel("")
        self._stats.setStyleSheet(f"color: {FG_DIM}; font-size: 8pt;")
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
        self._midi.close()

    # ------------------------------------------------------------------ #
    # MIDI                                                                #
    # ------------------------------------------------------------------ #

    def _refresh_ports(self) -> None:
        if self._refresh_lock:
            return
        self._refresh_lock = True
        try:
            current = self._port_cb.currentText()
            self._port_cb.clear()
            ports = list_output_ports()
            if not ports:
                self._port_cb.addItem("(no MIDI ports - install loopMIDI)")
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
            self._connect_btn.setText("CONNECT")
            self._set_status("MIDI: disconnected", color=FG_YELLOW)
            return
        name = self._port_cb.currentText()
        if not name or name.startswith("(no MIDI"):
            return
        try:
            self._midi.open(name)
            self._connect_btn.setText("DISCONNECT")
            self._set_status(f"MIDI: {name}", color=FG_GOOD)
        except Exception as e:
            self._set_status(f"MIDI ERR: {e}", color=FG_MAGENTA)

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
            self, "Load EEG file",
            start_dir,
            f"EEG files ({patterns});;All files (*)",
        )
        if not path:
            return
        try:
            loaded = load_eeg(path)
        except Exception as e:
            QMessageBox.warning(
                self, "Load failed",
                f"Could not read {os.path.basename(path)}:\n{e}",
            )
            return
        if loaded.n_samples == 0:
            QMessageBox.warning(self, "Load failed", "File has no samples.")
            return
        self.play_loaded(loaded)

    def play_loaded(self, loaded, autostart: bool = True) -> None:
        """Use ``loaded`` (a LoadedEEG) as the playback source."""
        self._playback = loaded
        self._sim_btn.setEnabled(True)
        self._source_lbl.setText(
            f"FILE: {loaded.source_name}  "
            f"({loaded.n_channels}ch {loaded.sample_rate_hz:.0f}Hz "
            f"{loaded.duration_seconds:.0f}s)"
        )
        self._source_lbl.setStyleSheet(f"color: {FG_YELLOW};")
        if self.is_streaming:
            self._stop_engine()
            self._start_engine()
        elif autostart:
            self._start_engine()

    def _use_simulator(self) -> None:
        self._playback = None
        self._sim_btn.setEnabled(False)
        self._source_lbl.setText("SIMULATOR")
        self._source_lbl.setStyleSheet(f"color: {FG_LIME};")
        if self.is_streaming:
            self._stop_engine()
            self._start_engine()

    # ------------------------------------------------------------------ #
    # Streaming                                                           #
    # ------------------------------------------------------------------ #

    def _toggle_stream(self) -> None:
        if self.is_streaming:
            self._stop_engine()
        else:
            self._start_engine()

    def _build_source_and_config(self):
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
        self._start_btn.setText("STOP")
        self._start_btn.setObjectName("DangerButton")
        self._start_btn.setStyleSheet("")
        self._start_btn.style().unpolish(self._start_btn)
        self._start_btn.style().polish(self._start_btn)
        kind = "PLAYBACK" if self._playback is not None else "STREAMING"
        self._set_status(kind, color=FG_GOOD)

    def _stop_engine(self) -> None:
        if self._engine is None:
            return
        self._engine.stop()
        self._engine = None
        self._start_btn.setText("START")
        self._start_btn.setObjectName("PrimaryButton")
        self._start_btn.setStyleSheet("")
        self._start_btn.style().unpolish(self._start_btn)
        self._start_btn.style().polish(self._start_btn)
        self._set_status("STOPPED", color=FG_YELLOW)

    def _on_source_finished(self) -> None:
        # The engine already flushed held notes on exhaustion; send a
        # belt-and-suspenders all-notes-off in case a strategy held state.
        try:
            self._midi.all_notes_off()
        except Exception:
            pass
        self._engine = None
        self._start_btn.setText("START")
        self._start_btn.setObjectName("PrimaryButton")
        self._start_btn.setStyleSheet("")
        self._start_btn.style().unpolish(self._start_btn)
        self._start_btn.style().polish(self._start_btn)
        self._set_status("PLAYBACK FINISHED", color=FG_YELLOW)
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
        self._rec_btn.setText("■ STOP REC")
        self._rec_lbl.setText("REC 0.0s")
        if not self.is_streaming:
            self._start_engine()

    def _stop_record(self, save: bool = True) -> None:
        self._recording = False
        self._rec_btn.setText("● REC")
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
                f"SAVED {rec.duration_seconds:.1f}s -> archive", color=FG_GOOD
            )
        except Exception as e:
            self._set_status(f"SAVE ERR: {e}", color=FG_MAGENTA)

    def _capture_packet(self, packet: UnicornPacket) -> None:
        # Track battery for the stats line.
        try:
            self._last_battery = float(packet.battery[-1])
        except Exception:
            pass
        if self._recording and self._recorder is not None:
            self._recorder.add_packet(packet)
            dur = self._recorder.duration_seconds
            self._rec_lbl.setText(f"REC {dur:.1f}s")
            if dur >= _MAX_RECORD_SECONDS:
                self._stop_record(save=True)

    # ------------------------------------------------------------------ #
    # Stats / waveform / spectrum                                         #
    # ------------------------------------------------------------------ #

    def _update_stats(self) -> None:
        ev = self._engine.event_count if self._engine else 0
        parts = [f"battery {self._last_battery:.0f}%", f"midi events {ev}"]
        if self._playback is not None and self._engine is not None:
            src = self._engine.source
            if isinstance(src, ArrayPlaybackSource):
                parts.append(
                    f"pos {src.position_seconds:.0f}/"
                    f"{src.duration_seconds:.0f}s"
                )
        self._stats.setText("   ".join(parts))

    def _set_status(self, text: str, color: str = FG_DIM) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(
            f"color: {color}; font-size: 10pt; font-weight: 700;"
        )

    def push_packet_to_waveform(self, packet: UnicornPacket) -> None:
        self._waveform.push_packet(packet)

    def push_spectrum(self, mean_log_powers: dict[str, float]) -> None:
        self._spectrum.update_powers(mean_log_powers)
