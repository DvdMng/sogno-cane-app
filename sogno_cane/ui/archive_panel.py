"""Archive browser: rename, trim, export to CSV, delete, and load to play.

Lists every recording saved to the shared :class:`Archive` and exposes the
edit operations requested for the archive: rename, cut (trim a start/end
range), export to CSV, delete, plus "load to HUMAN/DOG" which sends the
recording back through the live playback engine, and "load session" which
plays a UOMO+CANE pair recorded together on both panels at once.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sogno_cane.io.archive import Archive
from sogno_cane.io.loaders import LoadedEEG
from sogno_cane.io.recording import Recording
from sogno_cane.ui.theme import ACCENT_CYAN, ACCENT_PINK, NEON_LIME, TEXT_DIM


def _recording_to_loaded(rec: Recording) -> LoadedEEG:
    return LoadedEEG(
        data=rec.eeg,
        sample_rate_hz=rec.meta.sample_rate_hz,
        channel_names=list(rec.meta.channel_names),
        source_name=rec.meta.name,
        meta={"rec_id": rec.meta.rec_id, "format": "archive"},
    )


class ArchivePanel(QWidget):
    """Recordings manager."""

    # Emitted when the user asks to play a recording on a device.
    load_to_human = Signal(object)   # LoadedEEG
    load_to_dog = Signal(object)     # LoadedEEG
    # A UOMO+CANE session: (human LoadedEEG or None, dog LoadedEEG or None).
    load_session = Signal(object, object)

    def __init__(
        self, archive: Archive, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._archive = archive
        self._ids: list[str] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        title = QLabel("ARCHIVIO REGISTRAZIONI")
        title.setObjectName("SubHeader")
        outer.addWidget(title)

        hint = QLabel(
            "Si registra dalla scheda STUDIO: ● REGISTRA su un pannello, "
            "oppure ● REGISTRA UOMO+CANE per entrambi nello stesso istante "
            "(le due registrazioni hanno lo stesso nome di sessione). "
            "Seleziona una riga per rinominarla, tagliarla, esportarla in CSV, "
            "eliminarla o ricaricarla per riascoltarla."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {TEXT_DIM};")
        outer.addWidget(hint)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["NOME", "PROFILO", "DURATA", "SORGENTE", "CREATA"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 5):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_selection)
        outer.addWidget(self._table, 1)

        # Trim controls.
        trim_row = QHBoxLayout()
        trim_row.addWidget(QLabel("TAGLIA  da"))
        self._trim_start = QDoubleSpinBox()
        self._trim_start.setRange(0.0, 99999.0)
        self._trim_start.setSuffix(" s")
        self._trim_start.setDecimals(2)
        trim_row.addWidget(self._trim_start)
        trim_row.addWidget(QLabel("a"))
        self._trim_end = QDoubleSpinBox()
        self._trim_end.setRange(0.0, 99999.0)
        self._trim_end.setSuffix(" s")
        self._trim_end.setDecimals(2)
        trim_row.addWidget(self._trim_end)
        self._trim_btn = QPushButton("TAGLIA -> NUOVA")
        self._trim_btn.setToolTip(
            "Salva l'intervallo scelto [da, a) come nuova registrazione."
        )
        self._trim_btn.clicked.connect(self._do_trim)
        trim_row.addWidget(self._trim_btn)
        trim_row.addStretch(1)
        outer.addLayout(trim_row)

        # Action buttons.
        actions = QHBoxLayout()
        self._rename_btn = QPushButton("RINOMINA")
        self._rename_btn.clicked.connect(self._do_rename)
        actions.addWidget(self._rename_btn)
        self._csv_btn = QPushButton("ESPORTA CSV")
        self._csv_btn.clicked.connect(self._do_export_csv)
        actions.addWidget(self._csv_btn)
        self._del_btn = QPushButton("ELIMINA")
        self._del_btn.clicked.connect(self._do_delete)
        actions.addWidget(self._del_btn)
        actions.addStretch(1)
        self._play_human_btn = QPushButton("CARICA → UMANO")
        self._play_human_btn.clicked.connect(
            lambda: self._do_load("HUMAN")
        )
        actions.addWidget(self._play_human_btn)
        self._play_dog_btn = QPushButton("CARICA → CANE")
        self._play_dog_btn.clicked.connect(lambda: self._do_load("DOG"))
        actions.addWidget(self._play_dog_btn)
        self._play_session_btn = QPushButton("CARICA SESSIONE → UMANO + CANE")
        self._play_session_btn.setToolTip(
            "Per le registrazioni fatte con ● REGISTRA UOMO+CANE: carica la\n"
            "coppia sui due pannelli e la fa partire insieme."
        )
        self._play_session_btn.clicked.connect(self._do_load_session)
        actions.addWidget(self._play_session_btn)
        self._refresh_btn = QPushButton("AGGIORNA")
        self._refresh_btn.clicked.connect(self.refresh)
        actions.addWidget(self._refresh_btn)
        outer.addLayout(actions)

        self._set_actions_enabled(False)
        self.refresh()

    # ------------------------------------------------------------------ #

    def refresh(self) -> None:
        metas = self._archive.list()
        self._metas = metas
        self._ids = [m.rec_id for m in metas]
        self._table.setRowCount(len(metas))
        for r, m in enumerate(metas):
            cells = [
                m.name,
                m.profile,
                f"{m.duration_seconds:.1f} s",
                m.source,
                m.created,
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setForeground(_qcolor(ACCENT_CYAN))
                self._table.setItem(r, c, item)
        self._set_actions_enabled(False)

    def _selected_id(self) -> str | None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return None
        idx = rows[0].row()
        if 0 <= idx < len(self._ids):
            return self._ids[idx]
        return None

    def _on_selection(self) -> None:
        rec_id = self._selected_id()
        self._set_actions_enabled(rec_id is not None)
        self._play_session_btn.setEnabled(
            rec_id is not None and len(self._session_of(rec_id)) >= 2
        )
        if rec_id is None:
            return
        try:
            meta = next(
                m for m in self._archive.list() if m.rec_id == rec_id
            )
            self._trim_start.setValue(0.0)
            self._trim_end.setValue(round(meta.duration_seconds, 2))
            self._trim_start.setMaximum(meta.duration_seconds)
            self._trim_end.setMaximum(meta.duration_seconds)
        except StopIteration:
            pass

    def _set_actions_enabled(self, on: bool) -> None:
        for b in (
            self._rename_btn, self._csv_btn, self._del_btn,
            self._trim_btn, self._play_human_btn, self._play_dog_btn,
        ):
            b.setEnabled(on)
        if not on:
            self._play_session_btn.setEnabled(False)

    def _session_of(self, rec_id: str) -> list:
        """The recordings started together with ``rec_id`` (itself too)."""
        meta = next((m for m in getattr(self, "_metas", [])
                     if m.rec_id == rec_id), None)
        if meta is None or not meta.session:
            return []
        return [m for m in self._metas if m.session == meta.session]

    # ------------------------------------------------------------------ #
    # Actions                                                            #
    # ------------------------------------------------------------------ #

    def _do_rename(self) -> None:
        rec_id = self._selected_id()
        if not rec_id:
            return
        cur = self._archive.load(rec_id).meta.name
        name, ok = QInputDialog.getText(
            self, "Rinomina registrazione", "Nuovo nome:", text=cur
        )
        if ok and name.strip():
            self._archive.rename(rec_id, name.strip())
            self.refresh()

    def _do_export_csv(self) -> None:
        rec_id = self._selected_id()
        if not rec_id:
            return
        meta = self._archive.load(rec_id).meta
        default = os.path.join(
            _default_download_dir(),
            f"{_safe_filename(meta.name)}.csv",
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Esporta in CSV", default, "File CSV (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            self._archive.export_csv(rec_id, path)
            QMessageBox.information(
                self, "Esportata", f"CSV salvato:\n{path}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Esportazione fallita", str(e))

    def _do_delete(self) -> None:
        rec_id = self._selected_id()
        if not rec_id:
            return
        meta = self._archive.load(rec_id).meta
        if QMessageBox.question(
            self, "Elimina registrazione",
            f"Eliminare «{meta.name}»? Non si può annullare.",
        ) == QMessageBox.StandardButton.Yes:
            self._archive.delete(rec_id)
            self.refresh()

    def _do_trim(self) -> None:
        rec_id = self._selected_id()
        if not rec_id:
            return
        start = self._trim_start.value()
        end = self._trim_end.value()
        if end <= start:
            QMessageBox.warning(
                self, "Intervallo non valido",
                "La fine deve venire dopo l'inizio.",
            )
            return
        try:
            new_id = self._archive.trim(rec_id, start, end)
            self.refresh()
            self._select_id(new_id)
        except Exception as e:
            QMessageBox.warning(self, "Taglio fallito", str(e))

    def _do_load(self, which: str) -> None:
        rec_id = self._selected_id()
        if not rec_id:
            return
        try:
            rec = self._archive.load(rec_id)
        except Exception as e:
            QMessageBox.warning(self, "Caricamento fallito", str(e))
            return
        loaded = _recording_to_loaded(rec)
        if which == "HUMAN":
            self.load_to_human.emit(loaded)
        else:
            self.load_to_dog.emit(loaded)

    def _do_load_session(self) -> None:
        """Send both halves of a UOMO+CANE session to their panels."""
        rec_id = self._selected_id()
        members = self._session_of(rec_id) if rec_id else []
        if len(members) < 2:
            return
        human = dog = None
        for meta in members:
            # The panel it was recorded on decides; the profile is the
            # fallback for recordings without one.
            is_dog = (meta.panel.upper() == "CANE" if meta.panel
                      else meta.profile.upper() == "DOG")
            try:
                loaded = _recording_to_loaded(
                    self._archive.load(meta.rec_id)
                )
            except Exception as e:
                QMessageBox.warning(self, "Caricamento fallito", str(e))
                return
            if is_dog and dog is None:
                dog = loaded
            elif not is_dog and human is None:
                human = loaded
        self.load_session.emit(human, dog)

    def _select_id(self, rec_id: str) -> None:
        if rec_id in self._ids:
            self._table.selectRow(self._ids.index(rec_id))


def _qcolor(hex_str: str):
    from PySide6.QtGui import QColor
    return QColor(hex_str)


# Characters that are illegal in Windows filenames (also unsafe elsewhere).
_ILLEGAL_FILENAME = r'<>:"/\|?*'


def _safe_filename(name: str) -> str:
    """Turn a recording name into a filesystem-safe filename stem.

    Recording names embed a timestamp like ``HUMAN 2026-06-21 13:50:22`` whose
    colons are illegal in Windows filenames; without sanitisation the export
    save dialog produces an invalid path and the write fails with OSError.
    """
    out = []
    for ch in name.strip():
        if ch in _ILLEGAL_FILENAME or ord(ch) < 32:
            out.append("-")
        elif ch == " ":
            out.append("_")
        else:
            out.append(ch)
    cleaned = "".join(out).strip("._-")
    return cleaned or "recording"


def _default_download_dir() -> str:
    """Prefer the user's Downloads folder for exports, else the home dir."""
    home = os.path.expanduser("~")
    downloads = os.path.join(home, "Downloads")
    return downloads if os.path.isdir(downloads) else home
