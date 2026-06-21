"""On-disk archive of EEG recordings.

The archive is just a directory of ``<id>.npz`` + ``<id>.json`` pairs. It
provides listing, rename, trim, delete, and CSV export — the operations the
UI archive panel needs. IDs are timestamp-based and collision-safe.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict

from sogno_cane.io.recording import Recording, RecordingMeta


def _default_archive_dir() -> str:
    base = os.environ.get("SOGNO_CANE_HOME")
    if not base:
        base = os.path.join(
            os.path.expanduser("~"), ".sogno_cane"
        )
    return os.path.join(base, "recordings")


class Archive:
    """Manage a folder of saved recordings."""

    def __init__(self, root: str | None = None) -> None:
        self.root = root or _default_archive_dir()
        os.makedirs(self.root, exist_ok=True)
        self._counter = 0

    # ------------------------------------------------------------------ #
    # IDs                                                                #
    # ------------------------------------------------------------------ #
    def new_id(self) -> str:
        self._counter += 1
        stamp = time.strftime("%Y%m%d-%H%M%S")
        rec_id = f"rec-{stamp}-{self._counter:03d}"
        # Guarantee uniqueness even within the same second.
        while os.path.exists(os.path.join(self.root, f"{rec_id}.npz")):
            self._counter += 1
            rec_id = f"rec-{stamp}-{self._counter:03d}"
        return rec_id

    def now_iso(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------ #
    # CRUD                                                               #
    # ------------------------------------------------------------------ #
    def list(self) -> list[RecordingMeta]:
        """Return metadata for every recording, newest first."""
        out: list[RecordingMeta] = []
        for fn in os.listdir(self.root):
            if not fn.endswith(".npz"):
                continue
            try:
                rec = Recording.load(os.path.join(self.root, fn))
                out.append(rec.meta)
            except Exception:
                continue
        out.sort(key=lambda m: m.created, reverse=True)
        return out

    def npz_path(self, rec_id: str) -> str:
        return os.path.join(self.root, f"{rec_id}.npz")

    def exists(self, rec_id: str) -> bool:
        return os.path.exists(self.npz_path(rec_id))

    def save(self, recording: Recording) -> str:
        if not recording.meta.rec_id:
            recording.meta.rec_id = self.new_id()
        if not recording.meta.created:
            recording.meta.created = self.now_iso()
        recording.save(self.root)
        return recording.meta.rec_id

    def load(self, rec_id: str) -> Recording:
        return Recording.load(self.npz_path(rec_id))

    def rename(self, rec_id: str, new_name: str) -> None:
        rec = self.load(rec_id)
        rec.meta.name = new_name
        rec.save(self.root)

    def set_notes(self, rec_id: str, notes: str) -> None:
        rec = self.load(rec_id)
        rec.meta.notes = notes
        rec.save(self.root)

    def delete(self, rec_id: str) -> None:
        for ext in (".npz", ".json"):
            p = os.path.join(self.root, f"{rec_id}{ext}")
            if os.path.exists(p):
                os.remove(p)

    def trim(
        self,
        rec_id: str,
        start_seconds: float,
        end_seconds: float | None,
        in_place: bool = False,
        new_name: str | None = None,
    ) -> str:
        """Trim a recording. Returns the id of the (possibly new) recording."""
        rec = self.load(rec_id)
        trimmed = rec.trimmed(start_seconds, end_seconds, new_name=new_name)
        if in_place:
            trimmed.meta.rec_id = rec_id
            trimmed.meta.created = rec.meta.created
            trimmed.save(self.root)
            return rec_id
        trimmed.meta.rec_id = self.new_id()
        trimmed.meta.created = self.now_iso()
        if not new_name:
            trimmed.meta.name = f"{rec.meta.name} (trim)"
        trimmed.save(self.root)
        return trimmed.meta.rec_id

    def export_csv(
        self, rec_id: str, dest_path: str, include_time: bool = True
    ) -> str:
        rec = self.load(rec_id)
        rec.export_csv(dest_path, include_time=include_time)
        return dest_path

    def import_recording(self, recording: Recording) -> str:
        """Store an externally-built recording (e.g. from a loaded file)."""
        recording.meta.rec_id = self.new_id()
        recording.meta.created = self.now_iso()
        recording.save(self.root)
        return recording.meta.rec_id
