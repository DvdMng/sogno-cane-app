"""In-memory EEG recording model plus live capture.

A :class:`Recording` is a rectangular ``(n_samples, n_channels)`` matrix of
microvolts with metadata (name, originating profile, sample rate, channel
labels). It can be trimmed, exported to CSV, and round-tripped to a compact
``.npz`` + ``.json`` pair on disk.

A :class:`Recorder` accumulates live packets (from the simulator or a file
player) into such a recording with negligible overhead — it simply keeps a
list of the EEG chunks and concatenates them on demand.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class RecordingMeta:
    rec_id: str
    name: str
    profile: str
    channel_names: list[str]
    sample_rate_hz: float
    n_samples: int
    created: str
    source: str = "simulator"
    notes: str = ""

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate_hz <= 0:
            return 0.0
        return self.n_samples / self.sample_rate_hz


@dataclass
class Recording:
    eeg: np.ndarray              # (n_samples, n_channels) float64 microvolts
    meta: RecordingMeta

    # ------------------------------------------------------------------ #
    # Editing                                                            #
    # ------------------------------------------------------------------ #
    def trimmed(
        self,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
        new_name: str | None = None,
    ) -> "Recording":
        """Return a copy keeping only ``[start_seconds, end_seconds)``."""
        fs = self.meta.sample_rate_hz
        n = self.eeg.shape[0]
        i0 = max(0, int(round(start_seconds * fs)))
        i1 = n if end_seconds is None else min(n, int(round(end_seconds * fs)))
        if i1 <= i0:
            raise ValueError("trim range is empty")
        sliced = self.eeg[i0:i1].copy()
        meta = RecordingMeta(**{**asdict(self.meta)})
        meta.n_samples = sliced.shape[0]
        if new_name:
            meta.name = new_name
        return Recording(eeg=sliced, meta=meta)

    def renamed(self, new_name: str) -> "Recording":
        meta = RecordingMeta(**{**asdict(self.meta)})
        meta.name = new_name
        return Recording(eeg=self.eeg, meta=meta)

    # ------------------------------------------------------------------ #
    # Export                                                             #
    # ------------------------------------------------------------------ #
    def export_csv(self, path: str, include_time: bool = True) -> None:
        """Write the recording to a CSV with a header row of channel names."""
        fs = self.meta.sample_rate_hz or 1.0
        names = list(self.meta.channel_names)
        n, nc = self.eeg.shape
        if len(names) < nc:
            names = names + [f"ch{i}" for i in range(len(names), nc)]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = (["time_s"] if include_time else []) + names[:nc]
            w.writerow(header)
            if include_time:
                t = np.arange(n) / fs
                for i in range(n):
                    w.writerow(
                        [f"{t[i]:.6f}"]
                        + [f"{v:.4f}" for v in self.eeg[i].tolist()]
                    )
            else:
                for i in range(n):
                    w.writerow([f"{v:.4f}" for v in self.eeg[i].tolist()])

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def save(self, directory: str) -> tuple[str, str]:
        """Save as ``<id>.npz`` + ``<id>.json``; return both paths."""
        os.makedirs(directory, exist_ok=True)
        npz_path = os.path.join(directory, f"{self.meta.rec_id}.npz")
        json_path = os.path.join(directory, f"{self.meta.rec_id}.json")
        np.savez_compressed(
            npz_path,
            eeg=self.eeg.astype(np.float32),
            sample_rate_hz=np.float64(self.meta.sample_rate_hz),
            channel_names=np.array(self.meta.channel_names, dtype=object),
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.meta), f, indent=2)
        return npz_path, json_path

    @classmethod
    def load(cls, npz_path: str) -> "Recording":
        json_path = os.path.splitext(npz_path)[0] + ".json"
        with np.load(npz_path, allow_pickle=True) as z:
            eeg = np.asarray(z["eeg"], dtype=np.float64)
            rate = float(z["sample_rate_hz"])
            chans = [str(c) for c in z["channel_names"].tolist()]
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            meta = RecordingMeta(
                rec_id=m.get("rec_id", os.path.splitext(
                    os.path.basename(npz_path))[0]),
                name=m.get("name", "recording"),
                profile=m.get("profile", "HUMAN"),
                channel_names=m.get("channel_names", chans),
                sample_rate_hz=m.get("sample_rate_hz", rate),
                n_samples=m.get("n_samples", eeg.shape[0]),
                created=m.get("created", ""),
                source=m.get("source", "simulator"),
                notes=m.get("notes", ""),
            )
        else:
            meta = RecordingMeta(
                rec_id=os.path.splitext(os.path.basename(npz_path))[0],
                name="recording", profile="HUMAN", channel_names=chans,
                sample_rate_hz=rate, n_samples=eeg.shape[0], created="",
            )
        return cls(eeg=eeg, meta=meta)


class Recorder:
    """Accumulate live EEG packets into a :class:`Recording`."""

    def __init__(
        self,
        channel_names: list[str],
        sample_rate_hz: float,
        profile: str = "HUMAN",
        source: str = "simulator",
    ) -> None:
        self.channel_names = list(channel_names)
        self.sample_rate_hz = float(sample_rate_hz)
        self.profile = profile
        self.source = source
        self._chunks: list[np.ndarray] = []
        self._n_samples = 0

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate_hz <= 0:
            return 0.0
        return self._n_samples / self.sample_rate_hz

    def add_packet(self, packet) -> None:
        """Append a packet's EEG block (microvolts)."""
        eeg = np.asarray(packet.eeg, dtype=np.float32)
        self._chunks.append(eeg)
        self._n_samples += eeg.shape[0]

    def add_eeg(self, eeg: np.ndarray) -> None:
        eeg = np.asarray(eeg, dtype=np.float32)
        self._chunks.append(eeg)
        self._n_samples += eeg.shape[0]

    def clear(self) -> None:
        self._chunks.clear()
        self._n_samples = 0

    def is_empty(self) -> bool:
        return self._n_samples == 0

    def to_recording(
        self, rec_id: str, name: str, created: str
    ) -> Recording:
        if not self._chunks:
            eeg = np.zeros((0, len(self.channel_names)), dtype=np.float64)
        else:
            eeg = np.concatenate(self._chunks, axis=0).astype(np.float64)
        meta = RecordingMeta(
            rec_id=rec_id,
            name=name,
            profile=self.profile,
            channel_names=list(self.channel_names),
            sample_rate_hz=self.sample_rate_hz,
            n_samples=eeg.shape[0],
            created=created,
            source=self.source,
        )
        return Recording(eeg=eeg, meta=meta)
