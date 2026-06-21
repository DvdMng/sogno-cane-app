"""Load EEG recordings from disk into a uniform :class:`LoadedEEG`.

Supported formats (auto-detected by extension, with content sniffing for
delimited text):

* ``.edf`` / ``.bdf`` / ``.rec``  - European Data Format and BioSemi 24-bit.
* ``.csv`` / ``.tsv`` / ``.txt``  - delimited text; an optional header row of
  channel names and an optional leading time column are detected.
* ``.npz``                        - native SOGNO_CANE recording archive.
* ``.npy``                        - raw ``(n_samples, n_channels)`` array.

Everything is returned as float64 microvolts in a single rectangular matrix,
together with the sample rate (inferred from a time column when present) and
channel labels, ready to drive playback through the normal MIDI pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from sogno_cane.io.edf import read_edf

_TIME_NAMES = {"time", "timestamp", "t", "seconds", "sec", "s"}
_DEFAULT_RATE = 250.0

# Extensions we advertise in file-open dialogs.
SUPPORTED_EXTENSIONS = (
    ".edf", ".bdf", ".rec", ".csv", ".tsv", ".txt", ".npz", ".npy",
)


@dataclass
class LoadedEEG:
    data: np.ndarray                       # (n_samples, n_channels) float64
    sample_rate_hz: float
    channel_names: list[str]
    source_name: str
    physical_dim: str = "uV"
    meta: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1]) if self.data.ndim == 2 else 0

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate_hz <= 0:
            return 0.0
        return self.n_samples / self.sample_rate_hz


def load_eeg(path: str, default_rate_hz: float = _DEFAULT_RATE) -> LoadedEEG:
    """Load any supported EEG file into a :class:`LoadedEEG`."""
    ext = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)
    if ext in (".edf", ".bdf", ".rec"):
        return _from_edf(path, name)
    if ext == ".npz":
        return _from_npz(path, name, default_rate_hz)
    if ext == ".npy":
        return _from_npy(path, name, default_rate_hz)
    # Treat everything else as delimited text.
    return _from_text(path, name, default_rate_hz)


# --------------------------------------------------------------------------- #
# EDF / BDF                                                                    #
# --------------------------------------------------------------------------- #
def _from_edf(path: str, name: str) -> LoadedEEG:
    edf = read_edf(path)
    data = edf.data
    # Convert to microvolts if the physical dimension is millivolts/volts.
    dims = [d.lower() for d in edf.physical_dim]
    scale = np.ones(data.shape[1], dtype=np.float64)
    for j, d in enumerate(dims):
        if d in ("mv",):
            scale[j] = 1000.0
        elif d in ("v",):
            scale[j] = 1_000_000.0
    data = data * scale[None, :]
    return LoadedEEG(
        data=data.astype(np.float64),
        sample_rate_hz=edf.sample_rate_hz,
        channel_names=edf.channel_names,
        source_name=name,
        meta={"format": "bdf" if edf.is_bdf else "edf",
              "start": edf.start_datetime},
    )


# --------------------------------------------------------------------------- #
# Native npz / raw npy                                                         #
# --------------------------------------------------------------------------- #
def _from_npz(path: str, name: str, default_rate: float) -> LoadedEEG:
    with np.load(path, allow_pickle=True) as z:
        data = np.asarray(z["eeg"], dtype=np.float64)
        if data.ndim == 1:
            data = data[:, None]
        rate = float(z["sample_rate_hz"]) if "sample_rate_hz" in z else default_rate
        if "channel_names" in z:
            chans = [str(c) for c in z["channel_names"].tolist()]
        else:
            chans = [f"ch{i}" for i in range(data.shape[1])]
    return LoadedEEG(
        data=data, sample_rate_hz=rate, channel_names=chans,
        source_name=name, meta={"format": "npz"},
    )


def _from_npy(path: str, name: str, default_rate: float) -> LoadedEEG:
    data = np.asarray(np.load(path), dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    return LoadedEEG(
        data=data, sample_rate_hz=default_rate,
        channel_names=[f"ch{i}" for i in range(data.shape[1])],
        source_name=name, meta={"format": "npy"},
    )


# --------------------------------------------------------------------------- #
# Delimited text                                                              #
# --------------------------------------------------------------------------- #
def _sniff_delimiter(sample: str) -> str:
    for delim in ("\t", ",", ";", " "):
        if delim in sample:
            return delim
    return ","


def _from_text(path: str, name: str, default_rate: float) -> LoadedEEG:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = [ln.rstrip("\n\r") for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"{name}: file is empty")

    delim = _sniff_delimiter(lines[0])

    def split(ln: str) -> list[str]:
        if delim == " ":
            return ln.split()
        return [c.strip() for c in ln.split(delim)]

    first = split(lines[0])

    def is_number(tok: str) -> bool:
        try:
            float(tok)
            return True
        except ValueError:
            return False

    header: list[str] | None = None
    if not all(is_number(t) for t in first):
        header = first
        body = lines[1:]
    else:
        body = lines

    rows = []
    for ln in body:
        toks = split(ln)
        try:
            rows.append([float(t) for t in toks])
        except ValueError:
            continue  # skip malformed lines
    if not rows:
        raise ValueError(f"{name}: no numeric rows found")

    width = min(len(r) for r in rows)
    arr = np.array([r[:width] for r in rows], dtype=np.float64)

    # Detect a leading time/index column. When a header is present we trust
    # only an explicit time name; otherwise we accept a column that clearly
    # looks like a (fractional) time axis. We never strip an integer-valued
    # first column, since that is almost always real channel data.
    time_col = None
    if header is not None and header:
        if header[0].lower() in _TIME_NAMES:
            time_col = 0
    elif arr.shape[1] > 1 and _looks_like_time(arr[:, 0]):
        time_col = 0

    rate = default_rate
    if time_col is not None:
        t = arr[:, time_col]
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        if dt > 0:
            rate = _infer_rate_from_dt(dt, default_rate)
        data = np.delete(arr, time_col, axis=1)
        chans = (
            [header[i] for i in range(len(header)) if i != time_col]
            if header else [f"ch{i}" for i in range(data.shape[1])]
        )
    else:
        data = arr
        chans = header if header else [f"ch{i}" for i in range(data.shape[1])]

    chans = [str(c) for c in chans][: data.shape[1]]
    while len(chans) < data.shape[1]:
        chans.append(f"ch{len(chans)}")

    return LoadedEEG(
        data=data, sample_rate_hz=float(rate), channel_names=chans,
        source_name=name, meta={"format": "text", "delimiter": delim},
    )


def _infer_rate_from_dt(dt: float, default_rate: float) -> float:
    """Infer the sample rate from a median timestep of unknown unit.

    A timestep of e.g. 4.0 is ambiguous: 4 s (-> 0.25 Hz) or 4 ms (-> 250 Hz).
    We try seconds, milliseconds and microseconds and pick the unit whose
    resulting rate is physiologically plausible for EEG (1 Hz .. 20 kHz),
    preferring the largest unit (coarsest) that qualifies. This fixes the old
    ``dt > 5`` heuristic that mis-read >200 Hz millisecond data as ~0.25 Hz.
    """
    for scale in (1.0, 1000.0, 1_000_000.0):  # s, ms, us
        rate = scale / dt
        if 1.0 <= rate <= 20_000.0:
            return rate
    # Nothing plausible; fall back to a seconds interpretation.
    return 1.0 / dt if dt > 0 else default_rate


def _looks_like_time(col: np.ndarray) -> bool:
    """A monotonically increasing first column that looks like a time axis.

    Fractional, steadily-increasing columns are accepted. Integer columns are
    accepted only with a *strong* time signature — start at ~0, near-constant
    step, and enough rows — so that ordinary integer channel data (a very
    common headerless layout) is not mistaken for a time column.
    """
    if len(col) < 3:
        return False
    d = np.diff(col)
    if np.any(d <= 0):
        return False
    steady = float(np.std(d) / (np.mean(d) + 1e-12)) < 0.25
    if not steady:
        return False
    is_integer = bool(np.allclose(col, np.round(col)))
    if is_integer:
        # Only an integer ramp that starts at ~0 over many rows is treated as
        # a time axis (e.g. a 0,4,8,... millisecond column).
        return len(col) >= 16 and abs(float(col[0])) <= 1.0
    return True
