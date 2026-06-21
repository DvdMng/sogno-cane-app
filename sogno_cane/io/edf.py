"""Minimal, dependency-free EDF / EDF+ / BDF reader.

EDF (European Data Format) and its 24-bit BioSemi variant BDF are the de-facto
standard containers for clinical and research EEG. The format is fully
specified and small, so we parse it directly rather than pulling in a heavy
dependency such as ``mne`` or ``pyedflib`` — keeping the portable Windows
bundle self-contained.

Reference: Kemp & Olivan, "European data format 'plus' (EDF+)" (2003) and
https://www.edfplus.info/specs/edf.html .

The reader returns a plain :class:`EdfData` with a float64 ``(n_samples,
n_channels)`` matrix in physical units (microvolts for EEG), the common
sample rate, and channel labels. Signals that are sampled at different rates
inside the file are linearly resampled to the file's maximum rate so the
result is a single rectangular matrix. The reserved ``EDF Annotations``
channel is skipped.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np


@dataclass
class EdfData:
    data: np.ndarray            # (n_samples, n_channels) float64, physical units
    sample_rate_hz: float
    channel_names: list[str]
    physical_dim: list[str]
    is_bdf: bool
    start_datetime: str         # "dd.mm.yy hh.mm.ss" as stored


def _ascii(b: bytes) -> str:
    return b.decode("ascii", errors="replace").strip()


def _read_exact(f: BinaryIO, n: int) -> bytes:
    buf = f.read(n)
    if len(buf) != n:
        raise ValueError(
            f"Truncated EDF/BDF file: wanted {n} bytes, got {len(buf)}"
        )
    return buf


def read_edf(path: str) -> EdfData:
    """Read an EDF/EDF+/BDF file into an :class:`EdfData`."""
    with open(path, "rb") as f:
        version_raw = _read_exact(f, 8)
        is_bdf = version_raw[:1] == b"\xff"
        # Validate the magic/version so non-EDF files fail with a clear error
        # instead of a confusing parse error deep in the header.
        if is_bdf:
            if version_raw[1:8] != b"BIOSEMI":
                raise ValueError("Not a valid BDF file (bad BIOSEMI magic)")
        else:
            if version_raw.strip() not in (b"0", b""):
                raise ValueError(
                    "Not a valid EDF/BDF file (unrecognised version field)"
                )

        _patient = _ascii(_read_exact(f, 80))
        _recording = _ascii(_read_exact(f, 80))
        startdate = _ascii(_read_exact(f, 8))
        starttime = _ascii(_read_exact(f, 8))
        _header_bytes = int(_ascii(_read_exact(f, 8)) or "0")
        _reserved = _read_exact(f, 44)
        n_records = int(_ascii(_read_exact(f, 8)) or "-1")
        record_duration = float(_ascii(_read_exact(f, 8)) or "1")
        ns = int(_ascii(_read_exact(f, 4)) or "0")
        if ns <= 0:
            raise ValueError("EDF/BDF header declares zero signals")

        labels = [_ascii(_read_exact(f, 16)) for _ in range(ns)]
        _transducer = [_read_exact(f, 80) for _ in range(ns)]
        phys_dim = [_ascii(_read_exact(f, 8)) for _ in range(ns)]
        phys_min = [float(_ascii(_read_exact(f, 8))) for _ in range(ns)]
        phys_max = [float(_ascii(_read_exact(f, 8))) for _ in range(ns)]
        dig_min = [float(_ascii(_read_exact(f, 8))) for _ in range(ns)]
        dig_max = [float(_ascii(_read_exact(f, 8))) for _ in range(ns)]
        _prefilter = [_read_exact(f, 80) for _ in range(ns)]
        samples_per_record = [
            int(_ascii(_read_exact(f, 8))) for _ in range(ns)
        ]
        _sig_reserved = [_read_exact(f, 32) for _ in range(ns)]

        bytes_per_sample = 3 if is_bdf else 2
        record_size = sum(samples_per_record) * bytes_per_sample

        # If the record count is unknown (-1), infer it from file size.
        if n_records < 0:
            cur = f.tell()
            f.seek(0, 2)
            end = f.tell()
            f.seek(cur)
            if record_size > 0:
                n_records = (end - cur) // record_size
            else:
                n_records = 0

        raw_blob = f.read(n_records * record_size)

    # Recover gracefully from a truncated file: keep only whole records that
    # actually arrived rather than crashing on the incomplete tail.
    if record_size > 0:
        available = len(raw_blob) // record_size
        if available < n_records:
            n_records = available
            raw_blob = raw_blob[: n_records * record_size]
    if n_records <= 0:
        raise ValueError("EDF/BDF file contains no complete data records")

    # Decode all data records into per-signal sample streams.
    offsets = np.cumsum([0] + samples_per_record)
    per_signal: list[np.ndarray] = [
        np.empty(samples_per_record[i] * n_records, dtype=np.float64)
        for i in range(ns)
    ]
    write_pos = [0] * ns

    blob = np.frombuffer(raw_blob, dtype=np.uint8)
    rec_len_bytes = record_size
    for r in range(n_records):
        base = r * rec_len_bytes
        for i in range(ns):
            spr = samples_per_record[i]
            if spr == 0:
                continue
            start = base + offsets[i] * bytes_per_sample
            count = spr * bytes_per_sample
            chunk = blob[start:start + count]
            ints = _decode_ints(chunk, is_bdf, spr)
            per_signal[i][write_pos[i]:write_pos[i] + spr] = ints
            write_pos[i] += spr

    # Convert each signal from digital to physical units.
    eeg_idx: list[int] = []
    for i in range(ns):
        if "annotation" in labels[i].lower():
            continue  # EDF+ TAL channel, not numeric
        d_span = (dig_max[i] - dig_min[i]) or 1.0
        gain = (phys_max[i] - phys_min[i]) / d_span
        per_signal[i] = (per_signal[i] - dig_min[i]) * gain + phys_min[i]
        eeg_idx.append(i)

    if not eeg_idx:
        raise ValueError("EDF/BDF file contains no numeric signal channels")

    # Resample every kept signal to the maximum sample rate present, so the
    # output is a single rectangular matrix.
    rates = [samples_per_record[i] / record_duration for i in eeg_idx]
    target_rate = max(rates)
    target_n = int(round(target_rate * record_duration * n_records))
    target_n = max(target_n, 1)

    target_times = np.arange(target_n) / target_rate
    cols: list[np.ndarray] = []
    for i, rate in zip(eeg_idx, rates):
        sig = per_signal[i]
        if len(sig) == target_n:
            cols.append(sig)
        elif len(sig) <= 1:
            cols.append(np.full(target_n, sig[0] if len(sig) else 0.0))
        else:
            # Interpolate on real sample times (sample k of a rate-`rate`
            # signal is at t = k / rate) rather than endpoint-anchored
            # [0, 1] linspaces, which would time-shift slower channels.
            src_times = np.arange(len(sig)) / rate
            cols.append(np.interp(target_times, src_times, sig))

    data = np.stack(cols, axis=1).astype(np.float64)
    return EdfData(
        data=data,
        sample_rate_hz=float(target_rate),
        channel_names=[labels[i] or f"ch{i}" for i in eeg_idx],
        physical_dim=[phys_dim[i] for i in eeg_idx],
        is_bdf=is_bdf,
        start_datetime=f"{startdate} {starttime}".strip(),
    )


def _decode_ints(chunk: np.ndarray, is_bdf: bool, count: int) -> np.ndarray:
    """Decode a little-endian signed int16 (EDF) or int24 (BDF) byte block."""
    if not is_bdf:
        return chunk.view("<i2").astype(np.float64)
    # 24-bit little-endian signed -> int32.
    b = chunk.reshape(count, 3).astype(np.int32)
    val = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    # Sign-extend from 24 bits.
    val = np.where(val & 0x800000, val - 0x1000000, val)
    return val.astype(np.float64)
