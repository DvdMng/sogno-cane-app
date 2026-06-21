"""Round-trip test of the dependency-free EDF reader against a hand-written
EDF file. (The reader is also cross-validated against pyedflib on real EDF/BDF
files during manual testing; this keeps CI self-contained and offline.)"""
import struct

import numpy as np
import pytest

from sogno_cane.io.edf import read_edf
from sogno_cane.io.loaders import load_eeg


def _f(value, width):
    s = str(value)
    assert len(s) <= width, f"{s!r} too wide for {width}"
    return s.ljust(width).encode("ascii")


def _write_edf(path, signals, fs, record_duration=1.0):
    """signals: list of (label, int16 ndarray). All same length."""
    ns = len(signals)
    spr = int(fs * record_duration)
    n_total = len(signals[0][1])
    assert n_total % spr == 0
    n_records = n_total // spr

    header = b""
    header += _f("0", 8)                 # version
    header += _f("X X X test", 80)       # patient
    header += _f("Startdate", 80)        # recording
    header += _f("01.01.20", 8)
    header += _f("00.00.00", 8)
    header += _f(256 * (ns + 1), 8)      # header bytes
    header += _f("", 44)                 # reserved
    header += _f(n_records, 8)
    header += _f(int(record_duration), 8)
    header += _f(ns, 4)

    labels = b"".join(_f(lbl, 16) for lbl, _ in signals)
    transducer = b"".join(_f("", 80) for _ in signals)
    phys_dim = b"".join(_f("uV", 8) for _ in signals)
    phys_min = b"".join(_f(-100, 8) for _ in signals)
    phys_max = b"".join(_f(100, 8) for _ in signals)
    dig_min = b"".join(_f(-100, 8) for _ in signals)
    dig_max = b"".join(_f(100, 8) for _ in signals)
    prefilter = b"".join(_f("", 80) for _ in signals)
    spr_field = b"".join(_f(spr, 8) for _ in signals)
    sig_reserved = b"".join(_f("", 32) for _ in signals)

    header += (
        labels + transducer + phys_dim + phys_min + phys_max
        + dig_min + dig_max + prefilter + spr_field + sig_reserved
    )

    body = bytearray()
    for r in range(n_records):
        for _, data in signals:
            block = data[r * spr:(r + 1) * spr].astype("<i2")
            body += block.tobytes()

    with open(path, "wb") as f:
        f.write(header)
        f.write(bytes(body))


def test_edf_roundtrip(tmp_path):
    fs = 100
    n = 300
    ch0 = (np.arange(n) % 100 - 50).astype(np.int16)
    ch1 = (np.arange(n) % 50).astype(np.int16)
    path = tmp_path / "t.edf"
    _write_edf(str(path), [("Fz", ch0), ("Cz", ch1)], fs)

    edf = read_edf(str(path))
    assert edf.sample_rate_hz == fs
    assert edf.channel_names == ["Fz", "Cz"]
    assert edf.data.shape == (300, 2)
    # phys range == dig range -> physical equals digital.
    assert np.allclose(edf.data[:, 0], ch0)
    assert np.allclose(edf.data[:, 1], ch1)


def test_edf_via_loader(tmp_path):
    fs = 100
    ch = (np.arange(200) % 40 - 20).astype(np.int16)
    path = tmp_path / "u.edf"
    _write_edf(str(path), [("O1", ch)], fs, record_duration=1.0)
    lo = load_eeg(str(path))
    assert lo.n_channels == 1
    assert lo.sample_rate_hz == fs
    assert lo.duration_seconds == 2.0


def test_edf_rejects_non_edf(tmp_path):
    p = tmp_path / "bad.edf"
    p.write_bytes(b"this is not an edf file at all, just text padding...." * 8)
    with pytest.raises(ValueError):
        read_edf(str(p))


def test_edf_skips_annotation_channel(tmp_path):
    fs = 100
    ch = (np.arange(100) % 10).astype(np.int16)
    ann = np.zeros(100, dtype=np.int16)
    path = tmp_path / "v.edf"
    _write_edf(str(path), [("EEG Fz", ch), ("EDF Annotations", ann)], fs)
    edf = read_edf(str(path))
    assert edf.channel_names == ["EEG Fz"]
    assert edf.data.shape[1] == 1
