import numpy as np
import pytest

from sogno_cane.io.recording import Recorder, Recording, RecordingMeta


def _make_recording(n=1000, nc=8, fs=250.0):
    eeg = np.random.default_rng(0).normal(0, 10, size=(n, nc))
    meta = RecordingMeta(
        rec_id="rec-x", name="t", profile="HUMAN",
        channel_names=[f"c{i}" for i in range(nc)],
        sample_rate_hz=fs, n_samples=n, created="2020-01-01 00:00:00",
    )
    return Recording(eeg=eeg, meta=meta)


def test_recorder_accumulates():
    rec = Recorder(channel_names=["a"] * 8, sample_rate_hz=250)
    assert rec.is_empty()
    rec.add_eeg(np.zeros((25, 8)))
    rec.add_eeg(np.zeros((25, 8)))
    assert rec.n_samples == 50
    out = rec.to_recording("id1", "name", "2020-01-01")
    assert out.eeg.shape == (50, 8)
    assert out.meta.n_samples == 50


def test_trim_range():
    r = _make_recording(n=1000, fs=250.0)
    t = r.trimmed(1.0, 3.0)
    assert t.eeg.shape[0] == 500  # 2 s @ 250
    assert np.array_equal(t.eeg, r.eeg[250:750])


def test_trim_empty_raises():
    r = _make_recording()
    with pytest.raises(ValueError):
        r.trimmed(2.0, 1.0)


def test_save_load_roundtrip(tmp_path):
    r = _make_recording()
    npz, js = r.save(str(tmp_path))
    loaded = Recording.load(npz)
    assert loaded.eeg.shape == r.eeg.shape
    assert np.allclose(loaded.eeg, r.eeg, atol=1e-2)  # float32 storage
    assert loaded.meta.name == r.meta.name
    assert loaded.meta.sample_rate_hz == r.meta.sample_rate_hz


def test_export_csv_has_header_and_time(tmp_path):
    r = _make_recording(n=10, nc=3)
    p = tmp_path / "out.csv"
    r.export_csv(str(p), include_time=True)
    lines = p.read_text().splitlines()
    assert lines[0].split(",")[0] == "time_s"
    assert len(lines) == 11  # header + 10 rows
    assert len(lines[0].split(",")) == 4  # time + 3 channels


def test_export_csv_no_time(tmp_path):
    r = _make_recording(n=5, nc=2)
    p = tmp_path / "out.csv"
    r.export_csv(str(p), include_time=False)
    lines = p.read_text().splitlines()
    assert len(lines[0].split(",")) == 2
