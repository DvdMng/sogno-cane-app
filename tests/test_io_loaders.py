import numpy as np

from sogno_cane.io.loaders import load_eeg


def test_csv_with_header(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("Fz,Cz,Oz\n1,2,3\n4,5,6\n7,8,9\n")
    lo = load_eeg(str(p))
    assert lo.n_channels == 3
    assert lo.n_samples == 3
    assert lo.channel_names == ["Fz", "Cz", "Oz"]
    assert np.allclose(lo.data[0], [1, 2, 3])


def test_csv_no_header(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("1,2\n3,4\n5,6\n")
    lo = load_eeg(str(p))
    assert lo.n_channels == 2
    assert lo.n_samples == 3


def test_csv_time_column_detected(tmp_path):
    p = tmp_path / "c.csv"
    rows = ["time,ch0,ch1"]
    for i in range(10):
        rows.append(f"{i/250.0:.6f},{i},{i*2}")
    p.write_text("\n".join(rows))
    lo = load_eeg(str(p))
    assert lo.n_channels == 2          # time column stripped
    assert abs(lo.sample_rate_hz - 250.0) < 1.0
    assert lo.channel_names == ["ch0", "ch1"]


def test_tsv(tmp_path):
    p = tmp_path / "d.tsv"
    p.write_text("a\tb\n1\t2\n3\t4\n")
    lo = load_eeg(str(p))
    assert lo.n_channels == 2
    assert lo.channel_names == ["a", "b"]


def test_npy(tmp_path):
    arr = np.random.default_rng(0).normal(size=(100, 4))
    p = tmp_path / "e.npy"
    np.save(str(p), arr)
    lo = load_eeg(str(p))
    assert lo.data.shape == (100, 4)


def test_npz_native(tmp_path):
    from sogno_cane.io.recording import Recording, RecordingMeta
    eeg = np.random.default_rng(0).normal(size=(50, 8))
    meta = RecordingMeta(
        rec_id="rec-1", name="n", profile="DOG",
        channel_names=[f"c{i}" for i in range(8)],
        sample_rate_hz=200.0, n_samples=50, created="",
    )
    Recording(eeg=eeg, meta=meta).save(str(tmp_path))
    lo = load_eeg(str(tmp_path / "rec-1.npz"))
    assert lo.n_channels == 8
    assert lo.sample_rate_hz == 200.0


def test_semicolon_delimiter(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("x;y;z\n1;2;3\n4;5;6\n")
    lo = load_eeg(str(p))
    assert lo.n_channels == 3
