import numpy as np

from sogno_cane.io.archive import Archive
from sogno_cane.io.recording import Recording, RecordingMeta


def _rec(name="take", n=500, fs=250.0):
    eeg = np.random.default_rng(1).normal(0, 5, size=(n, 8))
    meta = RecordingMeta(
        rec_id="", name=name, profile="HUMAN",
        channel_names=[f"c{i}" for i in range(8)],
        sample_rate_hz=fs, n_samples=n, created="",
    )
    return Recording(eeg=eeg, meta=meta)


def test_save_list_load(tmp_path):
    arc = Archive(root=str(tmp_path))
    rid = arc.save(_rec("first"))
    assert arc.exists(rid)
    metas = arc.list()
    assert len(metas) == 1
    assert metas[0].name == "first"
    loaded = arc.load(rid)
    assert loaded.eeg.shape == (500, 8)


def test_unique_ids(tmp_path):
    arc = Archive(root=str(tmp_path))
    ids = {arc.save(_rec(f"r{i}")) for i in range(5)}
    assert len(ids) == 5
    assert len(arc.list()) == 5


def test_rename(tmp_path):
    arc = Archive(root=str(tmp_path))
    rid = arc.save(_rec("old"))
    arc.rename(rid, "new")
    assert arc.load(rid).meta.name == "new"


def test_delete(tmp_path):
    arc = Archive(root=str(tmp_path))
    rid = arc.save(_rec("x"))
    arc.delete(rid)
    assert not arc.exists(rid)
    assert arc.list() == []


def test_trim_creates_new(tmp_path):
    arc = Archive(root=str(tmp_path))
    rid = arc.save(_rec("orig", n=1000))
    new_id = arc.trim(rid, 0.0, 2.0)
    assert new_id != rid
    assert arc.exists(rid)  # original preserved
    assert abs(arc.load(new_id).meta.duration_seconds - 2.0) < 1e-6


def test_trim_in_place(tmp_path):
    arc = Archive(root=str(tmp_path))
    rid = arc.save(_rec("orig", n=1000))
    same = arc.trim(rid, 0.0, 1.0, in_place=True)
    assert same == rid
    assert abs(arc.load(rid).meta.duration_seconds - 1.0) < 1e-6


def test_export_csv(tmp_path):
    arc = Archive(root=str(tmp_path))
    rid = arc.save(_rec("x", n=20))
    dest = tmp_path / "e.csv"
    arc.export_csv(rid, str(dest))
    assert dest.exists() and dest.stat().st_size > 0
