"""Tests for the online auto-update client (uses local file:// URLs)."""
import hashlib
import json
import os
import pathlib
import zipfile

import pytest

from sogno_cane import update as up


def _file_url(path: str) -> str:
    return pathlib.Path(path).resolve().as_uri()


def _make_package_zip(dest_zip: str, version: str) -> None:
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("sogno_cane/__init__.py", f'__version__ = "{version}"\n')
        z.writestr("sogno_cane/core/__init__.py", "")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def test_parse_version_and_compare():
    assert up.parse_version("0.2.0") == (0, 2, 0)
    assert up.parse_version("v1.10.3") == (1, 10, 3)
    assert up.is_newer("0.3.0", "0.2.9")
    assert up.is_newer("1.0.0", "0.9.9")
    assert not up.is_newer("0.2.0", "0.2.0")
    assert not up.is_newer("0.1.0", "0.2.0")


def test_no_url_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    monkeypatch.delenv("SOGNO_CANE_UPDATE_URL", raising=False)
    monkeypatch.setattr(up, "DEFAULT_UPDATE_URL", "")  # clear baked default
    assert up.update_url() == ""
    assert up.check_for_update(current="0.2.0") is None


def test_baked_default_url_is_used(monkeypatch, tmp_path):
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    monkeypatch.delenv("SOGNO_CANE_UPDATE_URL", raising=False)
    monkeypatch.setattr(up, "DEFAULT_UPDATE_URL", "https://example/v.json")
    assert up.update_url() == "https://example/v.json"


def test_check_finds_newer(monkeypatch, tmp_path):
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    manifest = tmp_path / "version.json"
    manifest.write_text(json.dumps(
        {"version": "0.9.0", "url": "http://x/y.zip", "sha256": "", "notes": "n"}
    ))
    monkeypatch.setenv("SOGNO_CANE_UPDATE_URL", _file_url(str(manifest)))
    info = up.check_for_update(current="0.2.0")
    assert info is not None and info.version == "0.9.0"
    # Same version -> no update.
    assert up.check_for_update(current="0.9.0") is None


def test_download_stage_and_apply(monkeypatch, tmp_path):
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path / "home"))
    zip_path = tmp_path / "pkg.zip"
    _make_package_zip(str(zip_path), "0.9.9")
    info = up.UpdateInfo(
        version="0.9.9", url=_file_url(str(zip_path)),
        sha256=_sha256(str(zip_path)),
    )
    staged = up.download_and_stage(info)
    assert os.path.isfile(os.path.join(staged, "__init__.py"))
    assert up.has_pending_update()["version"] == "0.9.9"

    # Apply over a fake existing install.
    target = tmp_path / "site" / "sogno_cane"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text('__version__ = "0.2.0"\n')
    assert up.apply_pending_update(str(target)) is True
    assert '0.9.9' in (target / "__init__.py").read_text()
    assert up.has_pending_update() is None
    assert not (tmp_path / "site" / "sogno_cane_backup").exists()


def test_checksum_mismatch_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path / "home"))
    zip_path = tmp_path / "pkg.zip"
    _make_package_zip(str(zip_path), "0.9.9")
    info = up.UpdateInfo(
        version="0.9.9", url=_file_url(str(zip_path)), sha256="deadbeef",
    )
    with pytest.raises(ValueError):
        up.download_and_stage(info)


def test_apply_without_pending_is_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path / "home"))
    assert up.apply_pending_update(str(tmp_path / "nope")) is False
