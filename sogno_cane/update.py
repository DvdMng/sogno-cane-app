"""Online auto-update for SOGNO_CANE.

When the machine is connected, the running app checks a small JSON *manifest*
hosted by you (e.g. on GitHub Releases), and if a newer version is published it
downloads the new package, verifies its SHA-256, and *stages* it. The actual
file swap happens at the next launch via the standalone ``update_apply.py`` so
we never overwrite a package that is currently imported.

Only the Python standard library is used, so the portable bundle needs no
extra dependency. Every network call is best-effort and times out quickly:
offline simply means "no update".

The manifest URL is resolved, in order, from:
  1. the ``SOGNO_CANE_UPDATE_URL`` environment variable,
  2. ``update_url`` in the user's settings.json,
  3. :data:`DEFAULT_UPDATE_URL` baked in at build time.

Manifest format (JSON)::

    {
      "version": "0.3.0",
      "url": "https://.../sogno_cane_package.zip",
      "sha256": "<hex digest of the zip>",
      "notes": "What changed in this release"
    }

The referenced zip must contain a top-level ``sogno_cane/`` package folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Callable, Optional

from sogno_cane import __version__
from sogno_cane.settings import Settings, home_dir

# Official update channel (GitHub Releases "latest" — always points to the
# most recent published version.json). Overridable via SOGNO_CANE_UPDATE_URL
# or the settings.json 'update_url'.
DEFAULT_UPDATE_URL = (
    "https://github.com/DvdMng/sogno-cane-app"
    "/releases/latest/download/version.json"
)

_UA = {"User-Agent": f"SOGNO_CANE-updater/{__version__}"}


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
def update_url() -> str:
    env = os.environ.get("SOGNO_CANE_UPDATE_URL")
    if env:
        return env.strip()
    try:
        cfg = Settings.load().get("update_url", default="")
    except Exception:
        cfg = ""
    return (cfg or DEFAULT_UPDATE_URL).strip()


def auto_update_enabled() -> bool:
    try:
        return bool(Settings.load().get("auto_update", default=True))
    except Exception:
        return True


def updates_dir() -> str:
    d = os.path.join(home_dir(), "updates")
    os.makedirs(d, exist_ok=True)
    return d


def _pending_path() -> str:
    return os.path.join(updates_dir(), "pending.json")


# --------------------------------------------------------------------------- #
# Version comparison                                                          #
# --------------------------------------------------------------------------- #
def parse_version(v: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for p in str(v).strip().lstrip("vV").split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def is_newer(remote: str, local: str) -> bool:
    return parse_version(remote) > parse_version(local)


# --------------------------------------------------------------------------- #
# Manifest check + download                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class UpdateInfo:
    version: str
    url: str
    sha256: str = ""
    notes: str = ""


def check_for_update(
    timeout: float = 6.0, current: str = __version__
) -> Optional[UpdateInfo]:
    """Return an :class:`UpdateInfo` if a newer version is published, else None.

    Never raises: any network/parse error (or being offline) yields ``None``.
    """
    url = update_url()
    if not url:
        return None
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    version = str(data.get("version", "")).strip()
    if not version or not is_newer(version, current):
        return None
    return UpdateInfo(
        version=version,
        url=str(data.get("url", "")).strip(),
        sha256=str(data.get("sha256", "")).strip(),
        notes=str(data.get("notes", "")).strip(),
    )


def download_and_stage(
    info: UpdateInfo,
    timeout: float = 120.0,
    progress: Optional[Callable[[float], None]] = None,
) -> str:
    """Download, verify, and stage the update. Returns the staged package dir.

    Writes ``updates/pending.json`` so ``update_apply.py`` applies it at the
    next launch. Raises on download/verification failure.
    """
    if not info.url:
        raise ValueError("update manifest has no download url")

    fd, tmp_zip = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    try:
        req = urllib.request.Request(info.url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp, \
                open(tmp_zip, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0) or 0)
            done = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(min(1.0, done / total))

        if info.sha256:
            digest = _sha256(tmp_zip)
            if digest.lower() != info.sha256.lower():
                raise ValueError(
                    "checksum mismatch — download corrupted or tampered with"
                )

        staged_root = os.path.join(updates_dir(), "staged")
        shutil.rmtree(staged_root, ignore_errors=True)
        os.makedirs(staged_root)
        with zipfile.ZipFile(tmp_zip) as z:
            for name in z.namelist():
                norm = name.replace("\\", "/")
                if norm.startswith("/") or ".." in norm.split("/"):
                    raise ValueError("unsafe path in update archive")
            z.extractall(staged_root)

        pkg_dir = _find_package_dir(staged_root)
        if pkg_dir is None:
            raise ValueError(
                "update archive does not contain a 'sogno_cane' package"
            )
        with open(_pending_path(), "w", encoding="utf-8") as f:
            json.dump(
                {"version": info.version, "package_dir": pkg_dir,
                 "notes": info.notes},
                f, indent=2,
            )
        return pkg_dir
    finally:
        try:
            os.unlink(tmp_zip)
        except Exception:
            pass


def has_pending_update() -> Optional[dict]:
    try:
        with open(_pending_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def clear_pending() -> None:
    try:
        os.unlink(_pending_path())
    except Exception:
        pass
    shutil.rmtree(os.path.join(updates_dir(), "staged"), ignore_errors=True)


def apply_pending_update(target_dir: str) -> bool:
    """Replace ``target_dir`` (a ``sogno_cane`` package folder) with the staged
    update, keeping a backup for rollback. Returns True if applied.

    NOTE: in the portable bundle the swap is done by the standalone
    ``update_apply.py`` at launch (so it never imports the package it is
    replacing). This function is the importable equivalent, used by tests and
    as an alternative API.
    """
    pend = has_pending_update()
    if not pend:
        return False
    src = pend.get("package_dir", "")
    if not src or not os.path.isdir(src):
        clear_pending()
        return False
    backup = target_dir + "_backup"
    if os.path.isdir(backup):
        shutil.rmtree(backup, ignore_errors=True)
    if os.path.isdir(target_dir):
        shutil.move(target_dir, backup)
    try:
        shutil.copytree(src, target_dir)
    except Exception:
        if os.path.isdir(backup) and not os.path.isdir(target_dir):
            shutil.move(backup, target_dir)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    clear_pending()
    return True


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _find_package_dir(root: str) -> Optional[str]:
    """Find a folder named ``sogno_cane`` that contains ``__init__.py``."""
    for dirpath, _dirs, files in os.walk(root):
        if os.path.basename(dirpath) == "sogno_cane" and "__init__.py" in files:
            return dirpath
    return None
