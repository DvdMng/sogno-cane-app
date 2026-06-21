"""Build a SOGNO_CANE auto-update release.

Produces, in the output directory:
  * ``sogno_cane_package.zip``  — the package (top-level ``sogno_cane/`` folder)
  * ``version.json``            — the update manifest the app checks online

Usage (from the project root)::

    python tools/build_release.py \
        --url https://github.com/<user>/<repo>/releases/latest/download/sogno_cane_package.zip \
        --notes "What changed" \
        --out dist

Then upload BOTH files as assets of a GitHub Release (or any static host), and
point the installed apps' ``update_url`` at the published ``version.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zipfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "sogno_cane")
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def read_version() -> str:
    with open(os.path.join(PKG, "__init__.py"), encoding="utf-8") as f:
        text = f.read()
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if not m:
        raise SystemExit("could not find __version__ in sogno_cane/__init__.py")
    return m.group(1)


def build_zip(dest_zip: str) -> None:
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirs, files in os.walk(PKG):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fn in files:
                if fn.endswith(".pyc"):
                    continue
                full = os.path.join(dirpath, fn)
                arc = os.path.relpath(full, ROOT)  # -> sogno_cane/...
                z.write(full, arc)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a SOGNO_CANE release.")
    ap.add_argument(
        "--url", default="",
        help="Public download URL the package zip will live at "
             "(used by the in-app auto-updater).",
    )
    ap.add_argument(
        "--bundle-url", default="",
        help="Public download URL of the FULL portable bundle zip "
             "(used by SOGNO_CANE_Setup.bat for first-time install).",
    )
    ap.add_argument(
        "--bundle", default="",
        help="Path to the full portable bundle zip, to record its SHA-256 "
             "in the manifest.",
    )
    ap.add_argument("--notes", default="", help="Release notes.")
    ap.add_argument("--out", default="dist", help="Output directory.")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    version = read_version()

    zip_path = os.path.join(out, "sogno_cane_package.zip")
    build_zip(zip_path)
    digest = sha256(zip_path)

    manifest = {
        "version": version,
        "url": args.url,
        "sha256": digest,
        "bundle_url": args.bundle_url,
        "bundle_sha256": sha256(args.bundle) if args.bundle else "",
        "notes": args.notes,
    }
    with open(os.path.join(out, "version.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"version    : {version}")
    print(f"package    : {zip_path} ({os.path.getsize(zip_path)//1024} KB)")
    print(f"sha256     : {digest}")
    print(f"manifest   : {os.path.join(out, 'version.json')}")
    if not args.url:
        print("\nNOTE: --url was empty. Edit version.json's 'url' to the "
              "public download URL before publishing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
