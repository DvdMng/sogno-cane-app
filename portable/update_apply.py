"""Apply a staged SOGNO_CANE update, then exit. Runs BEFORE the app starts.

This script is intentionally standalone — it does NOT import the ``sogno_cane``
package, because it may be the very thing being replaced. It is placed at the
portable bundle root (next to START.bat) and run by START.bat before launching
the GUI:

    runtime\\python.exe "%~dp0update_apply.py"
    start "" runtime\\pythonw.exe -m sogno_cane

It reads ``%USERPROFILE%\\.sogno_cane\\updates\\pending.json`` (written by the
in-app updater), copies the staged package over
``runtime\\Lib\\site-packages\\sogno_cane``, and keeps a backup so a failure
rolls back cleanly. No-op if there is nothing pending.
"""
import json
import os
import shutil
import sys


def home_dir() -> str:
    base = os.environ.get("SOGNO_CANE_HOME") or os.path.join(
        os.path.expanduser("~"), ".sogno_cane"
    )
    return base


def site_packages_target() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        here, "runtime", "Lib", "site-packages", "sogno_cane"
    )


def main() -> int:
    pending = os.path.join(home_dir(), "updates", "pending.json")
    if not os.path.exists(pending):
        return 0
    try:
        with open(pending, encoding="utf-8") as f:
            info = json.load(f)
    except Exception:
        try:
            os.remove(pending)
        except Exception:
            pass
        return 0

    src = info.get("package_dir", "")
    target = site_packages_target()
    if not src or not os.path.isdir(src):
        try:
            os.remove(pending)
        except Exception:
            pass
        return 0

    backup = target + "_backup"
    try:
        if os.path.isdir(backup):
            shutil.rmtree(backup, ignore_errors=True)
        if os.path.isdir(target):
            shutil.move(target, backup)
        shutil.copytree(src, target)
        # Success: drop backup + pending marker + staged files.
        shutil.rmtree(backup, ignore_errors=True)
        os.remove(pending)
        shutil.rmtree(
            os.path.join(home_dir(), "updates", "staged"), ignore_errors=True
        )
        print(f"SOGNO_CANE updated to {info.get('version', '?')}")
    except Exception as exc:  # pragma: no cover - filesystem dependent
        # Roll back if the new copy did not complete.
        try:
            if os.path.isdir(backup) and not os.path.isdir(target):
                shutil.move(backup, target)
        except Exception:
            pass
        print(f"update apply failed, kept current version: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
