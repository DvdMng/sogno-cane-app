"""Lightweight JSON settings persistence.

The original build forgot every choice on exit — profile, MIDI port, window
size — forcing the user to reconfigure on each launch. This stores a small
JSON document under the user's SOGNO_CANE home directory and restores it on
startup. It never raises: a corrupt or missing file simply yields defaults.
"""
from __future__ import annotations

import json
import os
from typing import Any


def home_dir() -> str:
    base = os.environ.get("SOGNO_CANE_HOME")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".sogno_cane")
    os.makedirs(base, exist_ok=True)
    return base


def settings_path() -> str:
    return os.path.join(home_dir(), "settings.json")


DEFAULTS: dict[str, Any] = {
    "human": {"profile": "HUMAN", "port": "", "loop": True},
    "dog": {"profile": "DOG", "port": "", "loop": True},
    "window": {"w": 1600, "h": 1020},
    "last_file_dir": "",
    # Online auto-update. ``update_url`` points at the JSON manifest; leave
    # empty to disable. ``auto_update`` lets the app download a found update
    # automatically (it is always applied at the next launch).
    "update_url": "",
    "auto_update": True,
}


class Settings:
    """A dict-like settings store with safe load/save."""

    def __init__(self, data: dict | None = None) -> None:
        self._data: dict[str, Any] = json.loads(json.dumps(DEFAULTS))
        if data:
            self._merge(self._data, data)

    @staticmethod
    def _merge(base: dict, override: dict) -> None:
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                Settings._merge(base[k], v)
            else:
                base[k] = v

    @classmethod
    def load(cls) -> "Settings":
        try:
            # utf-8-sig tolerates a BOM (e.g. a settings file written by the
            # Windows installer via PowerShell), which plain utf-8 would not.
            with open(settings_path(), "r", encoding="utf-8-sig") as f:
                return cls(json.load(f))
        except Exception:
            return cls()

    def save(self) -> None:
        try:
            with open(settings_path(), "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception:
            pass

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self._data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def set(self, *keys_and_value: Any) -> None:
        *keys, value = keys_and_value
        node = self._data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    @property
    def data(self) -> dict:
        return self._data
