"""Save / load complete SOGNO_CANE configurations.

A *configuration* captures everything the user tunes: for each device (human
and dog) the profile, MIDI port, loop flag, and the full mapping bundle — every
strategy and every per-channel voice/rule with all its parameters.

Configs are JSON. The serializer walks the strategy dataclasses and keeps only
their ``init=True`` fields, so the volatile runtime state (RNG generators, note
gates, smoothing buffers — all ``init=False``) is never written. On load the
values are applied *in place* onto the existing bundle objects, so the running
engine and the mapping panels pick them up without rebuilding the object graph.
"""
from __future__ import annotations

import json
import os
from dataclasses import fields, is_dataclass

from sogno_cane import __version__
from sogno_cane.midi.scales import MidiNoteRange
from sogno_cane.settings import home_dir

CONFIG_VERSION = 1
_BUNDLE_KEYS = (
    "per_channel_band", "per_channel_cc", "threshold",
    "coherence", "markov", "clips",
)


def configs_dir() -> str:
    d = os.path.join(home_dir(), "configs")
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Serialize                                                                   #
# --------------------------------------------------------------------------- #
def _serialize(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _serialize(getattr(obj, f.name))
            for f in fields(obj) if f.init
        }
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def serialize_bundle(bundle) -> dict:
    return {k: _serialize(getattr(bundle, k)) for k in _BUNDLE_KEYS}


# --------------------------------------------------------------------------- #
# Apply (in place)                                                            #
# --------------------------------------------------------------------------- #
def _apply_scalars(obj, data: dict) -> None:
    """Set the scalar/tuple fields of a dataclass from ``data`` (in place)."""
    if not is_dataclass(obj) or not isinstance(data, dict):
        return
    init_names = {f.name for f in fields(obj)}
    for key, val in data.items():
        if key not in init_names:
            continue
        cur = getattr(obj, key, None)
        if isinstance(val, dict) and isinstance(cur, MidiNoteRange):
            try:
                setattr(obj, key, MidiNoteRange(**val))
            except Exception:
                pass
        elif isinstance(val, list):
            if val and isinstance(val[0], dict):
                continue  # nested dataclasses (voices/rules) handled elsewhere
            try:
                setattr(obj, key, tuple(val) if isinstance(cur, tuple) else val)
            except Exception:
                pass
        elif isinstance(val, dict):
            continue
        else:
            try:
                setattr(obj, key, val)
            except Exception:
                pass


def apply_bundle_config(bundle, data: dict) -> None:
    """Apply a serialized bundle onto ``bundle`` in place (best-effort)."""
    if not isinstance(data, dict):
        return
    # Per-channel strategies: scalars + each voice/rule by index.
    for key, list_attr in (
        ("per_channel_band", "voices"),
        ("per_channel_cc", "voices"),
        ("clips", "rules"),
    ):
        strat = getattr(bundle, key, None)
        sd = data.get(key, {})
        if strat is None or not isinstance(sd, dict):
            continue
        _apply_scalars(strat, sd)
        items = getattr(strat, list_attr, []) or []
        for obj, od in zip(items, sd.get(list_attr, []) or []):
            _apply_scalars(obj, od)
    # Flat strategies.
    for key in ("threshold", "coherence", "markov"):
        strat = getattr(bundle, key, None)
        if strat is not None:
            _apply_scalars(strat, data.get(key, {}))


# --------------------------------------------------------------------------- #
# Files                                                                       #
# --------------------------------------------------------------------------- #
def build_config(devices: dict) -> dict:
    """``devices`` maps name -> {profile, port, loop, bundle}."""
    out = {
        "config_version": CONFIG_VERSION,
        "app_version": __version__,
        "devices": {},
    }
    for name, dev in devices.items():
        out["devices"][name] = {
            "profile": dev["profile"],
            "port": dev["port"],
            "loop": dev["loop"],
            "bundle": serialize_bundle(dev["bundle"]),
        }
    return out


def save_config(path: str, config: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def list_saved_configs() -> list[str]:
    """Return saved config file paths in the configs dir, newest first."""
    d = configs_dir()
    out = [
        os.path.join(d, fn) for fn in os.listdir(d)
        if fn.lower().endswith(".json")
    ]
    out.sort(key=os.path.getmtime, reverse=True)
    return out
