import json

from sogno_cane import config as appconfig
from sogno_cane.midi.presets import rich_vocabulary_preset


def test_serialize_excludes_runtime_state():
    b = rich_vocabulary_preset()
    d = appconfig.serialize_bundle(b)
    # Runtime (_-prefixed, init=False) fields must not be serialized.
    text = json.dumps(d)
    assert "_gates" not in text
    assert "_rng" not in text
    assert "_last_notes" not in text
    # Real config fields ARE present.
    assert d["markov"]["scale"]
    assert isinstance(d["per_channel_band"]["voices"], list)
    assert len(d["per_channel_band"]["voices"]) == 8


def test_bundle_config_roundtrip_in_place():
    src = rich_vocabulary_preset()
    # Mutate a variety of fields.
    src.per_channel_band.voices[0].band = "gamma"
    src.per_channel_band.voices[0].scale = "dorian"
    src.per_channel_band.voices[0].min_interval_seconds = 4.0
    src.per_channel_cc.voices[2].cc_number = 99
    src.markov.scale = "phrygian"
    src.markov.max_notes_per_window = 7
    src.threshold.threshold_uv2 = 123.0
    src.coherence.channels_a = (0, 1)
    src.clips.rules[1].midi_note = 55

    data = appconfig.serialize_bundle(src)

    # Apply onto a fresh default bundle.
    dst = rich_vocabulary_preset()
    appconfig.apply_bundle_config(dst, data)

    assert dst.per_channel_band.voices[0].band == "gamma"
    assert dst.per_channel_band.voices[0].scale == "dorian"
    assert dst.per_channel_band.voices[0].min_interval_seconds == 4.0
    assert dst.per_channel_cc.voices[2].cc_number == 99
    assert dst.markov.scale == "phrygian"
    assert dst.markov.max_notes_per_window == 7
    assert dst.threshold.threshold_uv2 == 123.0
    assert tuple(dst.coherence.channels_a) == (0, 1)
    assert dst.clips.rules[1].midi_note == 55


def test_full_config_save_load(tmp_path, monkeypatch):
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    b = rich_vocabulary_preset()
    b.markov.root = "F"
    devices = {
        "human": {"profile": "HUMAN", "port": "PortA", "loop": True,
                  "bundle": b},
        "dog": {"profile": "DOG", "port": "", "loop": False,
                "bundle": rich_vocabulary_preset()},
    }
    cfg = appconfig.build_config(devices)
    path = tmp_path / "c.json"
    appconfig.save_config(str(path), cfg)
    loaded = appconfig.load_config(str(path))
    assert loaded["devices"]["human"]["profile"] == "HUMAN"
    assert loaded["devices"]["human"]["port"] == "PortA"
    assert loaded["devices"]["dog"]["loop"] is False
    assert loaded["devices"]["human"]["bundle"]["markov"]["root"] == "F"


def test_apply_is_robust_to_garbage():
    b = rich_vocabulary_preset()
    appconfig.apply_bundle_config(b, {"markov": {"nonsense": 1},
                                      "bogus": [1, 2]})  # must not raise
    appconfig.apply_bundle_config(b, "not a dict")       # must not raise


def test_configs_dir_created(tmp_path, monkeypatch):
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    d = appconfig.configs_dir()
    import os
    assert os.path.isdir(d)


# A frozen configuration as written by v0.3.0. Channels are stored 0-based and
# must still load (and route to the same wire channels) in later versions.
_LEGACY_0_3_0_CONFIG = {
    "config_version": 1,
    "app_version": "0.3.0",
    "devices": {
        "human": {
            "profile": "HUMAN", "port": "loopMIDI Port", "loop": True,
            "bundle": {
                "per_channel_band": {
                    "enabled": True,
                    "voices": [
                        {"eeg_channel": 0, "midi_channel": 7, "band": "beta",
                         "scale": "dorian", "root": "C",
                         "min_interval_seconds": 3.7},
                    ],
                },
                "per_channel_cc": {
                    "voices": [
                        {"eeg_channel": 0, "midi_channel": 5, "cc_number": 88,
                         "band": "alpha"},
                    ],
                },
                "markov": {"channel": 6, "scale": "lydian"},
                "threshold": {"channel": 9, "threshold_uv2": 70.0},
                "coherence": {"channel": 10},
                "clips": {
                    "rules": [
                        {"eeg_channel": 0, "band": "theta", "midi_note": 36,
                         "midi_channel": 15},
                    ],
                },
            },
        },
    },
}


def test_loads_legacy_0_3_0_config():
    """A config saved by v0.3.0 must apply unchanged (channels stay 0-based)."""
    b = rich_vocabulary_preset()
    appconfig.apply_bundle_config(
        b, _LEGACY_0_3_0_CONFIG["devices"]["human"]["bundle"]
    )
    assert b.per_channel_band.voices[0].midi_channel == 7      # wire ch 7
    assert b.per_channel_band.voices[0].scale == "dorian"
    assert b.per_channel_band.voices[0].min_interval_seconds == 3.7
    assert b.per_channel_cc.voices[0].midi_channel == 5
    assert b.per_channel_cc.voices[0].cc_number == 88
    assert b.markov.channel == 6
    assert b.markov.scale == "lydian"
    assert b.threshold.channel == 9
    assert b.coherence.channel == 10
    assert b.clips.rules[0].midi_channel == 15
