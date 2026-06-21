import os

from sogno_cane.settings import Settings


def test_defaults():
    s = Settings()
    assert s.get("human", "profile") == "HUMAN"
    assert s.get("dog", "profile") == "DOG"
    assert s.get("nope", default=123) == 123


def test_set_and_get():
    s = Settings()
    s.set("human", "port", "loopMIDI Port")
    assert s.get("human", "port") == "loopMIDI Port"


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    s = Settings()
    s.set("human", "port", "PortA")
    s.set("window", "w", 1234)
    s.save()
    assert os.path.exists(str(tmp_path / "settings.json"))
    s2 = Settings.load()
    assert s2.get("human", "port") == "PortA"
    assert s2.get("window", "w") == 1234


def test_corrupt_file_yields_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    (tmp_path / "settings.json").write_text("{ this is not json")
    s = Settings.load()
    assert s.get("human", "profile") == "HUMAN"   # falls back to defaults


def test_loads_settings_with_utf8_bom(tmp_path, monkeypatch):
    """The Windows installer (PowerShell) may write a BOM; it must still load."""
    import json
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    with open(tmp_path / "settings.json", "w", encoding="utf-8-sig") as f:
        json.dump({"update_url": "https://x/v.json", "auto_update": True}, f)
    s = Settings.load()
    assert s.get("update_url") == "https://x/v.json"
    assert s.get("auto_update") is True
