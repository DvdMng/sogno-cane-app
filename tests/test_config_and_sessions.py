"""Configuration in use, UOMO+CANE recording sessions, voice descriptions.

Field requests (0.9.4): show the configuration in use with a save-and-
overwrite button; start the human and dog recordings in sync; explain what
EVENTI, COERENZA, MARKOV and CLIP actually do.
"""
import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _window(tmp_path):
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.main_window import MainWindow
    return MainWindow(archive=Archive(root=str(tmp_path / "rec")))


def _pump(ms):
    from PySide6.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


class _Answer:
    """Stands in for a modal dialog: records each call, returns ``value``."""

    def __init__(self, value):
        self.value = value
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        return self.value


def _save_named(name, bundle_human=None):
    from sogno_cane import config as appconfig
    from sogno_cane.midi.presets import rich_vocabulary_preset
    path = os.path.join(appconfig.configs_dir(), f"{name}.json")
    appconfig.save_config(path, appconfig.build_config({
        "human": {"profile": "HUMAN", "port": "", "loop": True,
                  "bundle": bundle_human or rich_vocabulary_preset()},
        "dog": {"profile": "DOG", "port": "", "loop": True,
                "bundle": rich_vocabulary_preset()},
    }))
    return path


# --------------------------------------------------------------------------- #
# The configuration in use                                                    #
# --------------------------------------------------------------------------- #

def test_starts_on_the_default_configuration_and_flags_changes(app, tmp_path):
    win = _window(tmp_path)
    try:
        assert win._cfg_name_lbl.text() == "predefinita"
        assert not win._config_dirty()
        assert "●" not in win.windowTitle()
        win._human.bundle.markov.scale = "lydian"
        win._refresh_config_view()
        assert win._config_dirty()
        assert win._cfg_state_lbl.text() == "● non salvata"
        assert win.windowTitle().endswith("●")
        assert "●" in win._cfg_studio_lbl.text()
    finally:
        win.close()


def test_salva_asks_for_a_name_once_then_overwrites_silently(
        app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    from sogno_cane import config as appconfig
    ask = _Answer(("concerto", True))
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(ask))
    win = _window(tmp_path)
    try:
        win._human.bundle.markov.scale = "lydian"
        win._save_config()                     # no name yet: asks for one
        path = os.path.join(appconfig.configs_dir(), "concerto.json")
        assert len(ask.calls) == 1
        assert win._config_path == path and os.path.exists(path)
        assert win._cfg_name_lbl.text() == "concerto"
        assert win._cfg_state_lbl.text() == "salvata"

        win._human.bundle.markov.scale = "phrygian"
        assert win._config_dirty()
        win._save_config()                     # named: overwrites, no dialog
        assert len(ask.calls) == 1
        saved = appconfig.load_config(path)
        assert saved["devices"]["human"]["bundle"]["markov"]["scale"] \
            == "phrygian"
        assert not win._config_dirty()
    finally:
        win.close()


def test_salva_come_never_overwrites_another_configuration_silently(
        app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QInputDialog, QMessageBox
    other = _save_named("prova")
    before = open(other, encoding="utf-8").read()
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(_Answer(("prova", True))))
    no = _Answer(QMessageBox.StandardButton.No)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(no))
    win = _window(tmp_path)
    try:
        win._human.bundle.markov.scale = "lydian"
        win._save_config_as()
        assert len(no.calls) == 1, "must ask before overwriting"
        assert open(other, encoding="utf-8").read() == before
        assert win._config_path is None
        yes = _Answer(QMessageBox.StandardButton.Yes)
        monkeypatch.setattr(QMessageBox, "question", staticmethod(yes))
        win._save_config_as()
        assert win._config_path == other
        assert json.load(open(other, encoding="utf-8"))["devices"]["human"][
            "bundle"]["markov"]["scale"] == "lydian"
    finally:
        win.close()


def test_carica_warns_before_discarding_unsaved_changes(
        app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    from sogno_cane.midi.presets import rich_vocabulary_preset
    b = rich_vocabulary_preset()
    b.markov.root = "F"
    path = _save_named("notte", b)
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (path, "")))
    win = _window(tmp_path)
    try:
        win._human.bundle.markov.scale = "lydian"          # not saved
        no = _Answer(QMessageBox.StandardButton.No)
        monkeypatch.setattr(QMessageBox, "question", staticmethod(no))
        win._load_config()
        assert len(no.calls) == 1
        assert win._human.bundle.markov.scale == "lydian"
        assert win._config_path is None
        yes = _Answer(QMessageBox.StandardButton.Yes)
        monkeypatch.setattr(QMessageBox, "question", staticmethod(yes))
        win._load_config()
        assert win._config_path == path
        assert win._human.bundle.markov.root == "F"
        assert win._human.bundle.markov.scale == "dorian"  # from the file
        assert not win._config_dirty()
        assert win._cfg_name_lbl.text() == "notte"
    finally:
        win.close()


def test_loading_a_clean_configuration_does_not_ask(app, tmp_path,
                                                    monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    path = _save_named("pulita")
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (path, "")))
    asked = _Answer(QMessageBox.StandardButton.No)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(asked))
    win = _window(tmp_path)
    try:
        win._load_config()
        assert asked.calls == []
        assert win._config_path == path
    finally:
        win.close()


def test_reopens_on_the_configuration_in_use_with_unsaved_changes(
        app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(_Answer(("concerto", True))))
    win = _window(tmp_path)
    win._human.bundle.markov.scale = "lydian"
    win._save_config()                         # saved as "concerto"
    win._human.bundle.markov.root = "G"        # changed afterwards
    win.close()

    again = _window(tmp_path)
    try:
        assert again._cfg_name_lbl.text() == "concerto"
        assert again._human.bundle.markov.scale == "lydian"
        assert again._human.bundle.markov.root == "G"     # change kept...
        assert again._config_dirty()                     # ...and flagged
        assert again._cfg_state_lbl.text() == "● modifiche non salvate"
        again._human.bundle.markov.root = "D"            # the saved value
        assert not again._config_dirty()
    finally:
        again.close()


def test_a_deleted_configuration_reopens_as_unsaved(app, tmp_path,
                                                    monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(_Answer(("effimera", True))))
    win = _window(tmp_path)
    win._human.bundle.markov.scale = "lydian"
    win._save_config()
    path = win._config_path
    win.close()
    os.remove(path)

    again = _window(tmp_path)
    try:
        assert again._config_path is None
        assert again._cfg_name_lbl.text() == "predefinita"
        assert again._human.bundle.markov.scale == "lydian"  # not lost
        assert again._config_dirty()
    finally:
        again.close()


def test_choosing_another_midi_port_does_not_flag_the_configuration(
        app, tmp_path):
    win = _window(tmp_path)
    try:
        win._human._port_cb.addItem("Porta di prova")
        win._human._port_cb.setCurrentText("Porta di prova")
        assert not win._config_dirty()
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# UOMO+CANE recording sessions                                                #
# --------------------------------------------------------------------------- #

def test_registra_uomo_cane_starts_and_stops_both_together(app, tmp_path):
    win = _window(tmp_path)
    h, d = win._human, win._dog
    try:
        win._toggle_record_both()
        assert h.is_recording and d.is_recording
        assert h.is_streaming and d.is_streaming
        assert h._rec_session and h._rec_session == d._rec_session
        assert h._rec_started and h._rec_started == d._rec_started
        assert win._rec_both_btn.text() == "■ FERMA REGISTRAZIONI"
        _pump(700)
        win._toggle_record_both()
        assert not h.is_recording and not d.is_recording
        assert win._rec_both_btn.text() == "● REGISTRA UOMO+CANE"
        metas = win._archive.list()
        assert len(metas) == 2
        assert metas[0].session and metas[0].session == metas[1].session
        assert {m.panel for m in metas} == {"UMANO", "CANE"}
        assert metas[0].created == metas[1].created
        assert sorted(m.name for m in metas) == [
            f"Sessione {metas[0].created} · CANE",
            f"Sessione {metas[0].created} · UMANO",
        ]
        assert abs(metas[0].duration_seconds
                   - metas[1].duration_seconds) <= 0.2
    finally:
        win._stop_all()
        win.close()


def test_a_single_panel_recording_is_unchanged(app, tmp_path):
    win = _window(tmp_path)
    try:
        win._human._toggle_stream()
        win._human._toggle_record()
        assert win._rec_both_btn.text() == "■ FERMA REGISTRAZIONI"
        _pump(400)
        win._human._toggle_record()
        (meta,) = win._archive.list()
        assert meta.session == ""
        assert meta.panel == "UMANO"
        assert meta.name.startswith("HUMAN ")
        assert win._rec_both_btn.text() == "● REGISTRA UOMO+CANE"
    finally:
        win._stop_all()
        win.close()


def test_the_global_button_also_stops_a_single_panel_recording(app,
                                                               tmp_path):
    win = _window(tmp_path)
    try:
        win._dog._toggle_record()
        _pump(300)
        win._toggle_record_both()
        assert not win._dog.is_recording and not win._human.is_recording
        assert len(win._archive.list()) == 1
    finally:
        win._stop_all()
        win.close()


def test_carica_sessione_plays_the_pair_on_both_panels(app, tmp_path):
    win = _window(tmp_path)
    try:
        win._toggle_record_both()
        _pump(600)
        win._toggle_record_both()
        win._stop_all()
        panel = win._archive_panel
        panel.refresh()
        panel._table.selectRow(0)
        assert panel._play_session_btn.isEnabled()
        panel._do_load_session()
        assert win._human.is_streaming and win._dog.is_streaming
        assert win._human.current_mode() == "file"
        assert win._dog.current_mode() == "file"
        assert win._human._playback.source_name.endswith("· UMANO")
        assert win._dog._playback.source_name.endswith("· CANE")
    finally:
        win._stop_all()
        win.close()


def test_a_cut_copy_leaves_the_session(app, tmp_path):
    win = _window(tmp_path)
    try:
        win._toggle_record_both()
        _pump(600)
        win._toggle_record_both()
        win._stop_all()
        arc = win._archive
        first = arc.list()[0]
        cut_id = arc.trim(first.rec_id, 0.0, 0.2)
        assert arc.load(cut_id).meta.session == ""
        assert len(arc.session_members(first.session)) == 2
        panel = win._archive_panel
        panel.refresh()
        panel._select_id(cut_id)
        assert not panel._play_session_btn.isEnabled()
    finally:
        win._stop_all()
        win.close()


def test_recordings_from_older_versions_still_load(tmp_path):
    from sogno_cane.io.recording import Recording, RecordingMeta
    meta = RecordingMeta(
        rec_id="r1", name="vecchia", profile="HUMAN", channel_names=["a"],
        sample_rate_hz=250.0, n_samples=3, created="2026-01-01 10:00:00",
    )
    Recording(eeg=np.zeros((3, 1)), meta=meta).save(str(tmp_path))
    p = tmp_path / "r1.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    del data["session"], data["panel"]       # as 0.9.3 wrote it
    p.write_text(json.dumps(data), encoding="utf-8")
    back = Recording.load(str(tmp_path / "r1.npz"))
    assert back.meta.session == "" and back.meta.panel == ""
    assert back.meta.name == "vecchia"


# --------------------------------------------------------------------------- #
# Voice descriptions                                                          #
# --------------------------------------------------------------------------- #

def test_every_voice_tab_describes_what_it_does(app):
    from PySide6.QtWidgets import QLabel, QTabWidget
    from sogno_cane.midi.presets import rich_vocabulary_preset
    from sogno_cane.ui.mapping_panel import MappingPanel
    panel = MappingPanel(rich_vocabulary_preset(), "T")
    tabs = panel.findChild(QTabWidget)
    text = {}
    for i in range(tabs.count()):
        labels = tabs.widget(i).findChildren(QLabel)
        text[tabs.tabText(i)] = " ".join(
            lbl.text() for lbl in labels
            if lbl.wordWrap() and len(lbl.text()) > 150
        )
    for name in ("NOTE / CANALE", "CC / CANALE", "EVENTI", "COERENZA",
                 "MARKOV", "CLIP"):
        assert text[name], f"{name} has no description"
    assert "Non confronta l'uomo con il cane" in text["COERENZA"]
    assert "canale MIDI 10" in text["EVENTI"]
    assert "canale MIDI 11" in text["COERENZA"]
    assert "canale MIDI 9" in text["MARKOV"]
    assert "canale MIDI 12" in text["CLIP"]


def test_descriptions_match_the_default_preset():
    """The descriptions quote the defaults: if these change, update them."""
    from sogno_cane.midi.presets import rich_vocabulary_preset
    b = rich_vocabulary_preset()
    assert b.threshold.channel + 1 == 10 and b.threshold.band == "beta"
    assert tuple(b.threshold.notes[:2]) == (60, 62)   # pentatonic from C4
    assert b.coherence.channel + 1 == 11 and b.coherence.cc_number == 1
    assert b.coherence.channels_a == (0, 1, 2, 3)
    assert b.markov.channel + 1 == 9
    assert (b.markov.scale, b.markov.root) == ("dorian", "D")
    assert {r.midi_channel + 1 for r in b.clips.rules} == {12}
    assert [r.midi_note for r in b.clips.rules] == list(range(36, 44))
    assert b.clips.relative_threshold == 3.0
    assert {v.midi_channel + 1 for v in b.per_channel_cc.voices} == {13}
