"""UX guarantees added in 0.8.0.

These cover the things a user complained about being unable to tell:
which version is running, whether an update actually changed anything,
which source is active, and whether sound can come out at all.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from sogno_cane import __version__, update as updater  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    a = QApplication.instance() or QApplication([])
    yield a


# --------------------------------------------------------------------------- #
# "Did anything change?" — version visibility and update notice               #
# --------------------------------------------------------------------------- #

def test_applied_update_notice_round_trip(tmp_path, monkeypatch):
    """An installed update must leave exactly one readable trace."""
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    assert updater.take_applied_notice() is None      # nothing yet
    updater.record_applied("9.9.9", "cose nuove")
    notice = updater.take_applied_notice()
    assert notice is not None
    assert notice["version"] == "9.9.9"
    assert notice["notes"] == "cose nuove"
    # One-shot: it must not nag on every launch.
    assert updater.take_applied_notice() is None


def test_apply_pending_records_the_notice(tmp_path, monkeypatch):
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    staged = tmp_path / "staged" / "sogno_cane"
    staged.mkdir(parents=True)
    (staged / "__init__.py").write_text('__version__ = "9.9.9"')
    os.makedirs(updater.updates_dir(), exist_ok=True)
    import json
    with open(
        os.path.join(updater.updates_dir(), "pending.json"), "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {"version": "9.9.9", "package_dir": str(staged),
             "notes": "novità"}, f,
        )
    target = tmp_path / "install" / "sogno_cane"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text('__version__ = "0.0.1"')

    assert updater.apply_pending_update(str(target)) is True
    notice = updater.take_applied_notice()
    assert notice and notice["version"] == "9.9.9"
    assert notice["notes"] == "novità"


def test_window_title_and_tag_show_the_version(app, tmp_path, monkeypatch):
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.main_window import MainWindow

    win = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    try:
        assert __version__ in win.windowTitle()
        assert __version__ in win._version_lbl.text()
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# Source mode is visible, exclusive, and always truthful                      #
# --------------------------------------------------------------------------- #

def _panel(tmp_path):
    from sogno_cane.eeg.profiles import HUMAN_PROFILE
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.device_panel import DevicePanel
    return DevicePanel(
        "UMANO", HUMAN_PROFILE, archive=Archive(root=str(tmp_path / "rec")),
    )


def test_mode_buttons_are_exclusive_and_start_on_simulator(app, tmp_path):
    panel = _panel(tmp_path)
    try:
        assert panel.current_mode() == "sim"
        assert panel._sim_btn.isChecked()
        assert not panel._load_btn.isChecked()
        assert not panel._live_btn.isChecked()
        assert panel._mode_group.exclusive()
    finally:
        panel.shutdown()


def test_cancelled_file_choice_leaves_mode_truthful(app, tmp_path, monkeypatch):
    """A cancelled file dialog must not leave FILE selected while the
    simulator is what is actually running."""
    from PySide6.QtWidgets import QFileDialog
    panel = _panel(tmp_path)
    try:
        monkeypatch.setattr(
            QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: ("", "")),
        )
        panel._load_btn.setChecked(True)      # what the click would do
        panel._on_mode_file()
        assert panel.current_mode() == "sim"
        assert panel._sim_btn.isChecked()
        assert not panel._load_btn.isChecked()
    finally:
        panel.shutdown()


def test_loop_checkbox_only_enabled_for_files(app, tmp_path):
    panel = _panel(tmp_path)
    try:
        panel._sync_mode_buttons()
        assert not panel._loop_cb.isEnabled(), "loop is meaningless live"
    finally:
        panel.shutdown()


def test_human_and_dog_panels_are_visually_distinct(app, tmp_path):
    from sogno_cane.eeg.profiles import DOG_PROFILE, HUMAN_PROFILE
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.device_panel import DevicePanel
    arc = Archive(root=str(tmp_path / "rec"))
    h = DevicePanel("UMANO", HUMAN_PROFILE, archive=arc)
    d = DevicePanel("CANE", DOG_PROFILE, archive=arc)
    try:
        assert h.objectName() == "DevicePanelHuman"
        assert d.objectName() == "DevicePanelDog"
        assert h.objectName() != d.objectName()
    finally:
        h.shutdown()
        d.shutdown()


# --------------------------------------------------------------------------- #
# Silent failure is never silent                                              #
# --------------------------------------------------------------------------- #

def test_starting_without_midi_says_so_without_blocking(app, tmp_path):
    """No modal (it would freeze a performance) but an unmissable message."""
    panel = _panel(tmp_path)
    try:
        assert not panel._midi.is_open
        panel._toggle_stream()               # must return immediately
        assert panel.is_streaming
        assert "MIDI" in panel._status.text().upper()
        panel._update_stats()
        assert "MIDI NON COLLEGATO" in panel._stats.text()
    finally:
        panel.shutdown()


def test_health_line_reports_the_simulator_when_idle(app, tmp_path):
    panel = _panel(tmp_path)
    try:
        panel._update_stats()
        text = panel._stats.text()
        assert "SIMULATORE" in text
        assert "MIDI" in text
    finally:
        panel.shutdown()


def test_connect_button_is_a_call_to_action_until_connected(app, tmp_path):
    panel = _panel(tmp_path)
    try:
        assert panel._connect_btn.objectName() == "CallToAction"
        assert panel._connect_btn.text() == "COLLEGA"
    finally:
        panel.shutdown()
