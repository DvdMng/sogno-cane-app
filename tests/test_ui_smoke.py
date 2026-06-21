"""Headless (offscreen) smoke tests for the Qt UI.

Skipped automatically if PySide6 cannot start an offscreen platform.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    a = QApplication.instance() or QApplication([])
    yield a


def test_main_window_builds(app, tmp_path, monkeypatch):
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.main_window import MainWindow
    from sogno_cane.ui.theme import apply_theme

    apply_theme(app)
    win = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    win.show()
    assert win._tabs.count() == 3
    assert {win._tabs.tabText(i) for i in range(3)} == {
        "STUDIO", "MAPPING", "ARCHIVE",
    }
    win.close()


def test_record_and_archive_flow(app, tmp_path, monkeypatch):
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    from PySide6.QtCore import QEventLoop, QTimer
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.main_window import MainWindow

    arc = Archive(root=str(tmp_path / "rec"))
    win = MainWindow(archive=arc)
    win.show()

    def pump(ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    win._human._toggle_stream()
    win._human._start_record()
    pump(700)
    win._human._stop_record(save=True)
    pump(100)
    assert len(arc.list()) >= 1
    win._stop_all()
    win.close()


def test_playback_via_device(app, tmp_path, monkeypatch):
    import numpy as np
    from PySide6.QtCore import QEventLoop, QTimer
    from sogno_cane.io.archive import Archive
    from sogno_cane.io.loaders import LoadedEEG
    from sogno_cane.ui.main_window import MainWindow

    win = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    loaded = LoadedEEG(
        data=np.random.default_rng(0).normal(0, 20, (500, 8)),
        sample_rate_hz=250.0,
        channel_names=[f"c{i}" for i in range(8)],
        source_name="synthetic.csv",
    )
    win._dog.play_loaded(loaded, autostart=True)

    def pump(ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    pump(400)
    assert win._dog.is_streaming
    win._stop_all()
    win.close()
