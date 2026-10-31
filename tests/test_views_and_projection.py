"""Field requests for 0.9.6, display and CC side.

* Morbidezza: choose the stretch of the knob a CC moves over (e.g. 30-90),
  and set the smoothing to the hundredth, with its response time shown.
* The dog's amplifier has 4 outputs: draw only some waves, keeping the
  option to show all eight.
* A 3D view joining the human's and the dog's waves.
* Full screen on a projector set up as an extended desktop, while the app
  keeps working on the PC's screen; everything about it configurable.

And for 0.9.7: the 3D view as one surface (mesh), and a window that works
at any width a screen gives it, the transport row wrapping instead of
cutting its buttons.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from sogno_cane.eeg.bands import BandPowers  # noqa: E402
from sogno_cane.eeg.profiles import DOG_PROFILE, HUMAN_PROFILE  # noqa: E402
from sogno_cane.eeg.unicorn_packet import (  # noqa: E402
    PACKET_VALUES,
    VALIDATION_INDEX,
    UnicornPacket,
)
from sogno_cane.midi.context import WindowContext  # noqa: E402
from sogno_cane.midi.presets import rich_vocabulary_preset  # noqa: E402

FS = 250
BANDS = ("delta", "theta", "alpha", "beta", "gamma")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _pump(ms):
    from PySide6.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _packet(rows):
    s = np.zeros((rows.shape[0], PACKET_VALUES), dtype=np.float32)
    s[:, :8] = rows
    s[:, VALIDATION_INDEX] = 1.0
    return UnicornPacket(samples=s, timestamp=0.0, sample_rate=FS)


def _eeg(seconds, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    x = rng.normal(0.0, 8.0, (t.size, 8))
    x += 20.0 * np.sin(2 * np.pi * 10.0 * t)[:, None]
    return x + 50000.0


def _bands(alpha):
    return BandPowers(by_band={b: np.full(8, alpha if b == "alpha" else 5.0)
                               for b in BANDS},
                      sample_rate_hz=FS, window_samples=FS)


def _ctx(t):
    return WindowContext(t_seconds=t, hop_seconds=0.1, sample_rate_hz=FS,
                         window_samples=FS)


# --------------------------------------------------------------------------- #
# CC output range and smoothing                                               #
# --------------------------------------------------------------------------- #

def test_cc_per_channel_moves_only_within_its_range():
    b = rich_vocabulary_preset()
    s = b.per_channel_cc
    for v in s.voices:
        v.out_min, v.out_max, v.smoothing = 30, 90, 0.0
    values = []
    for i in range(600):
        p = 10 ** (1.0 + 1.5 * np.sin(i / 7.0))       # sweeps the range
        values += [e.value for e in s.on_window(np.zeros((FS, 8)),
                                                 _bands(p), _ctx(i * 0.1))]
    assert values and min(values) == 30 and max(values) == 90


def test_cc_defaults_still_use_the_whole_range():
    b = rich_vocabulary_preset()
    assert {(v.out_min, v.out_max) for v in b.per_channel_cc.voices} \
        == {(0, 127)}
    assert (b.coherence.out_min, b.coherence.out_max) == (0, 127)


def test_coherence_moves_only_within_its_range():
    from sogno_cane.core.sources import ArrayPlaybackSource  # noqa: F401
    from sogno_cane.midi.mapper import MappingConfig, MappingEngine
    b = rich_vocabulary_preset()
    s = b.coherence
    s.out_min, s.out_max = 40, 80
    eng = MappingEngine(MappingConfig(sample_rate_hz=FS))
    eng.add_strategy(s)
    eng.reset_strategy_state()
    x = _eeg(20)
    values = []
    for i in range(0, x.shape[0], 25):
        values += [e.value for e in eng.process_packet(_packet(x[i:i + 25]))
                   if e.kind == "cc"]
    assert values and min(values) >= 40 and max(values) <= 80
    assert max(values) > 70        # a shared 10 Hz rhythm: near the top


def test_ranges_and_smoothing_survive_a_saved_configuration():
    from sogno_cane import config as appconfig
    b1 = rich_vocabulary_preset()
    b1.per_channel_cc.voices[3].out_min = 25
    b1.per_channel_cc.voices[3].out_max = 100
    b1.per_channel_cc.voices[3].smoothing = 0.93
    b1.coherence.out_max = 64
    data = appconfig.serialize_bundle(b1)
    b2 = rich_vocabulary_preset()
    appconfig.apply_bundle_config(b2, data)
    v = b2.per_channel_cc.voices[3]
    assert (v.out_min, v.out_max, v.smoothing) == (25, 100, 0.93)
    assert b2.coherence.out_max == 64


def test_morbidezza_is_set_as_a_time(app):
    """The morbidezza reads and sets the time the CC takes to follow the
    signal, not the coefficient behind it; the voice keeps its coefficient,
    so a configuration saved with 0,93 opens at 1,4 s."""
    import math
    from sogno_cane.ui.cc_controls import SmoothingControl, tau_from_smoothing
    voice = rich_vocabulary_preset().per_channel_cc.voices[0]
    voice.smoothing = 0.93
    ctl = SmoothingControl(voice, setattr, 0.5)
    assert ctl.field.text() == "1,4 s"
    assert voice.smoothing == 0.93                 # opening changes nothing
    ctl.field.setValue(2.0)
    assert math.isclose(voice.smoothing, math.exp(-0.1 / 2.0))
    assert math.isclose(ctl.seconds(), 2.0)
    ctl.slider.setValue(ctl.slider.maximum())
    assert ctl.field.text() == "30 s"
    assert math.isclose(tau_from_smoothing(voice.smoothing), 30.0)
    ctl.slider.setValue(0)
    assert voice.smoothing == 0.0 and ctl.field.text() == "nessuna"
    # As fine at the fast end as at the slow one: one step of the slider
    # moves the time by well under 1 %.
    for pos in (10, 500, 990):
        ctl.slider.setValue(pos)
        a = ctl.seconds()
        ctl.slider.setValue(pos + 1)
        assert 1.0 < ctl.seconds() / a < 1.01, pos
    # The wheel over an unclicked slider leaves it alone (page scrolling).
    assert ctl.slider.focusPolicy() == ctl.field.focusPolicy()


def _mouse(widget, kind, x):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    pos = QPointF(x, widget.height() / 2)
    held = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease \
        else Qt.MouseButton.LeftButton
    QApplication.sendEvent(widget, QMouseEvent(
        kind, pos, widget.mapToGlobal(pos), Qt.MouseButton.LeftButton,
        held, Qt.KeyboardModifier.NoModifier))


def test_range_controls_never_cross_and_reach_the_voice(app):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication
    from sogno_cane.ui.cc_controls import RangeControl
    voice = rich_vocabulary_preset().per_channel_cc.voices[0]
    ctl = RangeControl(voice, setattr)
    lo, hi = ctl.low, ctl.high
    hi.setValue(90)
    lo.setValue(30)
    assert (voice.out_min, voice.out_max) == (30, 90)
    assert ctl.bar.values() == (30, 90)
    lo.setValue(120)                       # cannot pass the high end
    assert voice.out_min == 90
    hi.setValue(10)                        # nor the high end the low one
    assert voice.out_max == 90
    # Dragged on the bar: the nearer end follows, and stops at the other.
    lo.setValue(0)
    hi.setValue(127)
    bar = ctl.bar
    _mouse(bar, QEvent.Type.MouseButtonPress, bar._x(0))
    _mouse(bar, QEvent.Type.MouseMove, bar._x(30))
    _mouse(bar, QEvent.Type.MouseButtonRelease, bar._x(30))
    assert (voice.out_min, voice.out_max) == (30, 127)
    assert (lo.value(), hi.value()) == (30, 127)
    _mouse(bar, QEvent.Type.MouseButtonPress, bar._x(127))
    _mouse(bar, QEvent.Type.MouseMove, bar._x(5))
    _mouse(bar, QEvent.Type.MouseButtonRelease, bar._x(5))
    assert (voice.out_min, voice.out_max) == (30, 30)
    # The arrow keys move the end touched last; Shift by ten.
    QApplication.sendEvent(bar, QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Right,
        Qt.KeyboardModifier.ShiftModifier))
    assert (voice.out_min, voice.out_max) == (30, 40)
    assert hi.value() == 40


def test_cc_tabs_set_morbidezza_and_range_with_the_new_controls(app):
    from PySide6.QtWidgets import QTabWidget
    from sogno_cane.ui.cc_controls import RangeControl, SmoothingControl
    from sogno_cane.ui.mapping_panel import MappingPanel
    bundle = rich_vocabulary_preset()
    panel = MappingPanel(bundle, "T")
    tabs = panel.findChild(QTabWidget)
    names = [tabs.tabText(i) for i in range(tabs.count())]
    voices = len(bundle.per_channel_cc.voices)
    for name, count in (("CC / CANALE", voices), ("COERENZA", 1)):
        page = tabs.widget(names.index(name))
        assert len(page.findChildren(SmoothingControl)) == count, name
        assert len(page.findChildren(RangeControl)) == count, name


# --------------------------------------------------------------------------- #
# Waves shown                                                                 #
# --------------------------------------------------------------------------- #

def test_trace_view_shows_only_the_chosen_waves(app):
    from sogno_cane.ui.waveform_widget import WaveformWidget
    w = WaveformWidget(DOG_PROFILE)
    w.set_visible_channels([0, 1, 2, 3])
    assert w.visible_channels() == [0, 1, 2, 3]
    assert [c.isVisible() for c in w._curves] == [True] * 4 + [False] * 4
    w.set_visible_channels([])             # nothing chosen: all of them
    assert w.visible_channels() == list(range(8))


def test_dog_panel_isolates_waves_and_keeps_at_least_one(app):
    from sogno_cane.ui.device_panel import DevicePanel
    panel = DevicePanel("CANE", DOG_PROFILE)
    seen = []
    panel.visible_channels_changed.connect(seen.append)
    panel.set_visible_channels([0, 1, 2, 3])
    assert panel.visible_channels() == [0, 1, 2, 3]
    assert seen[-1] == [0, 1, 2, 3]
    assert panel._chan_btn.text() == "ONDE 4/8"
    for act in panel._chan_actions[:4]:
        act.setChecked(False)              # the last one cannot go
    assert len(panel.visible_channels()) == 1
    panel.set_visible_channels(range(8))
    assert panel._chan_btn.text() == "ONDE 8/8"
    panel.shutdown()


def test_chosen_waves_survive_a_restart(app, tmp_path):
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.main_window import MainWindow
    win = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    win._dog.set_visible_channels([1, 3, 5, 7])
    win._view3d_btn.setChecked(True)
    win.close()
    again = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    assert again._dog.visible_channels() == [1, 3, 5, 7]
    assert again._view3d_btn.isChecked()
    assert again._landscape.visible_channels("dog") == [1, 3, 5, 7]
    again.close()


# --------------------------------------------------------------------------- #
# 3D view                                                                     #
# --------------------------------------------------------------------------- #

def _feed(view, seconds=4.0):
    human, dog = _eeg(seconds, 1), _eeg(seconds, 2)
    for i in range(0, human.shape[0], 25):
        view.push("human", _packet(human[i:i + 25]))
        view.push("dog", _packet(dog[i:i + 25]))
    view.advance()


def test_3d_view_draws_both_beings_without_trails(app):
    from sogno_cane.ui.landscape_view import LandscapeView
    view = LandscapeView({"human": HUMAN_PROFILE, "dog": DOG_PROFILE},
                         background="#000000")
    view.resize(800, 450)
    view.set_visible_channels("dog", [0, 1, 2, 3])
    _feed(view)
    first = view.grab().toImage()
    lit = sum(1 for x in range(0, 800, 4) for y in range(0, 450, 4)
              if first.pixelColor(x, y).value() > 40)
    assert lit > 200, "the strands must be drawn"
    # Same data, nothing new: the next frame is the same picture (no
    # motion trails, no grain changing from frame to frame).
    view.drift = False
    a = view.grab().toImage()
    b = view.grab().toImage()
    assert a == b


def test_3d_view_follows_channels_profile_and_order(app):
    from sogno_cane.ui.landscape_view import LandscapeView
    view = LandscapeView({"human": HUMAN_PROFILE, "dog": DOG_PROFILE})
    view.resize(600, 300)
    _feed(view, 3.0)
    view.set_visible_channels("dog", [2, 3])
    assert view.visible_channels("dog") == [2, 3]
    view.set_profile("dog", HUMAN_PROFILE)      # another headset there
    assert view.visible_channels("dog") == [2, 3]
    for order in ("grouped", "interleaved"):
        view.order = order
        view.grab()                             # must not raise


def test_mesh_joins_the_waves_into_one_surface(app):
    """The «mesh» shape: one surface woven from the waves, still drawn
    afresh each frame (same data, same picture), and it hides what lies
    behind it (the frame differs from the plain lines)."""
    from sogno_cane.ui.landscape_view import LandscapeView
    view = LandscapeView({"human": HUMAN_PROFILE, "dog": DOG_PROFILE},
                         background="#000000")
    view.resize(800, 450)
    view.drift = False
    _feed(view)
    lines = view.grab().toImage()
    view.shape = "mesh"
    a = view.grab().toImage()
    b = view.grab().toImage()
    assert a == b
    assert a != lines
    lit = sum(1 for x in range(0, 800, 4) for y in range(0, 450, 4)
              if a.pixelColor(x, y).value() > 40)
    assert lit > 300, "the surface's threads must be drawn"
    view.set_visible_channels("human", [0])
    view.set_visible_channels("dog", [])         # a single wave: lines
    view.grab()


def test_studio_shape_and_order_buttons_drive_the_view(app, tmp_path):
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.main_window import MainWindow
    win = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    win._shape_btns["mesh"].click()
    win._order_btns["interleaved"].click()
    assert win._landscape.shape == "mesh"
    assert win._landscape.order == "interleaved"
    win.close()
    again = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    assert again._landscape.shape == "mesh"
    assert again._shape_btns["mesh"].isChecked()
    assert again._landscape.order == "interleaved"
    again.close()


def test_view_3d_takes_the_place_of_the_traces(app, tmp_path):
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.main_window import MainWindow
    win = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    win.show()
    win._view3d_btn.setChecked(True)
    _pump(50)
    assert win._landscape.isVisible()
    assert not win._human._waveform.isVisible()
    assert not win._dog._waveform.isVisible()
    win._view3d_btn.setChecked(False)
    _pump(50)
    assert not win._landscape.isVisible()
    assert win._human._waveform.isVisible()
    win.close()


# --------------------------------------------------------------------------- #
# Projection                                                                  #
# --------------------------------------------------------------------------- #

class _Screen:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


def test_projection_picks_the_projector_not_the_pc_screen():
    from sogno_cane.ui.projection import default_screen
    pc, projector = _Screen("PC"), _Screen("EPSON")
    assert default_screen([pc, projector], pc) is projector
    assert default_screen([pc, projector], pc, preferred="PC") is pc
    assert default_screen([pc], pc) is pc


def test_projection_settings_round_trip_and_tolerate_old_files():
    from sogno_cane.ui.projection import ProjectionSettings
    s = ProjectionSettings(screen="EPSON", fullscreen=False,
                           content="traces", layout="side", shape="mesh",
                           seconds=9.0, gain=1.5, labels=False, drift=False)
    assert ProjectionSettings.from_dict(s.to_dict()) == s
    old = ProjectionSettings.from_dict({"content": "3d", "grain": True,
                                        "layout": "side"})
    assert old.layout == "grouped"     # not a 3D layout: first one instead
    assert old.shape == "lines"        # 0.9.6 files: the shape they had
    assert ProjectionSettings.from_dict({"shape": "?"}).shape == "lines"
    assert ProjectionSettings.from_dict(None) == ProjectionSettings()


def test_projection_controls_apply_every_change_at_once(app):
    from sogno_cane.ui.projection import ProjectionControls, \
        ProjectionSettings
    ctl = ProjectionControls(ProjectionSettings())
    seen = []
    ctl.settings_changed.connect(seen.append)
    ctl._content_cb.setCurrentIndex(1)              # Tracce
    assert seen[-1].content == "traces"
    assert seen[-1].layout == "stacked"
    ctl._gain_sb.setValue(2.0)
    assert seen[-1].gain == 2.0
    ctl._full_cb.setChecked(False)
    assert seen[-1].fullscreen is False
    assert not ctl._drift_cb.isEnabled()            # 3D only
    assert not ctl._shape_cb.isEnabled()
    ctl._content_cb.setCurrentIndex(0)              # back to 3D
    ctl._shape_cb.setCurrentIndex(1)
    assert seen[-1].shape == "mesh" and ctl._shape_cb.isEnabled()


def test_projection_window_is_fed_and_reconfigured_live(app):
    from sogno_cane.ui.projection import ProjectionSettings, \
        ProjectionWindow
    win = ProjectionWindow({"human": HUMAN_PROFILE, "dog": DOG_PROFILE},
                           ProjectionSettings(), {"dog": [0, 1, 2, 3]})
    win.place(None)
    x = _eeg(2.0)
    for i in range(0, x.shape[0], 25):
        win.push("human", _packet(x[i:i + 25]))
        win.push("dog", _packet(x[i:i + 25]))
    _pump(60)
    assert win._view is not None
    assert win._view.visible_channels("dog") == [0, 1, 2, 3]
    view = win._view
    win.apply(ProjectionSettings(gain=2.0))         # live, same picture
    assert win._view is view and view.gain == 2.0
    win.apply(ProjectionSettings(gain=2.0, shape="mesh"))
    assert win._view is view and view.shape == "mesh"
    win.apply(ProjectionSettings(content="traces", layout="dog"))
    assert win._view is None and list(win._traces) == ["dog"]
    assert win._traces["dog"].visible_channels() == [0, 1, 2, 3]
    closed = []
    win.closed.connect(lambda: closed.append(True))
    win.close()
    assert closed == [True]


def test_projection_opens_and_closes_from_the_app(app, tmp_path):
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.main_window import MainWindow
    win = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    win._show_projection_controls()
    win._toggle_projection()
    assert win._projection is not None
    # One screen only (offscreen): a window, not full screen over the PC's.
    assert not win._projection.isFullScreen()
    assert win._projection_btn.text() == "PROIEZIONE ●"
    win._toggle_projection()
    _pump(30)
    assert win._projection is None
    assert win._projection_btn.text() == "PROIEZIONE"
    win.close()


# --------------------------------------------------------------------------- #
# Narrow windows (0.9.7)                                                      #
# --------------------------------------------------------------------------- #


def test_transport_row_wraps_instead_of_cutting_its_buttons(app, tmp_path):
    """A maximized window on a small or scaled screen can be narrower than
    the transport row: Qt then narrowed the widest button, and «● REGISTRA
    UOMO+CANE» lost its first and last letters. Now the group on the right
    moves to a second line, given the height it needs, and no button is
    ever narrower than its text."""
    from PySide6.QtWidgets import QLayout, QPushButton
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.main_window import MainWindow
    win = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    win.show()
    # What the window manager does to a maximized window: any width at all.
    win.layout().setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
    win.setMinimumSize(0, 0)
    _pump(30)
    row = win._transport.layout()
    margins = win.width() - win._transport.width()
    one_line = row.sizeHint().width() + margins
    least = win.minimumSizeHint().width()
    # Widths relative to the row itself: fonts differ with the platform.
    for width, lines in ((one_line + 20, 1), (one_line - 40, 2),
                         (least, None)):
        win.resize(width, 880)
        _pump(30)
        assert row.lines() == lines if lines else row.lines() >= 2, width
        assert win._transport.height() >= \
            row.heightForWidth(win._transport.width())
        for b in win._transport.findChildren(QPushButton):
            need = min(b.sizeHint().width(), b.maximumWidth())   # «?»: fixed
            assert b.width() >= need, (width, b.text())
    # The row on one line no longer holds the window that wide (with the
    # app's theme the minimum went from 1533 px to 1121).
    assert least < one_line - 100
    win.close()


def test_window_fits_the_screen_it_opens_on(app, tmp_path, monkeypatch):
    """A size saved larger than the screen (1613 px wide on a 1536 px
    screen, 1920 at 125 %) opens reduced to fit, frame included; on a
    bigger screen it stays as saved, and a screen too small for the window
    at all leaves it as it was. Sizes relative to the window's own minimum,
    which depends on the fonts."""
    from PySide6.QtCore import QRect
    from sogno_cane.io.archive import Archive
    from sogno_cane.settings import Settings
    from sogno_cane.ui.main_window import MainWindow

    class _Screen:
        def __init__(self, w, h):
            self._geometry = QRect(0, 0, w, h)

        def availableGeometry(self):  # noqa: N802 - Qt name
            return self._geometry

    probe = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    least = probe.minimumSizeHint()
    probe.close()
    saved = (least.width() + 300, least.height() + 200)

    def opened_on(w, h):
        settings = Settings.load()
        settings.set("window", "w", saved[0])
        settings.set("window", "h", saved[1])
        settings.save()
        monkeypatch.setattr(MainWindow, "screen",
                            lambda self: _Screen(w, h))
        win = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
        size = (win.width(), win.height())
        win.close()
        return size

    w, h = least.width() + 16 + 100, least.height() + 40 + 50
    assert opened_on(w, h) == (w - 16, h - 40)
    assert opened_on(saved[0] + 400, saved[1] + 400) == saved
    assert opened_on(least.width() - 100, least.height() - 100) == saved
