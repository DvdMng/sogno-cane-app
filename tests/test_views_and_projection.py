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
import subprocess
import sys
import textwrap

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
    ctl._gain_row.slider.setValue(200)
    assert seen[-1].gain == 2.0
    assert ctl._gain_row.label.text() == "2,00 ×"
    ctl._seconds_row.slider.setValue(12)
    assert seen[-1].seconds == 12.0
    ctl._full_cb.setChecked(False)
    assert seen[-1].fullscreen is False
    assert not ctl._drift_cb.isVisibleTo(ctl)       # 3D only: not shown
    assert not ctl._shape_cb.isVisibleTo(ctl)
    ctl._content_cb.setCurrentIndex(0)              # back to 3D
    ctl._shape_cb.setCurrentIndex(1)
    assert seen[-1].shape == "mesh" and ctl._shape_cb.isVisibleTo(ctl)


def test_the_projection_panel_shows_only_what_can_be_set(app):
    """Field report on 0.9.11: with the 3D view «secondi» and «ampiezza»
    were shown switched off («non sono configurabili»), and spin boxes are
    a poor way to adjust a look. Now every row shown can be set, as a
    slider where it is a quantity: for the 3D view STUDIO's REGOLAZIONI
    (the fabric only with MESH), for the traces their field and height."""
    from PySide6.QtWidgets import (
        QAbstractButton,
        QAbstractSpinBox,
        QComboBox,
        QSlider,
    )
    from sogno_cane.ui.landscape_view import ViewStyle
    from sogno_cane.ui.projection import ProjectionControls, \
        ProjectionSettings
    style = ViewStyle()
    ctl = ProjectionControls(ProjectionSettings(), style)
    assert not ctl.findChildren(QAbstractSpinBox)

    def shown():
        return [w for kind in (QAbstractButton, QComboBox, QSlider)
                for w in ctl.findChildren(kind) if w.isVisibleTo(ctl)]

    rows = ctl._style_rows.rows
    for content, shape in ((0, 0), (0, 1), (1, 0), (1, 1)):
        ctl._content_cb.setCurrentIndex(content)
        ctl._shape_cb.setCurrentIndex(shape)
        assert all(w.isEnabled() for w in shown()), (content, shape)
        is_3d, mesh = content == 0, shape == 1
        for name, row in rows.items():
            assert row.slider.isVisibleTo(ctl) == (
                is_3d and (name != "fabric" or mesh)), (name, content, shape)
        assert ctl._seconds_row.slider.isVisibleTo(ctl) is not is_3d
        assert ctl._gain_row.slider.isVisibleTo(ctl) is not is_3d
    # The 3D rows move the style they were given (STUDIO's), at once.
    ctl._content_cb.setCurrentIndex(0)
    seen = []
    ctl.style_changed.connect(lambda: seen.append(True))
    rows["contrast"].slider.setValue(170)
    assert style.contrast == pytest.approx(1.7) and seen == [True]
    assert rows["contrast"].label.text() == "170 %"


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
    wave = win._traces["dog"]                       # another field: kept
    win.apply(ProjectionSettings(content="traces", layout="dog",
                                 seconds=12.0))
    assert win._traces["dog"] is wave and wave._n_visible == 12 * FS
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
    assert win._projection_btn.text() == "PROIEZIONE ●"
    win._toggle_projection()
    _pump(30)
    assert win._projection is None
    assert win._projection_btn.text() == "PROIEZIONE"
    win.close()


def _key(widget, key):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication
    QApplication.sendEvent(widget, QKeyEvent(
        QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))


def test_full_screen_projection_has_no_title_bar_and_esc_brings_it_back(
        app, tmp_path):
    """Field request: at full screen, no title bar, even on the only screen
    there is; Esc brings the window (and its title bar) back, a second Esc
    closes it, F11 goes back to full screen. Esc reaching the app (the
    keys may still go to it) does the same."""
    from PySide6.QtCore import Qt
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.main_window import MainWindow
    win = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    win._show_projection_controls()
    win._toggle_projection()
    proj = win._projection
    frameless = Qt.WindowType.FramelessWindowHint
    assert proj.isFullScreen() and proj.windowFlags() & frameless
    _key(proj, Qt.Key.Key_Escape)
    assert not proj.isFullScreen()
    assert not proj.windowFlags() & frameless          # title bar back
    assert win._projection is proj                     # still open
    _key(proj, Qt.Key.Key_F11)
    assert proj.isFullScreen()
    win._on_escape()                                   # Esc in the app
    assert not proj.isFullScreen()
    _key(proj, Qt.Key.Key_Escape)                      # second Esc: closes
    _pump(30)
    assert win._projection is None
    win.close()


_FULL_SCREEN_CHILD = textwrap.dedent(r'''
    import os, sys, tempfile, time
    sys.path.insert(0, ROOT)
    os.environ["SOGNO_CANE_HOME"] = tempfile.mkdtemp()
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QGuiApplication, QKeyEvent
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    if app.platformName() != "windows":
        print("NOT-WINDOWS")
        sys.exit(0)
    from sogno_cane.eeg.profiles import DOG_PROFILE, HUMAN_PROFILE
    from sogno_cane.ui import projection as pj

    place = pj.ProjectionWindow.place

    def hidden(self, screen, full=None):       # never on the screen
        self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        place(self, screen, full)

    pj.ProjectionWindow.place = hidden

    def pump(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.005)

    def key(k):
        QApplication.sendEvent(win, QKeyEvent(
            QEvent.Type.KeyPress, k, Qt.KeyboardModifier.NoModifier))

    win = pj.ProjectionWindow({"human": HUMAN_PROFILE, "dog": DOG_PROFILE},
                              pj.ProjectionSettings(fullscreen=True))
    closed = []
    win.closed.connect(lambda: closed.append(True))
    win.place(QGuiApplication.primaryScreen())
    pump(0.4)
    seen = [win.isFullScreen()]
    for k in [Qt.Key.Key_Escape, Qt.Key.Key_F11] * 2 + [Qt.Key.Key_Escape]:
        if closed:                     # closed too early: the states so far
            break
        key(k)
        pump(0.4)
        seen.append("closed" if closed else win.isFullScreen())
    if not closed:
        key(Qt.Key.Key_Escape)
        pump(0.2)
    print("STATES", seen, "CLOSED", closed == [True])
''').replace("ROOT", repr(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def test_esc_takes_the_projection_out_of_full_screen_on_windows():
    """On Windows itself: changing the title bar makes a new native window,
    and Windows brought it back still full screen, so the first Esc only
    put the title bar back. Out of full screen first, now every Esc and F11
    does what it says. A child process on the real platform, the window
    kept off the screen."""
    env = dict(os.environ)
    env.pop("QT_QPA_PLATFORM", None)
    env["PYTHONIOENCODING"] = "utf-8"
    done = subprocess.run([sys.executable, "-c", _FULL_SCREEN_CHILD],
                          env=env, capture_output=True, text=True,
                          timeout=120)
    if "NOT-WINDOWS" in done.stdout:
        return
    assert "STATES" in done.stdout, done.stdout + done.stderr
    assert "STATES [True, False, True, False, True, False] CLOSED True" \
        in done.stdout, done.stdout


# --------------------------------------------------------------------------- #
# A navigable 3D view (0.9.10)                                                #
# --------------------------------------------------------------------------- #


def _drag(widget, start, end, button=None):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    button = button or Qt.MouseButton.LeftButton
    for kind, (x, y) in ((QEvent.Type.MouseButtonPress, start),
                         (QEvent.Type.MouseMove, end),
                         (QEvent.Type.MouseButtonRelease, end)):
        pos = QPointF(x, y)
        held = Qt.MouseButton.NoButton \
            if kind == QEvent.Type.MouseButtonRelease else button
        QApplication.sendEvent(widget, QMouseEvent(
            kind, pos, widget.mapToGlobal(pos), button, held,
            Qt.KeyboardModifier.NoModifier))


def _wheel(widget, notches):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication
    pos = QPointF(widget.width() / 2, widget.height() / 2)
    QApplication.sendEvent(widget, QWheelEvent(
        pos, widget.mapToGlobal(pos), QPoint(0, 0),
        QPoint(0, int(120 * notches)), Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase,
        False))


def _double_click(widget):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    pos = QPointF(widget.width() / 2, widget.height() / 2)
    QApplication.sendEvent(widget, QMouseEvent(
        QEvent.Type.MouseButtonDblClick, pos, widget.mapToGlobal(pos),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier))


def test_the_3d_view_turns_moves_and_comes_closer_with_the_mouse(app):
    from PySide6.QtCore import Qt
    from sogno_cane.ui.landscape_view import LandscapeView
    view = LandscapeView({"human": HUMAN_PROFILE, "dog": DOG_PROFILE},
                         background="#000000")
    view.resize(800, 450)
    view.drift = False
    view.shape = "mesh"
    _feed(view)
    framed = view.grab().toImage()
    _drag(view, (400, 220), (520, 260))                # turn
    nav = view.navigation
    assert nav.yaw != 0.0 and nav.pitch != 0.0
    turned = view.grab().toImage()
    assert turned != framed
    _drag(view, (400, 220), (480, 220), Qt.MouseButton.RightButton)
    assert nav.pan_x > 0.0                             # moved, right
    _wheel(view, 3)
    assert nav.zoom > 1.2                              # closer
    _double_click(view)
    assert nav.to_dict() == type(nav)().to_dict()      # framed again
    assert view.grab().toImage() == framed


def test_the_point_of_view_is_kept_and_the_projector_shares_it(app,
                                                               tmp_path):
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.main_window import MainWindow
    win = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    win._landscape.navigation.turn(150.0, 40.0)
    win._landscape.navigation.scale(4)
    kept = win._landscape.navigation.to_dict()
    win._show_projection_controls()
    win._toggle_projection()
    proj = win._projection
    assert proj._view.navigation is win._landscape.navigation
    win._landscape.navigation.move(0.1, 0.0)           # turned on the PC
    assert proj._view.navigation.pan_x == 0.1          # and on the screen
    _double_click(proj._view)                          # still closes it
    _pump(30)
    assert win._projection is None
    assert win._landscape.navigation.pan_x == 0.1      # view untouched
    win._landscape.navigation.move(-0.1, 0.0)
    win.close()
    again = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    assert again._landscape.navigation.to_dict() == kept
    again.close()


# --------------------------------------------------------------------------- #
# REGOLAZIONI of the 3D view (0.9.11)                                         #
# --------------------------------------------------------------------------- #


def test_view_style_keeps_to_its_ranges_and_tolerates_old_files():
    from sogno_cane.ui.landscape_view import ViewStyle
    style = ViewStyle()
    assert style.to_dict() == ViewStyle.DEFAULTS
    style.set("contrast", 99)
    style.set("thickness", 0)
    style.set("fabric", -1)
    style.set("seconds", "dieci")              # not a number: unchanged
    style.set("gain", float("nan"))
    assert (style.contrast, style.thickness, style.fabric) == (2.5, 1.0, 0.0)
    assert (style.seconds, style.gain) == (6.0, 1.0)
    assert ViewStyle.from_dict(style.to_dict()).to_dict() == style.to_dict()
    assert ViewStyle.from_dict(None).to_dict() == ViewStyle.DEFAULTS
    old = ViewStyle.from_dict({"seconds": 40, "contrast": "?", "x": 1})
    assert old.seconds == 20.0 and old.contrast == 1.0


def _lit(image, threshold=40):
    return sum(1 for x in range(0, image.width(), 3)
               for y in range(0, image.height(), 3)
               if image.pixelColor(x, y).value() > threshold)


def _brightness(image):
    return sum(image.pixelColor(x, y).value()
               for x in range(0, image.width(), 3)
               for y in range(0, image.height(), 3))


def test_another_field_in_view_keeps_what_was_there(app):
    """The field in view is a slider, dragged even while projecting: it
    must not empty the view. The newest samples stay, in the 3D view and in
    the projection's traces, the field growing into the past or losing
    its oldest part."""
    from sogno_cane.ui.landscape_view import LandscapeView
    from sogno_cane.ui.waveform_widget import WaveformWidget
    view = LandscapeView({"human": HUMAN_PROFILE, "dog": DOG_PROFILE},
                         background="#000000")
    view.resize(800, 450)
    view.drift = False
    view.set_visible_channels("dog", [0, 1])
    _feed(view, 4.0)

    def newest(s, k):
        return s._ring[s._head + s.n - k:s._head + s.n].copy()

    before = newest(view._streams["human"], 500)
    assert np.abs(before).sum() > 0
    for seconds in (12.0, 3.0):
        view.style.seconds = seconds
        view._follow_style()
        s = view._streams["human"]
        assert s.n >= seconds * FS - s.step
        assert np.array_equal(newest(s, 500), before), seconds
    assert view.visible_channels("dog") == [0, 1]
    assert _lit(view.grab().toImage()) > 200

    wave = WaveformWidget(HUMAN_PROFILE, seconds_visible=4.0)
    rows = np.random.default_rng(3).normal(0, 20, (4 * FS, 8))
    wave._append(rows.astype(np.float32))

    def window(w):
        return w._ring[w._head:w._head + w._n_visible]

    for seconds in (12.0, 2.0):
        wave.set_seconds_visible(seconds)
        assert wave._n_visible == seconds * FS
        k = int(min(4.0, seconds) * FS)
        assert np.allclose(window(wave)[-k:], rows[-k:]), seconds


def test_the_regolazioni_change_the_3d_picture(app):
    """Contrast brightens the lines, thickness widens them, the fabric goes
    from nothing (only the waves) to twice as present; another field in
    view goes on at that length, keeping the waves chosen. The height of
    the waves is the same setting as the view's."""
    from sogno_cane.ui.landscape_view import LandscapeView
    view = LandscapeView({"human": HUMAN_PROFILE, "dog": DOG_PROFILE},
                         background="#000000")
    view.resize(800, 450)
    view.drift = False
    view.shape = "mesh"
    _feed(view)
    style = view.style
    base = view.grab().toImage()
    style.contrast = 2.0
    assert _brightness(view.grab().toImage()) > 1.2 * _brightness(base)
    style.contrast = 1.0
    style.thickness = 3.0
    assert _lit(view.grab().toImage()) > 1.2 * _lit(base)
    style.thickness = 1.0
    style.fabric = 0.0
    bare = view.grab().toImage()
    style.fabric = 2.0
    dense = view.grab().toImage()
    assert _brightness(bare) < _brightness(base) < _brightness(dense)
    style.fabric = 1.0
    assert view.grab().toImage() == base       # back as it was
    view.gain = 2.0
    assert style.gain == 2.0
    style.gain = 1.0
    view.set_visible_channels("dog", [0, 1])
    style.seconds = 12.0
    view.advance()
    assert view._seconds == 12.0
    assert all(s.n >= 12 * FS - s.step for s in view._streams.values())
    assert view.visible_channels("dog") == [0, 1]
    view.grab()                                # must not raise


def test_the_regolazioni_panel_moves_the_style_and_resets_it(app):
    from sogno_cane.ui.landscape_view import ViewStyle
    from sogno_cane.ui.style_controls import ViewStyleControls
    style = ViewStyle()
    ctl = ViewStyleControls(style)
    seen = []
    ctl.changed.connect(lambda: seen.append(True))
    for name, pos, value, text in (
            ("contrast", 150, 1.5, "150 %"),
            ("thickness", 25, 2.5, "2,5 px"),
            ("seconds", 12, 12.0, "12 s"),
            ("gain", 150, 1.5, "1,50 ×"),
            ("fabric", 0, 0.0, "solo le onde")):
        row = ctl.rows[name]
        row.slider.setValue(pos)
        assert getattr(style, name) == pytest.approx(value), name
        assert row.label.text() == text
    assert len(seen) == 5                       # every change at once
    ctl.reset()
    assert style.to_dict() == ViewStyle.DEFAULTS
    assert ctl.rows["contrast"].slider.value() == 100
    assert ctl.rows["fabric"].label.text() == "100 %"
    other = ViewStyle.from_dict({"thickness": 2.0})
    ctl.set_style(other)
    assert ctl.rows["thickness"].slider.value() == 20
    assert len(seen) == 6                       # loading is not a change
    # The fabric acts only on the MESH shape: only there is it shown.
    ctl.set_shape("lines")
    assert not ctl.rows["fabric"].slider.isVisibleTo(ctl)
    assert ctl.rows["contrast"].slider.isVisibleTo(ctl)
    ctl.set_shape("mesh")
    assert ctl.rows["fabric"].slider.isVisibleTo(ctl)


def test_the_regolazioni_are_kept_and_the_projector_shares_them(app,
                                                               tmp_path):
    from PySide6.QtWidgets import QPushButton
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.main_window import MainWindow
    from sogno_cane.ui.projection import ProjectionSettings
    win = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    button = [b for b in win.findChildren(QPushButton)
              if b.text() == "REGOLAZIONI"]
    assert len(button) == 1
    button[0].click()
    ctl = win._style_controls
    assert ctl is not None and ctl.isVisible()
    ctl.rows["contrast"].slider.setValue(180)
    assert win._landscape.style.contrast == pytest.approx(1.8)
    assert not ctl.rows["fabric"].slider.isVisibleTo(ctl)   # STUDIO: LINEE
    win._shape_btns["mesh"].click()
    assert ctl.rows["fabric"].slider.isVisibleTo(ctl)
    # The projection's panel has the same REGOLAZIONI, on the same style:
    # moved there, STUDIO's panel shows it, and the other way round.
    win._show_projection_controls()
    pc = win._projection_controls
    rows = pc._style_rows.rows
    assert rows["contrast"].slider.value() == 180
    rows["thickness"].slider.setValue(20)
    assert win._landscape.style.thickness == pytest.approx(2.0)
    assert ctl.rows["thickness"].slider.value() == 20
    ctl.rows["gain"].slider.setValue(140)
    assert rows["gain"].slider.value() == 140
    ctl.rows["gain"].slider.setValue(100)
    win._toggle_projection()
    proj = win._projection
    assert proj._view.style is win._landscape.style
    proj.apply(ProjectionSettings(gain=2.5))    # the traces' height only
    assert win._landscape.style.gain == 1.0
    win._toggle_projection()
    _pump(30)
    kept = win._landscape.style.to_dict()
    win.close()
    again = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    assert again._landscape.style.to_dict() == kept
    assert again._landscape.style.contrast == pytest.approx(1.8)
    again.close()


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
    # Widths relative to the row itself (fonts differ with the platform),
    # and the rule checked on the width the row really got: one line
    # exactly when it has room for one.
    for width, lines in ((one_line + 200, 1), (one_line - 40, None),
                         (least, 2)):
        win.resize(width, 880)
        _pump(30)
        room = win._transport.width() >= row.sizeHint().width()
        assert (row.lines() == 1) == room, (width, row.lines())
        if lines == 1:
            assert row.lines() == 1, width
        elif lines:
            assert row.lines() >= 2, width
        assert win._transport.height() >= \
            row.heightForWidth(win._transport.width())
        for b in win._transport.findChildren(QPushButton):
            need = min(b.sizeHint().width(), b.maximumWidth())   # «?»: fixed
            assert b.width() >= need, (width, b.text())
    # The row on one line no longer holds the window that wide (with the
    # app's theme the minimum went from 1533 px to 1121).
    assert least < one_line - 100
    win.close()


def test_the_3d_options_never_widen_the_window(app, tmp_path):
    """The note on how to move about the 3D view wraps on a narrow window:
    as one long line it held the whole window 600 px wider with VISTA 3D
    on, the very squeeze that cut the transport buttons."""
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.main_window import MainWindow
    win = MainWindow(archive=Archive(root=str(tmp_path / "rec")))
    win.show()
    win._view3d_btn.setChecked(False)
    _pump(30)
    off = win.minimumSizeHint().width()
    win._view3d_btn.setChecked(True)
    _pump(30)
    assert win.minimumSizeHint().width() <= off + 2
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
