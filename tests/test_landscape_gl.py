"""The 3D view on the graphics card (0.9.8).

Field request: smoother scrolling, more depth, and the dense moments of a
real EEG kept instead of smoothed away, without cheap effects. The scene's
data path is tested here directly; the drawing itself in a child process
on the real platform, since the rest of the suite runs offscreen, where
there is no OpenGL (the app then falls back to the QPainter view, covered
by the other tests).
"""
import os
import subprocess
import sys
import textwrap

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor  # noqa: E402

from sogno_cane.eeg.profiles import DOG_PROFILE, HUMAN_PROFILE  # noqa: E402
from sogno_cane.eeg.unicorn_packet import (  # noqa: E402
    PACKET_VALUES,
    VALIDATION_INDEX,
    UnicornPacket,
)

FS = 250
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _packet(rows):
    s = np.zeros((rows.shape[0], PACKET_VALUES), dtype=np.float32)
    s[:, :8] = rows
    s[:, VALIDATION_INDEX] = 1.0
    return UnicornPacket(samples=s, timestamp=0.0, sample_rate=FS)


def _eeg(seconds, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    x = rng.normal(0.0, 6.0, (t.size, 8))
    x += 15.0 * np.sin(2 * np.pi * 10.0 * t)[:, None]
    return x + 40000.0


def _scene():
    from sogno_cane.ui.landscape_gl import _Scene
    return _Scene({"human": HUMAN_PROFILE, "dog": DOG_PROFILE}, 6.0,
                  QColor("#000000"))


def _run(scene, human, dog=None, seconds=10.0, dog_rate=FS, clock=1000.0):
    fh, fd = 0, 0
    for k in range(int(seconds * 60)):
        clock += 1.0 / 60.0
        while fh < (k / 60.0) * FS + 25 and fh + 25 <= human.shape[0]:
            scene.push("human", _packet(human[fh:fh + 25]), clock)
            fh += 25
        while dog is not None and fd < (k / 60.0) * dog_rate + 25 \
                and fd + 25 <= dog.shape[0]:
            scene.push("dog", _packet(dog[fd:fd + 25]), clock)
            fd += 25
        scene.advance(clock)
        assert 0.0 <= scene.frac < 1.0
    return clock


def test_both_beings_share_the_columns_even_with_a_fast_clock():
    """The two headsets each keep their own clock (a Unicorn runs at about
    250.6 Hz): the one ahead loses a sample now and then, so the human and
    the dog stay on the same columns and the surface can join them."""
    scene = _scene()
    _run(scene, _eeg(12, 1), _eeg(12, 2), seconds=10.0, dog_rate=251.0)
    assert abs(scene.count - 10 * FS) < 120          # a column per sample
    assert all(f.queued <= 25 + scene.step for f in scene.feeds.values())


def test_the_dense_moments_of_an_eeg_come_through():
    """A one-second burst at 20 Hz keeps most of its size on the strands
    (the QPainter surface, smoothed at 20 ms, kept 4 % of it); the threads
    between the strands keep less, so the fabric stays calm."""
    t = np.arange(8 * FS) / FS
    x = np.zeros((t.size, 8)) + 40000.0
    x[:, :] += 8.0 * np.sin(2 * np.pi * 3.0 * t)[:, None]
    burst = (t >= 5.0) & (t < 6.0)
    x[burst, 0] += 30.0 * np.sin(2 * np.pi * 20.0 * t[burst])
    scene = _scene()
    _run(scene, x, seconds=7.5)
    scene.make_layout(scene.strands())
    cols = np.arange(scene.count - scene.n, scene.count)
    heights = scene.heights(cols)
    raw = scene.values["human"][cols % scene.n, 0]
    strand = heights[:, scene._lay["strand_rows"][0]]
    thread = heights[:, scene._lay["strand_rows"][0] + 1]

    def at_20hz(sig):
        seg = sig - np.convolve(sig, np.ones(25) / 25, mode="same")
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
        freqs = np.fft.rfftfreq(seg.size, 1.0 / FS)
        return spec[np.argmin(np.abs(freqs - 20.0))]

    ref = at_20hz(raw)
    assert ref > 0
    assert at_20hz(strand) / ref > 0.6
    assert at_20hz(thread) < at_20hz(strand)


def test_a_chaotic_moment_stays_large_against_the_calm():
    """Field request: the 3D view was "too normalized", chaotic moments and
    interference did not show. A two-second artefact at eight times the
    usual size now stands well above the calm without flat tops (the old
    soft limit capped everything at 1), and the calm after it keeps its
    size: the scale follows the median of the activity, slowly, so the
    artefact barely moves it."""
    rng = np.random.default_rng(3)
    t = np.arange(14 * FS) / FS
    x = rng.normal(0.0, 10.0, (t.size, 8)) + 40000.0
    burst = (t >= 10.0) & (t < 12.0)
    x[burst] += rng.normal(0.0, 80.0, (int(burst.sum()), 8))
    scene = _scene()
    _run(scene, x, seconds=13.5)
    cols = np.arange(scene.count - scene.n, scene.count)
    size = np.abs(scene.values["human"][cols % scene.n]).mean(axis=1)
    sec = cols / FS                      # one column per sample here
    calm_before = np.median(size[(sec > 7.5) & (sec < 9.8)])
    loud = np.percentile(size[(sec > 10.2) & (sec < 11.8)], 95)
    calm_after = np.median(size[sec > 12.3])
    assert loud > 1.1                    # above the old ceiling of 1
    assert loud > 4.0 * calm_before
    assert calm_after > 0.8 * calm_before


def test_navigation_starts_from_the_framed_view_and_stays_above():
    """Field request: "rendi navigabile il 3d". Untouched, the point of
    view is the framed one; turned, it stays 4 to 85 degrees above the
    surface, and dragging back answers at once (no dead zone)."""
    import math
    from sogno_cane.ui.landscape_view import Navigation
    scene = _scene()
    scene.shape = "mesh"
    base = scene._camera(6.0, 0.3, navigated=False)
    same = scene._camera(6.0, 0.3)
    assert np.allclose(base[0], same[0])
    target = np.array([0.35 * 6.0, 0.0, 0.42 * 26.0])
    nav = Navigation()
    nav.turn(400.0, 0.0)                         # a turn: same distance
    turned = nav.orbit(base[0], target)
    assert math.isclose(np.linalg.norm(turned - target),
                        np.linalg.norm(base[0] - target))
    assert not np.allclose(turned, base[0])
    nav.turn(0.0, 5000.0)                        # far past the top
    top = nav.orbit(base[0], target)
    assert math.degrees(math.asin((top - target)[1] / np.linalg.norm(
        top - target))) <= 85.0 + 1e-6
    nav.turn(0.0, -20.0)                         # back: moves at once
    assert not np.allclose(nav.orbit(base[0], target), top)
    nav.scale(30)
    assert nav.zoom == 4.0
    nav.move(5.0, -5.0)
    assert (nav.pan_x, nav.pan_y) == (1.0, -1.0)
    again = Navigation.from_dict(nav.to_dict())
    assert again.to_dict() == nav.to_dict()
    odd = Navigation.from_dict({"zoom": "x", "yaw": float("nan"),
                                "pan_x": 9})
    assert (odd.zoom, odd.yaw, odd.pan_x) == (1.0, 0.0, 1.0)


def test_rows_are_the_strands_and_the_threads_between_them():
    """One surface, nothing under it: no walls, no base (the block of 0.9.8
    was not wanted)."""
    from sogno_cane.ui.landscape_view import _SUBDIV, _THREAD_WEIGHT
    scene = _scene()
    scene.make_layout(scene.strands())
    lay = scene._lay
    rows = lay["strand_rows"]
    assert lay["rows"] == 15 * _SUBDIV + 1
    assert list(rows) == [i * _SUBDIV for i in range(16)]
    assert set(lay["w"][rows]) == {1.0}
    threads = [i for i in range(lay["rows"]) if i not in set(rows)]
    assert np.allclose(lay["w"][threads], _THREAD_WEIGHT)
    mix = lay["mix_s"] + lay["mix_t"]
    for j, r in enumerate(rows):
        assert mix[r, j] == 1.0 and mix[r].sum() == 1.0     # on the strand
    assert np.allclose(mix[threads].sum(axis=1), 1.0)


def test_another_field_in_view_keeps_the_columns():
    """Changing the field in view (a slider, dragged even while projecting)
    respaces the columns already made instead of starting empty: shorter,
    the newest ones as they were; longer, each new column the mean of the
    old ones it spans. The feeds go on (same cleaning, pacing, scale)."""
    scene = _scene()
    clock = _run(scene, _eeg(12, 1), _eeg(12, 2), seconds=8.0)
    assert scene.count > scene.n                      # the field is full
    feeds = scene.feeds
    order = np.arange(scene.count - scene.n, scene.count) % scene.n
    old = {k: v[order] for k, v in scene.values.items()}
    short = scene.resized(3.0)
    assert short.feeds is feeds and short.n == 750 and short.count == 750
    for kind, cols in old.items():
        assert np.allclose(short.values[kind][:750], cols[-750:])
    long = scene.resized(12.0)
    assert long.step == 2 and long.count == scene.n // 2
    for kind, cols in old.items():
        pairs = cols.reshape(-1, 2, cols.shape[1]).mean(axis=1)
        assert np.allclose(long.values[kind][:long.count], pairs,
                           atol=1e-5)
    assert all(f._recent.shape[0] == long.total for f in feeds.values())
    # And it goes on from there.
    _run(long, _eeg(4, 3), _eeg(4, 4), seconds=2.0, clock=clock)
    assert long.count > scene.n // 2 + 150


def test_without_opengl_the_app_keeps_the_qpainter_view():
    from PySide6.QtWidgets import QApplication
    from sogno_cane.ui.landscape_gl import gl_available, make_landscape_view
    from sogno_cane.ui.landscape_view import LandscapeView
    QApplication.instance() or QApplication([])
    assert not gl_available()                     # offscreen: no OpenGL
    view = make_landscape_view({"human": HUMAN_PROFILE, "dog": DOG_PROFILE})
    assert isinstance(view, LandscapeView)
    assert view.FRAME_MS == 33


_CHILD = textwrap.dedent(r'''
    import os, sys, time
    import numpy as np
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from PySide6.QtGui import (QGuiApplication, QOffscreenSurface,
                               QOpenGLContext, QSurfaceFormat, QColor)
    from PySide6.QtOpenGL import (QOpenGLFramebufferObject,
                                  QOpenGLFramebufferObjectFormat)
    app = QGuiApplication([])
    from sogno_cane.ui import landscape_gl as lg
    if not lg.gl_available():
        print("NO-GL")
        sys.exit(0)
    from test_landscape_gl import _eeg, _packet
    from sogno_cane.eeg.profiles import DOG_PROFILE, HUMAN_PROFILE
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    ctx = QOpenGLContext()
    ctx.setFormat(fmt)
    assert ctx.create()
    surf = QOffscreenSurface()
    surf.setFormat(ctx.format())
    surf.create()
    assert ctx.makeCurrent(surf)
    W, H = 960, 540
    ff = QOpenGLFramebufferObjectFormat()
    ff.setAttachment(QOpenGLFramebufferObject.Attachment.Depth)
    fbo = QOpenGLFramebufferObject(W, H, ff)
    f = ctx.functions()

    def frame(scene, clock):
        fbo.bind()
        scene.render(f, W, H, clock)
        assert f.glGetError() == 0
        img = fbo.toImage()
        a = np.frombuffer(img.constBits(), np.uint8).reshape(H, W, 4)
        return a[..., :3].astype(int).copy()

    scene = lg._Scene({"human": HUMAN_PROFILE, "dog": DOG_PROFILE}, 6.0,
                      QColor("#000000"))
    scene.shape = "mesh"
    scene.drift = False
    assert scene.init_gl()
    human, dog = _eeg(10, 1), _eeg(10, 2)
    clock, fed, cost = 1000.0, 0, []
    for k in range(420):
        clock += 1.0 / 60.0
        while fed < (k / 60.0) * 250 + 25:
            scene.push("human", _packet(human[fed:fed + 25]), clock)
            scene.push("dog", _packet(dog[fed:fed + 25]), clock)
            fed += 25
        t0 = time.perf_counter()
        scene.advance(clock)
        fbo.bind()
        scene.render(f, W, H, clock)
        cost.append(time.perf_counter() - t0)
    a = frame(scene, clock)
    b, g, r = a[..., 0], a[..., 1], a[..., 2]      # QImage bytes: B, G, R
    lit = (r + g + b) > 60
    pink = lit & (r > g + 40)
    cyan = lit & (b > r + 30) & (g > r + 20)
    assert lit.sum() > 5000 and pink.sum() > 300 and cyan.sum() > 300, \
        (lit.sum(), pink.sum(), cyan.sum())
    # Moves by fractions of a sample: a step with no new column, only a
    # larger fraction, and the picture has moved.
    prev, t = a, clock
    for _ in range(40):
        t += 0.0005
        count, frac = scene.count, scene.frac
        scene.advance(t)
        later = frame(scene, t)
        if scene.count == count and scene.frac > frac:
            assert np.abs(later - prev).sum() > 0
            break
        prev = later
    else:
        raise AssertionError("no step by a fraction of a sample")
    # Hidden lines: with the surface sunk far below its lines, nothing
    # hides them any more, so more of them show.
    sink = lg._SINK
    lg._SINK = 50.0
    bare = frame(scene, t)
    lg._SINK = sink
    assert (bare.sum(axis=2) > 60).sum() > (later.sum(axis=2) > 60).sum()
    # REGOLAZIONI: contrast brightens the lines, thickness widens them, the
    # fabric goes from nothing (only the waves) to twice as present.
    def lit(img):
        return int((img.sum(axis=2) > 60).sum())
    style = scene.style
    look = frame(scene, t)
    style.contrast = 2.0
    assert frame(scene, t).sum() > 1.2 * look.sum()
    style.contrast = 1.0
    style.thickness = 3.0
    assert lit(frame(scene, t)) > 1.2 * lit(look)
    style.thickness = 1.0
    style.fabric = 0.0
    no_fabric = frame(scene, t)
    style.fabric = 2.0
    dense = frame(scene, t)
    style.fabric = 1.0
    assert no_fabric.sum() < look.sum() < dense.sum()
    # Fewer waves, another order, the lines: rebuilt without a GL error.
    scene.feeds["dog"].visible = [0, 1, 2, 3]
    frame(scene, clock)
    scene.order = "interleaved"
    frame(scene, clock)
    scene.shape = "lines"
    frame(scene, clock)
    # Another field in view, as the widget does before a frame: the scene
    # goes on at that length with what it holds, its GL objects moved
    # along, and the very next frame shows the past (nothing new fed).
    import types
    holder = types.SimpleNamespace(_scene=scene, _ready=True)
    scene.shape = "mesh"
    held = lit(frame(scene, clock))
    style.seconds = 12.0
    lg.LandscapeGLView._follow_style(holder)
    fresh = holder._scene
    assert fresh is not scene and fresh.seconds == 12.0
    assert fresh._gl is not None and scene._gl is None
    assert fresh.style is style and fresh.navigation is scene.navigation
    assert fresh.feeds["dog"].visible == [0, 1, 2, 3]
    assert (fresh.order, fresh.shape) == ("interleaved", "mesh")
    assert fresh.count == scene.n // 2
    assert lit(frame(fresh, clock)) > 0.4 * held, (held,)
    style.seconds = 3.0                       # and shorter
    lg.LandscapeGLView._follow_style(holder)
    assert lit(frame(holder._scene, clock)) > 0.4 * held
    style.seconds = 12.0
    lg.LandscapeGLView._follow_style(holder)
    fresh = holder._scene
    for k in range(60):
        clock += 1.0 / 60.0
        while fed + 25 <= human.shape[0] and \
                fed < (420 + k) / 60.0 * 250 + 25:
            fresh.push("human", _packet(human[fed:fed + 25]), clock)
            fresh.push("dog", _packet(dog[fed:fed + 25]), clock)
            fed += 25
        fresh.advance(clock)
    assert fresh.count > 0
    assert lit(frame(fresh, clock)) > 1000
    print("cpu ms", round(sorted(cost[60:])[len(cost[60:]) // 2] * 1000, 2))
    print("OK")
''').replace("ROOT", repr(ROOT))


def test_the_gl_view_draws_colours_hides_lines_and_moves_smoothly():
    env = dict(os.environ)
    env.pop("QT_QPA_PLATFORM", None)
    env["PYTHONIOENCODING"] = "utf-8"
    done = subprocess.run([sys.executable, "-c", _CHILD], env=env,
                          capture_output=True, text=True, timeout=300)
    if "NO-GL" in done.stdout:
        return                     # no OpenGL on this machine: nothing to see
    assert "OK" in done.stdout, done.stdout + done.stderr
    cpu = float(done.stdout.split("cpu ms")[1].split()[0])
    assert cpu < 6.0, cpu          # the processor's share of a frame
