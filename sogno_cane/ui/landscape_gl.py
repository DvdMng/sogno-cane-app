"""Both EEGs in one perspective, drawn by the graphics card.

The scene of ``landscape_view`` (time running into the distance, the
strands of both beings side by side or interleaved, pink and cyan from one
channel to the next, the past dissolving), with what QPainter could not
afford inside the app's time budget:

* every sample, 250 a second per strand: the dense moments of a real EEG (a
  burst of fast activity, a muscle artefact) come through as dense texture
  instead of being averaged away;
* steady motion: the time axis moves by the fraction of a sample elapsed,
  sixty times a second, and each sample keeps its place on the surface as
  it recedes (nothing is re-averaged from one frame to the next);
* a volume: the surface is the top of a slab whose front and sides drop to
  a base, and hidden lines are removed exactly by the depth buffer,
  however dense the lines;
* almost no work for the processor: each frame only the newest samples
  are processed and uploaded, the card draws the rest, and the MIDI
  engine, in the same interpreter, keeps its time.

Lines only: no shading, no gradients, nothing blended over the last frame.
``make_landscape_view`` falls back to the QPainter view where OpenGL is
missing (and in the offscreen tests).
"""
from __future__ import annotations

import math
import os
import time
from collections import deque

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QMatrix4x4,
    QOffscreenSurface,
    QOpenGLContext,
    QPainter,
    QSurfaceFormat,
)
from PySide6.QtWidgets import QWidget

from sogno_cane.eeg.profiles import DeviceProfile
from sogno_cane.eeg.unicorn_packet import UnicornPacket
from sogno_cane.ui import landscape_view as lv
from sogno_cane.ui.theme import BG_DEEP, TEXT_DIM
from sogno_cane.ui.waveform_widget import _DisplayCleaner, _Playout, _palette

try:
    from PySide6.QtOpenGL import (
        QOpenGLBuffer,
        QOpenGLFramebufferObject,
        QOpenGLFramebufferObjectFormat,
        QOpenGLShader,
        QOpenGLShaderProgram,
    )
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
except ImportError:                  # a runtime without Qt's OpenGL modules
    QOpenGLWidget = None

_GL_FLOAT = 0x1406
_GL_UNSIGNED_INT = 0x1405
_GL_LINES = 0x0001
_GL_LINE_STRIP = 0x0003
_GL_TRIANGLES = 0x0004
_GL_TRIANGLE_STRIP = 0x0005
_GL_DEPTH_TEST = 0x0B71
_GL_BLEND = 0x0BE2
_GL_POLYGON_OFFSET_FILL = 0x8037
_GL_SRC_ALPHA = 0x0302
_GL_ONE_MINUS_SRC_ALPHA = 0x0303
_GL_COLOR_BUFFER_BIT = 0x4000
_GL_DEPTH_BUFFER_BIT = 0x0100
_GL_LESS = 0x0201
_GL_LEQUAL = 0x0203
_GL_TRUE, _GL_FALSE = 1, 0
_GL_FRAMEBUFFER = 0x8D40
_GL_READ_FRAMEBUFFER = 0x8CA8
_GL_DRAW_FRAMEBUFFER = 0x8CA9
_GL_NEAREST = 0x2600
_SAMPLES = 4                 # antialiasing: samples per pixel

_MAX_COLUMNS = 1500          # columns along time at most (6 s at 250 Hz)
_SECTION_S = 0.144           # a cross-section every so long (as before)
_FLOATS = 7                  # x, height, slot, weight, r, g, b
_BASE_DEPTH = 0.6            # the slab's base, under the surface
# Smoothing in time (Gaussian sigma): the strands keep what an EEG has up
# to about 20 Hz, so its dense moments come through; the threads between
# them a little less, so the fabric stays calm. Close enough that the
# surface has no pleats where the two meet.
_STRAND_SIGMA_S = 0.006
_THREAD_SIGMA_S = 0.010
# The opaque surface lies this far (a share of the relief) under its own
# lines: a ripple smaller than that does not cut a line behind it, a crest
# does.
_SINK = 0.06
_SCALE_TIME_S = 6.0          # the amplitude scale follows slowly: a dense
                             # burst keeps its size instead of being evened
_SCALE_EVERY_S = 0.5
_FIT_MARGIN = (0.90, 0.84)   # share of the frame the scene fills

_VERTEX = """
attribute vec4 a_xhsw;
attribute vec3 a_rgb;
uniform mat4 u_mvp;
uniform float u_newest;
uniform float u_span;
uniform float u_frac;
uniform float u_depth;
uniform float u_amp;
uniform float u_base;
uniform float u_sink;
uniform float u_mode;
uniform float u_alpha;
uniform float u_power;
uniform vec3 u_haze;
uniform vec3 u_bg;
uniform vec3 u_grid;
varying vec4 v_col;
void main() {
    float age = clamp((u_newest - a_xhsw.z + u_frac) / u_span, 0.0, 1.0);
    float y = a_xhsw.w < 0.0 ? u_base : max(a_xhsw.y * u_amp, u_base);
    if (u_mode < 0.5 && a_xhsw.w >= 0.0) {
        y -= u_sink;
    }
    gl_Position = u_mvp * vec4(a_xhsw.x, y, age * u_depth, 1.0);
    float fade = pow(1.0 - age, u_power);
    if (u_mode < 0.5) {
        v_col = vec4(u_bg, 1.0);
    } else if (u_mode < 1.5) {
        vec3 c = mix(a_rgb, u_haze, min(1.0, 1.3 * age));
        v_col = vec4(c, u_alpha * abs(a_xhsw.w) * fade);
    } else {
        v_col = vec4(u_grid, u_alpha * fade);
    }
}
"""

_FRAGMENT = """
varying vec4 v_col;
void main() {
    gl_FragColor = v_col;
}
"""

_GL_OK: bool | None = None


def gl_available() -> bool:
    """OpenGL usable here: a context, and our shaders compile on it."""
    global _GL_OK
    if _GL_OK is None:
        _GL_OK = _probe()
    return _GL_OK


def _probe() -> bool:
    if QOpenGLWidget is None or os.environ.get("SOGNO_CANE_NO_GL"):
        return False
    app = QGuiApplication.instance()
    if app is None or app.platformName() in ("offscreen", "minimal"):
        return False
    try:
        ctx = QOpenGLContext()
        if not ctx.create() or ctx.isOpenGLES():
            return False
        surface = QOffscreenSurface()
        surface.setFormat(ctx.format())
        surface.create()
        if not ctx.makeCurrent(surface):
            return False
        program = _program()
        ok = program is not None and ctx.format().majorVersion() >= 2
        ctx.doneCurrent()
        return ok
    except Exception:  # noqa: BLE001 - any failure: the QPainter view
        return False


def _program():
    p = QOpenGLShaderProgram()
    if not p.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex,
                                     _VERTEX):
        return None
    if not p.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment,
                                     _FRAGMENT):
        return None
    p.bindAttributeLocation("a_xhsw", 0)
    p.bindAttributeLocation("a_rgb", 1)
    return p if p.link() else None


def _rgb(c: QColor) -> tuple[float, float, float]:
    return c.redF(), c.greenF(), c.blueF()


class _Feed:
    """One being: cleaned and paced like the trace view. What the pacing
    reveals waits here until the scene turns it into columns."""

    def __init__(self, profile: DeviceProfile, window: int) -> None:
        self.profile = profile
        self.fs = float(profile.sample_rate_hz)
        self.n_ch = len(profile.channel_names)
        self.visible = list(range(self.n_ch))
        self._cleaner = _DisplayCleaner(self.n_ch, self.fs)
        self.playout = _Playout(self.fs)
        self._fifo: deque[np.ndarray] = deque()
        self.queued = 0
        self.seen = -1.0e9                   # when samples last came out
        self._recent = np.zeros((window, self.n_ch), dtype=np.float32)
        self._pos = 0
        self.filled = 0
        self.scale = 0.0                     # typical amplitude (uV)

    def push(self, packet: UnicornPacket, now: float) -> None:
        self.playout.push(self._cleaner.process(packet.eeg), now)

    def take(self, now: float) -> None:
        rows = self.playout.take(now)
        if rows is None:
            return
        rows = np.asarray(rows, dtype=np.float32)
        self._fifo.append(rows)
        self.queued += rows.shape[0]
        self.seen = now
        n = self._recent.shape[0]
        rows = rows[-n:]
        k = rows.shape[0]
        first = min(k, n - self._pos)
        self._recent[self._pos:self._pos + first] = rows[:first]
        if k > first:
            self._recent[:k - first] = rows[first:]
        self._pos = (self._pos + k) % n
        self.filled = min(n, self.filled + k)

    def pop(self, n: int) -> np.ndarray:
        out, need = [], n
        while need > 0:
            head = self._fifo[0]
            if head.shape[0] <= need:
                out.append(self._fifo.popleft())
                need -= head.shape[0]
            else:
                out.append(head[:need])
                self._fifo[0] = head[need:]
                need = 0
        self.queued -= n
        return out[0] if len(out) == 1 else np.concatenate(out)

    def drop(self, n: int) -> None:
        if n > 0:
            self.pop(n)

    def target_scale(self) -> float:
        if self.filled == 0 or not self.visible:
            return 0.0
        recent = self._recent[:self.filled, self.visible]
        return float(np.percentile(np.abs(recent), 90))


class _Scene:
    """Columns of samples on a conveyor running into the distance, and how
    to draw them. Pure state plus GL calls on the current context, so the
    widget and offscreen renderings share it."""

    def __init__(self, profiles: dict[str, DeviceProfile], seconds: float,
                 background: QColor) -> None:
        fs = max(float(p.sample_rate_hz) for p in profiles.values())
        total = max(16, int(round(seconds * fs)))
        self.total = total
        self.step = max(1, math.ceil(total / _MAX_COLUMNS))
        self.n = max(16, total // self.step)            # columns shown
        self.col_s = self.step / fs                     # seconds per column
        self.feeds = {k: _Feed(p, total) for k, p in profiles.items()}
        self.values = {k: np.zeros((self.n, f.n_ch), dtype=np.float32)
                       for k, f in self.feeds.items()}
        self._sums = {k: np.zeros(f.n_ch) for k, f in self.feeds.items()}
        self.count = 0              # columns made so far
        self.partial = 0            # samples in the column being made
        self.frac = 0.0             # how far the next column has come
        self.pending: set[int] = set()   # columns to (re)upload
        sig_s = max(0.3, _STRAND_SIGMA_S / self.col_s)
        sig_t = max(0.3, _THREAD_SIGMA_S / self.col_s)
        k = max(1, int(math.ceil(2.5 * max(sig_s, sig_t))))
        self._offsets = np.arange(-k, k + 1)
        self._kernels = (np.exp(-0.5 * (self._offsets / sig_s) ** 2),
                         np.exp(-0.5 * (self._offsets / sig_t) ** 2))
        self.order = "grouped"
        self.shape = "lines"
        self.show_labels = True
        self.gain = 1.0
        self.drift = True
        self.bg = QColor(background)
        self._t0 = time.perf_counter()
        self._next_scale = 0.0
        self._layout_key = None
        self._fit_key = None
        self._fit = (1.0, 0.0, 0.0)
        self._gl = None             # GL objects, made on the context
        self.labels: list[tuple[str, float, float]] = []

    # ------------------------------------------------------------------ #
    # Data                                                               #
    # ------------------------------------------------------------------ #

    def push(self, kind: str, packet: UnicornPacket, now: float) -> None:
        feed = self.feeds.get(kind)
        if feed is not None:
            feed.push(packet, now)

    def advance(self, now: float) -> None:
        """Turn what the pacing revealed into columns."""
        for feed in self.feeds.values():
            feed.take(now)
        # The amplitude scale: from the very first samples (without it the
        # first columns saturate into square waves), quickly over the first
        # seconds, then slowly, so a dense burst keeps its size.
        tick = now >= self._next_scale
        if tick:
            self._next_scale = now + _SCALE_EVERY_S
        for feed in self.feeds.values():
            if not tick and feed.scale > 0.0:
                continue
            target = feed.target_scale()
            if target > 0.0:
                if feed.scale <= 0.0:
                    a = 1.0
                elif feed.filled < 2.0 * feed.fs:
                    a = 0.5
                else:
                    a = min(1.0, _SCALE_EVERY_S / _SCALE_TIME_S)
                feed.scale += a * (target - feed.scale)
        active = {k: f for k, f in self.feeds.items() if now - f.seen < 1.0}
        if not active:
            return
        lead = min(active, key=lambda k: active[k].queued)
        n = active[lead].queued
        if n:
            chunks = {k: (f.pop(n) if k in active else None)
                      for k, f in self.feeds.items()}
            self._add(chunks, n)
        # A being whose clock runs a little fast gets ahead of the other:
        # drop its surplus now and then, so both stay on the same columns.
        for f in active.values():
            if f.queued > 25 + self.step:
                f.drop(f.queued - self.step)
        phase = active[lead].playout.phase()
        self.frac = min(0.999, (self.partial + phase) / self.step)

    def _add(self, chunks: dict, n: int) -> None:
        i = 0
        while i < n:
            take = min(self.step - self.partial, n - i)
            for k, block in chunks.items():
                if block is not None:
                    self._sums[k] += block[i:i + take].sum(axis=0)
            self.partial += take
            i += take
            if self.partial < self.step:
                break
            pos = self.count % self.n
            for k, feed in self.feeds.items():
                if chunks[k] is None:
                    self.values[k][pos] = 0.0
                else:
                    s = max(feed.scale, 1.0e-3)
                    raw = self._sums[k] / self.step
                    self.values[k][pos] = np.tanh(raw / (2.4 * s))
                self._sums[k][:] = 0.0
            self.count += 1
            self.partial = 0
            # This column, and the ones whose smoothed threads it changes.
            k = int(self._offsets[-1])
            self.pending.update(range(max(0, self.count - 1 - k),
                                      self.count))

    # ------------------------------------------------------------------ #
    # Layout                                                             #
    # ------------------------------------------------------------------ #

    def strands(self) -> list[tuple[str, int, QColor, float]]:
        """(kind, channel, colour, x), left to right, as the QPainter view."""
        per = {}
        for k, f in self.feeds.items():
            colors = _palette(len(f.visible))
            per[k] = [(k, ch, QColor(colors[i]))
                      for i, ch in enumerate(f.visible)]
        human, dog = per.get("human", []), per.get("dog", [])
        if self.order == "interleaved":
            ordered = []
            for i in range(max(len(human), len(dog))):
                ordered += human[i:i + 1] + dog[i:i + 1]
            xs = [i * lv._SPACING for i in range(len(ordered))]
        else:
            ordered = human + dog
            xs = [i * lv._SPACING for i in range(len(human))]
            start = (len(human) - 1) * lv._SPACING + lv._GROUP_GAP \
                if human else 0.0
            xs += [start + i * lv._SPACING for i in range(len(dog))]
        if not ordered:
            return []
        mid = (xs[0] + xs[-1]) / 2.0
        return [s + (x - mid,) for s, x in zip(ordered, xs)]

    def _rows(self, strands: list):
        """Rows across: the base under the first strand, the strands with
        the threads between them, the base under the last one. The heights
        of every row are a fixed linear mix of the strands (Catmull-Rom),
        so ``mix`` turns strand values into row values in one product."""
        k = len(strands)
        sub = lv._SUBDIV if k >= 2 else 1
        t = np.arange((k - 1) * sub + 1) / sub
        i0 = np.minimum(np.floor(t).astype(int), max(0, k - 2))
        f = t - i0
        mix = np.zeros((t.shape[0], k))
        if k == 1:
            mix[0, 0] = 1.0
        else:
            w = 0.5 * np.stack([
                -f + 2 * f ** 2 - f ** 3,
                2 - 5 * f ** 2 + 3 * f ** 3,
                f + 4 * f ** 2 - 3 * f ** 3,
                -f ** 2 + f ** 3,
            ], axis=1)
            for j, off in enumerate((-1, 0, 1, 2)):
                idx = np.clip(i0 + off, 0, k - 1)
                np.add.at(mix, (np.arange(t.shape[0]), idx), w[:, j])
        xs = np.array([s[3] for s in strands])
        cols = [s[2] for s in strands]
        if k == 1:
            row_x = xs.copy()
            row_rgb = np.array([_rgb(cols[0])])
            row_w = np.array([1.0])
        else:
            row_x = xs[i0] + (xs[i0 + 1] - xs[i0]) * f
            row_rgb = np.array([
                _rgb(lv._mix(cols[a], cols[min(a + 1, k - 1)], float(b)))
                for a, b in zip(i0, f)
            ])
            row_w = np.where(np.isclose(f, 0.0) | np.isclose(f, 1.0), 1.0,
                             lv._THREAD_WEIGHT)
        # The bases of the two side walls.
        row_x = np.concatenate([[row_x[0]], row_x, [row_x[-1]]])
        row_rgb = np.concatenate([row_rgb[:1], row_rgb, row_rgb[-1:]])
        row_w = np.concatenate([[-1.0], row_w, [-1.0]])
        mix = np.vstack([np.zeros((1, k)), mix, np.zeros((1, k))])
        strand_rows = 1 + np.arange(k) * sub
        is_strand = np.zeros(row_x.shape[0], dtype=bool)
        is_strand[strand_rows] = True
        mix_s = mix * is_strand[:, None]
        mix_t = mix * (~is_strand)[:, None]
        return mix_s, mix_t, row_x, row_rgb, row_w, strand_rows

    # ------------------------------------------------------------------ #
    # Camera                                                             #
    # ------------------------------------------------------------------ #

    def _amp(self) -> float:
        relief = lv._MESH_RELIEF if self.shape == "mesh" else 1.0
        return lv._HEIGHT * self.gain * relief

    def _camera(self, half: float, phase: float):
        """As the QPainter view: to one side, looking along the strands
        into the past; higher for the surface (a three-quarter view)."""
        sway = math.sin(phase)
        target = np.array([0.35 * half, -0.25 * self._amp(),
                           0.42 * lv._DEPTH])
        mesh = self.shape == "mesh"
        pos = np.array([
            -half - 3.2 + 1.4 * sway,
            (5.2 if mesh else 2.3) + 0.35 * math.sin(phase * 0.7 + 1.0),
            -6.0 if mesh else -4.6,
        ])
        f = target - pos
        f /= np.linalg.norm(f)
        r = np.cross([0.0, 1.0, 0.0], f)
        r /= np.linalg.norm(r)
        u = np.cross(f, r)
        return pos, r, u, f

    @staticmethod
    def _project(points: np.ndarray, cam):
        pos, r, u, f = cam
        v = points - pos
        zc = np.maximum(v @ f, 1e-2)
        return (v @ r) / zc, (v @ u) / zc

    def _fit_to(self, half: float, w: int, h: int):
        key = (w, h, round(half, 3), self.shape, round(self.gain, 3))
        if key == self._fit_key:
            return self._fit
        cam = self._camera(half, 0.0)
        amp = self._amp()
        low = -_BASE_DEPTH * amp if self.shape == "mesh" else -amp
        corners = np.array([
            [x, y, z]
            for x in (-half, half)
            for y in (low, amp)
            for z in (0.0, lv._DEPTH * 0.9)
        ])
        u, v = self._project(corners, cam)
        span_x, span_y = u.max() - u.min(), v.max() - v.min()
        scale = min(_FIT_MARGIN[0] * w / span_x, _FIT_MARGIN[1] * h / span_y)
        off_x = 0.52 * w - scale * (u.max() + u.min()) / 2
        off_y = 0.54 * h + scale * (v.max() + v.min()) / 2
        self._fit_key, self._fit = key, (scale, off_x, off_y)
        return self._fit

    @staticmethod
    def _mvp(cam, scale: float, off_x: float, off_y: float,
             w: int, h: int) -> QMatrix4x4:
        pos, r, u, f = cam
        view = QMatrix4x4(
            r[0], r[1], r[2], -float(r @ pos),
            u[0], u[1], u[2], -float(u @ pos),
            f[0], f[1], f[2], -float(f @ pos),
            0.0, 0.0, 0.0, 1.0,
        )
        near, far = 0.5, 400.0
        a = (far + near) / (far - near)
        b = -2.0 * far * near / (far - near)
        proj = QMatrix4x4(
            2.0 * scale / w, 0.0, 2.0 * off_x / w - 1.0, 0.0,
            0.0, 2.0 * scale / h, 1.0 - 2.0 * off_y / h, 0.0,
            0.0, 0.0, a, b,
            0.0, 0.0, 1.0, 0.0,
        )
        return proj * view

    # ------------------------------------------------------------------ #
    # GL                                                                 #
    # ------------------------------------------------------------------ #

    def init_gl(self) -> bool:
        program = _program()
        if program is None:
            self._gl = None
            return False
        self._gl = {"program": program, "vbo": None, "front": None,
                    "ibo": {}, "rows": 0}
        self._layout_key = None
        return True

    def release_gl(self) -> None:
        if not self._gl:
            return
        for buf in (self._gl["vbo"], self._gl["front"]):
            if buf is not None:
                buf.destroy()
        self._gl = None

    def _buffer(self, kind, data: np.ndarray, dynamic: bool = False):
        buf = QOpenGLBuffer(kind)
        buf.create()
        buf.setUsagePattern(QOpenGLBuffer.UsagePattern.DynamicDraw
                            if dynamic else
                            QOpenGLBuffer.UsagePattern.StaticDraw)
        buf.bind()
        raw = np.ascontiguousarray(data)
        buf.allocate(raw.tobytes(), raw.nbytes)
        return buf

    def _vertices(self, slots: np.ndarray, heights: np.ndarray) -> np.ndarray:
        """(columns, rows, 7) for the given slots and row heights."""
        lay = self._lay
        c, r = heights.shape
        out = np.empty((c, r, _FLOATS), dtype=np.float32)
        out[:, :, 0] = lay["x"][None, :]
        out[:, :, 1] = heights
        out[:, :, 2] = slots[:, None]
        out[:, :, 3] = lay["w"][None, :]
        out[:, :, 4:7] = lay["rgb"][None, :, :]
        return out

    def _heights(self, cols: np.ndarray) -> np.ndarray:
        """Row heights (len(cols), rows) at the absolute columns ``cols``:
        the strands and the threads, each from the strands smoothed in time
        as much as it is meant to be (centred, over the columns there
        are)."""
        lay = self._lay
        made = min(self.count, self.n)
        idx = cols[:, None] + self._offsets[None, :]
        ok = (idx >= self.count - made) & (idx <= self.count - 1)
        pos = np.where(ok, idx, cols[:, None]) % self.n
        weights = []
        for kernel in self._kernels:
            w = kernel[None, :] * ok
            weights.append(w / np.maximum(w.sum(axis=1, keepdims=True),
                                          1e-9))
        k = len(lay["strands"])
        sm = [np.empty((len(cols), k), dtype=np.float32) for _ in weights]
        for j, (kind, ch, _c, _x) in enumerate(lay["strands"]):
            v = self.values[kind][:, ch][pos]
            for out, w in zip(sm, weights):
                out[:, j] = (v * w).sum(axis=1)
        return (sm[0] @ lay["mix_s"].T + sm[1] @ lay["mix_t"].T).astype(
            np.float32)

    def make_layout(self, strands: list) -> None:
        """The rows for these strands (no GL: also for the tests)."""
        mix_s, mix_t, row_x, row_rgb, row_w, strand_rows = \
            self._rows(strands)
        self._lay = {"mix_s": mix_s, "mix_t": mix_t,
                     "x": row_x.astype(np.float32),
                     "rgb": row_rgb.astype(np.float32),
                     "w": row_w.astype(np.float32), "rows": row_x.shape[0],
                     "strands": strands, "strand_rows": strand_rows}

    def heights(self, cols) -> np.ndarray:
        """Row heights at absolute columns (after ``make_layout``)."""
        return self._heights(np.asarray(cols, dtype=np.int64))

    def _rebuild(self, strands: list) -> None:
        """New rows (channels, order, profile): every column again."""
        gl = self._gl
        for buf in (gl["vbo"], gl["front"]):
            if buf is not None:
                buf.destroy()
        self.make_layout(strands)
        rows, strand_rows = self._lay["rows"], self._lay["strand_rows"]
        n = self.n
        made = min(self.count, n)
        heights = np.zeros((2 * n, rows), dtype=np.float32)
        if made:
            cols = np.arange(self.count - made, self.count)
            vals = self._heights(cols)
            heights[cols % n] = vals
            heights[cols % n + n] = vals
        slots = np.arange(2 * n, dtype=np.float32)
        gl["vbo"] = self._buffer(QOpenGLBuffer.Type.VertexBuffer,
                                 self._vertices(slots, heights), True)
        front = 2 * rows + 2 * len(strand_rows) + 2
        gl["front"] = self._buffer(QOpenGLBuffer.Type.VertexBuffer,
                                   np.zeros((front, _FLOATS), np.float32),
                                   True)
        # Indices over the doubled ring, column by column, so the window of
        # the newest columns is always one contiguous slice. They stay in
        # memory and go with each draw: PySide6 cannot hand glDrawElements
        # an offset into a bound index buffer (it draws nothing).
        c = np.arange(2 * n - 1, dtype=np.uint32)[:, None]
        r = np.arange(rows - 1, dtype=np.uint32)[None, :]
        a = c * rows + r
        quads = np.stack([a, a + 1, a + rows, a + 1, a + rows + 1, a + rows],
                         axis=2)
        gl["ibo"] = {"fill": np.ascontiguousarray(quads.reshape(-1))}
        thread_rows = np.array([i for i in range(1, rows - 1)
                                if i not in set(strand_rows.tolist())],
                               dtype=np.uint32)
        for name, sel in (("strands", strand_rows.astype(np.uint32)),
                          ("threads", thread_rows),
                          ("base", np.array([0, rows - 1], np.uint32))):
            if sel.size == 0:
                continue
            seg = np.stack([c * rows + sel[None, :],
                            (c + 1) * rows + sel[None, :]], axis=2)
            gl["ibo"][name] = np.ascontiguousarray(seg.reshape(-1))
        gl["counts"] = {"fill": 6 * (rows - 1),
                        "strands": 2 * strand_rows.size,
                        "threads": 2 * thread_rows.size, "base": 4}
        gl["rows"] = rows
        self.pending = set()

    def _upload(self) -> None:
        """The columns made since the last frame, into both halves."""
        if not self.pending:
            return
        gl, n, rows = self._gl, self.n, self._lay["rows"]
        cols = np.array(sorted(j for j in self.pending
                               if j >= self.count - n), dtype=np.int64)
        self.pending = set()
        if cols.size == 0:
            return
        positions = cols % n
        heights = self._heights(cols)
        gl["vbo"].bind()
        stride = rows * _FLOATS * 4
        for half in (0, n):
            slots = (positions + half).astype(np.float32)
            data = self._vertices(slots, heights.astype(np.float32))
            # Contiguous runs of positions: one write each.
            start = 0
            for i in range(1, len(positions) + 1):
                if i == len(positions) or positions[i] != positions[i - 1] + 1:
                    block = np.ascontiguousarray(data[start:i])
                    gl["vbo"].write(int(positions[start] + half) * stride,
                                    block.tobytes(), block.nbytes)
                    start = i

    def render(self, f, w: int, h: int, now: float) -> None:
        """Draw the frame on the current context, ``w`` x ``h`` pixels."""
        bg = self.bg
        f.glViewport(0, 0, w, h)
        f.glClearColor(bg.redF(), bg.greenF(), bg.blueF(), 1.0)
        f.glClear(_GL_COLOR_BUFFER_BIT | _GL_DEPTH_BUFFER_BIT)
        self.labels = []
        if not self._gl:
            return
        strands = self.strands()
        if not strands:
            return
        key = (tuple((s[0], s[1], round(s[3], 4)) for s in strands),)
        if key != self._layout_key:
            self._rebuild(strands)
            self._layout_key = key
        self._upload()
        if self.count == 0:
            return
        gl, n, rows = self._gl, self.n, self._lay["rows"]
        half = max(abs(strands[0][3]), abs(strands[-1][3])) + lv._SPACING
        phase = 2.0 * math.pi * (now - self._t0) / lv._DRIFT_PERIOD_S \
            if self.drift else 0.0
        cam = self._camera(half, phase)
        scale, off_x, off_y = self._fit_to(half, w, h)
        mesh = self.shape == "mesh" and len(strands) >= 2
        amp = self._amp()
        base = -_BASE_DEPTH * amp if mesh else -1.0e6
        made = min(self.count, n)
        head = self.count % n          # first slot of the window
        newest = head + n - 1
        first = newest - (made - 1)    # oldest column with data

        p = gl["program"]
        p.bind()
        p.setUniformValue("u_mvp", self._mvp(cam, scale, off_x, off_y, w, h))
        p.setUniformValue1f("u_newest", float(newest))
        p.setUniformValue1f("u_span", float(n - 1))
        p.setUniformValue1f("u_frac", float(self.frac))
        p.setUniformValue1f("u_depth", float(lv._DEPTH))
        p.setUniformValue1f("u_amp", float(amp))
        p.setUniformValue1f("u_base", float(base))
        p.setUniformValue1f("u_sink", float(_SINK * amp))
        haze, grid = lv._HAZE, QColor(lv._GRID)
        p.setUniformValue("u_haze", haze.redF(), haze.greenF(), haze.blueF())
        p.setUniformValue("u_bg", bg.redF(), bg.greenF(), bg.blueF())
        p.setUniformValue("u_grid", grid.redF(), grid.greenF(), grid.blueF())

        def attrs(buf) -> None:
            buf.bind()
            p.enableAttributeArray(0)
            p.enableAttributeArray(1)
            p.setAttributeBuffer(0, _GL_FLOAT, 0, 4, _FLOATS * 4)
            p.setAttributeBuffer(1, _GL_FLOAT, 16, 3, _FLOATS * 4)

        def look(mode: float, alpha: float, power: float) -> None:
            p.setUniformValue1f("u_mode", mode)
            p.setUniformValue1f("u_alpha", alpha)
            p.setUniformValue1f("u_power", power)

        def indexed(mode: int, name: str, c0: int, c1: int) -> None:
            """The elements of columns c0 to c1 (slots)."""
            if name not in gl["ibo"] or c1 <= c0:
                return
            per = gl["counts"][name]
            f.glDrawElements(mode, (c1 - c0) * per, _GL_UNSIGNED_INT,
                             gl["ibo"][name][c0 * per:c1 * per])

        def along(name: str, c0: int, c1: int) -> None:
            """Segments along time between columns c0 and c1."""
            indexed(_GL_LINES, name, c0, c1)

        QOpenGLBuffer.release(QOpenGLBuffer.Type.IndexBuffer)
        attrs(gl["vbo"])
        f.glEnable(_GL_BLEND)
        f.glBlendFunc(_GL_SRC_ALPHA, _GL_ONE_MINUS_SRC_ALPHA)
        near = max(first, newest - n // 2)
        if mesh:
            # The solid: surface and side walls, then the front wall, in
            # the background colour, pushed back a little so the lines on
            # them pass the depth test and the lines behind them do not.
            f.glEnable(_GL_DEPTH_TEST)
            f.glDepthFunc(_GL_LESS)
            f.glDepthMask(_GL_TRUE)
            f.glEnable(_GL_POLYGON_OFFSET_FILL)
            f.glPolygonOffset(1.0, 1.0)
            look(0.0, 1.0, 1.0)
            indexed(_GL_TRIANGLES, "fill", first, newest)
            front = self._front(newest)
            gl["front"].bind()
            gl["front"].write(0, front.tobytes(), front.nbytes)
            attrs(gl["front"])
            f.glDrawArrays(_GL_TRIANGLE_STRIP, 0, 2 * rows)
            f.glDisable(_GL_POLYGON_OFFSET_FILL)
            f.glDepthFunc(_GL_LEQUAL)
            f.glDepthMask(_GL_FALSE)
            attrs(gl["vbo"])
            # The fabric between the strands.
            look(1.0, 0.95, 1.7)
            f.glLineWidth(1.0)
            along("threads", first, newest)
            # Cross-sections: across the surface and down both walls.
            look(2.0, 0.52, 2.2)
            every = max(1, round(_SECTION_S / self.col_s))
            j0 = self.count - made
            first_section = j0 + (-j0) % every
            for j in range(first_section, self.count - 1, every):
                slot = head + (j - (self.count - n))
                f.glDrawArrays(_GL_LINE_STRIP, slot * rows, rows)
            # The base of the side walls.
            look(2.0, 0.40, 2.2)
            along("base", first, newest)
            # The front wall: a drop under each strand and its base.
            attrs(gl["front"])
            k = len(self._lay["strand_rows"])
            look(2.0, 0.30, 1.0)
            f.glDrawArrays(_GL_LINES, 2 * rows, 2 * k)
            look(2.0, 0.45, 1.0)
            f.glDrawArrays(_GL_LINES, 2 * rows + 2 * k, 2)
            attrs(gl["vbo"])
            look(2.0, 0.83, 1.0)
            f.glLineWidth(1.4)
            f.glDrawArrays(_GL_LINE_STRIP, newest * rows, rows)
        else:
            f.glDisable(_GL_DEPTH_TEST)
            f.glDepthMask(_GL_FALSE)
        # The strands, heavier near the present.
        look(1.0, 0.95, 1.7)
        f.glLineWidth(1.0)
        along("strands", first, near)
        f.glLineWidth(1.6)
        along("strands", near, newest)
        f.glLineWidth(1.0)
        f.glDepthMask(_GL_TRUE)
        f.glDisable(_GL_DEPTH_TEST)
        f.glDisable(_GL_BLEND)
        p.disableAttributeArray(0)
        p.disableAttributeArray(1)
        p.release()
        QOpenGLBuffer.release(QOpenGLBuffer.Type.VertexBuffer)
        QOpenGLBuffer.release(QOpenGLBuffer.Type.IndexBuffer)
        self.labels = self._label_spots(strands, cam, scale, off_x, off_y,
                                        amp, base if mesh else None, h)

    def _front(self, newest: int) -> np.ndarray:
        """The front wall: each row's newest point and the base under it,
        alternating (a triangle strip), then a drop under each strand and
        the base line, as segments."""
        rows = self._lay["rows"]
        heights = self._heights(np.array([self.count - 1]))[0]
        top = self._vertices(np.array([float(newest)], np.float32),
                             heights[None, :].astype(np.float32))[0]
        bottom = top.copy()
        bottom[:, 3] = -1.0
        strip = np.empty((2 * rows, _FLOATS), dtype=np.float32)
        strip[0::2] = top
        strip[1::2] = bottom
        s = self._lay["strand_rows"]
        drops = np.empty((2 * len(s), _FLOATS), dtype=np.float32)
        drops[0::2] = top[s]
        drops[1::2] = bottom[s]
        base = bottom[[0, rows - 1]]
        return np.concatenate([strip, drops, base])

    def _label_spots(self, strands, cam, scale, off_x, off_y, amp, base,
                     h) -> list:
        """UOMO / CANE under the present of each group (pixels)."""
        if self.order == "interleaved":
            return []
        pos = (self.count - 1) % self.n
        out = []
        for kind in lv.KINDS:
            mine = [s for s in strands if s[0] == kind]
            if not mine:
                continue
            if base is not None:          # on the wall, near its foot
                pts = [[x, 0.72 * base, 0.0] for _k, _ch, _c, x in mine]
            else:
                pts = [[x, float(self.values[k][pos, ch]) * amp, 0.0]
                       for k, ch, _c, x in mine]
            u, v = self._project(np.array(pts), cam)
            sx = off_x + scale * u
            sy = off_y - scale * v
            y = float(sy.mean()) - 0.011 * h if base is not None \
                else float(sy.max()) + 0.03 * h
            out.append((lv.LABELS[kind], float(sx.mean()), y))
        return out


_GLBase = QOpenGLWidget if QOpenGLWidget is not None else QWidget


class LandscapeGLView(_GLBase):
    """The 3D view on the graphics card; same interface as
    ``LandscapeView``."""

    FRAME_MS = 16

    def __init__(self, profiles: dict[str, DeviceProfile],
                 seconds: float = 6.0, background: str = BG_DEEP,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)
        self._scene = _Scene(profiles, seconds, QColor(background))
        self._ready = False
        # Our own multisampled framebuffer: the widget's own one gets no
        # samples on some drivers (it got none here), and lines need them.
        self._msaa = None
        self._msaa_ok = True
        self.setMinimumHeight(180)

    # The settings the app changes live, kept by the scene.
    def _get(name):
        return property(lambda self: getattr(self._scene, name),
                        lambda self, v: setattr(self._scene, name, v))

    order = _get("order")
    shape = _get("shape")
    show_labels = _get("show_labels")
    gain = _get("gain")
    drift = _get("drift")
    del _get

    def push(self, kind: str, packet: UnicornPacket) -> None:
        self._scene.push(kind, packet, time.perf_counter())

    def advance(self) -> None:
        """Ask for a frame; the data move on when it is drawn, so the
        time axis is where the clock says at that very moment."""
        self.update()

    def set_profile(self, kind: str, profile: DeviceProfile) -> None:
        old = self._scene.feeds.get(kind)
        if old is None:
            return
        visible = old.visible
        scene = self._scene
        feed = _Feed(profile, scene.total)
        scene.feeds[kind] = feed
        scene.values[kind] = np.zeros((scene.n, feed.n_ch), dtype=np.float32)
        scene._sums[kind] = np.zeros(feed.n_ch)
        self.set_visible_channels(kind, visible)

    def set_visible_channels(self, kind: str, indices) -> None:
        feed = self._scene.feeds.get(kind)
        if feed is None:
            return
        feed.visible = sorted({int(i) for i in indices
                               if 0 <= i < feed.n_ch})
        self.update()

    def visible_channels(self, kind: str) -> list[int]:
        return list(self._scene.feeds[kind].visible)

    def initializeGL(self) -> None:  # noqa: N802 - Qt name
        self._ready = self._scene.init_gl()
        ctx = self.context()
        if ctx is not None:
            ctx.aboutToBeDestroyed.connect(self._release)

    def _release(self) -> None:
        self.makeCurrent()
        self._scene.release_gl()
        self._msaa = None
        self.doneCurrent()

    def _sampled(self, w: int, h: int):
        """The multisampled framebuffer at this size (None: render
        straight into the widget's, without antialiasing)."""
        if not self._msaa_ok:
            return None
        if self._msaa is not None and self._msaa.width() == w \
                and self._msaa.height() == h:
            return self._msaa
        fmt = QOpenGLFramebufferObjectFormat()
        fmt.setSamples(_SAMPLES)
        fmt.setAttachment(QOpenGLFramebufferObject.Attachment.Depth)
        fbo = QOpenGLFramebufferObject(w, h, fmt)
        if not fbo.isValid():
            self._msaa_ok = False
            return None
        self._msaa = fbo
        return fbo

    def paintGL(self) -> None:  # noqa: N802 - Qt name
        dpr = self.devicePixelRatioF()
        w = max(1, int(round(self.width() * dpr)))
        h = max(1, int(round(self.height() * dpr)))
        now = time.perf_counter()
        self._scene.advance(now)
        ctx = self.context()
        f = ctx.functions()
        target = self.defaultFramebufferObject()
        fbo = self._sampled(w, h)
        if fbo is not None:
            fbo.bind()
            self._scene.render(f, w, h, now)
            ex = ctx.extraFunctions()
            ex.glBindFramebuffer(_GL_READ_FRAMEBUFFER, fbo.handle())
            ex.glBindFramebuffer(_GL_DRAW_FRAMEBUFFER, target)
            ex.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h,
                                 _GL_COLOR_BUFFER_BIT, _GL_NEAREST)
            f.glBindFramebuffer(_GL_FRAMEBUFFER, target)
        else:
            self._scene.render(f, w, h, now)
        if self.show_labels and self._scene.labels:
            p = QPainter(self)
            draw_labels(p, self.font(), self._scene.labels, dpr)
            p.end()


def draw_labels(p: QPainter, base_font: QFont, labels, dpr: float) -> None:
    font = QFont(base_font)
    font.setPointSizeF(8.0)
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 130)
    p.setFont(font)
    c = QColor(TEXT_DIM)
    c.setAlphaF(0.85)
    p.setPen(c)
    for text, x, y in labels:
        p.drawText(QRectF(x / dpr - 60, y / dpr, 120, 20),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                   text)


def make_landscape_view(profiles: dict[str, DeviceProfile],
                        seconds: float = 6.0, background: str = BG_DEEP,
                        parent: QWidget | None = None):
    """The 3D view on the graphics card where OpenGL works, otherwise the
    QPainter one."""
    if gl_available():
        return LandscapeGLView(profiles, seconds=seconds,
                               background=background, parent=parent)
    return lv.LandscapeView(profiles, seconds=seconds, background=background,
                            parent=parent)


__all__ = ["LandscapeGLView", "gl_available", "make_landscape_view"]
