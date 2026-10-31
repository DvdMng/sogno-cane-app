"""Both EEGs in one perspective: time runs into the distance.

Every visible channel of the human and of the dog is a strand. The newest
samples are near the viewer, the older ones recede toward a vanishing
point off to one side and dissolve as they go; the point of view drifts
slowly, so the depth reads. The strands are drawn a little soft (at
reduced resolution, then eased up) and each frame is drawn afresh: no film
grain, no motion trails (each frame fading over the last), no sparkle,
which all read as cheap effects. Colours as
in the trace view, pink and cyan from one channel to the next; going back
in time every strand turns toward the tone halfway between the two and
dissolves.

Plain QPainter with numpy projection. Since 0.9.8 the app draws the scene
on the graphics card (``landscape_gl``); this view is what it falls back
to where OpenGL is missing, and what the offscreen tests see.
"""
from __future__ import annotations

import math
import time

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget
from pyqtgraph.functions import arrayToQPath

from sogno_cane.eeg.profiles import DeviceProfile
from sogno_cane.eeg.unicorn_packet import UnicornPacket
from sogno_cane.ui.theme import ACCENT_CYAN, ACCENT_PINK, BG_DEEP, TEXT_DIM
from sogno_cane.ui.waveform_widget import _DisplayCleaner, _Playout, _palette

KINDS = ("human", "dog")
LABELS = {"human": "UOMO", "dog": "CANE"}
ORDERS = ("grouped", "interleaved")
SHAPES = ("lines", "mesh")


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    return QColor(
        round(a.red() + (b.red() - a.red()) * t),
        round(a.green() + (b.green() - a.green()) * t),
        round(a.blue() + (b.blue() - a.blue()) * t),
    )


# Where pink and cyan meet: the past of every strand, human or dog, fades
# through this shared tone before it dissolves.
_HAZE = _mix(QColor(ACCENT_PINK), QColor(ACCENT_CYAN), 0.5)

# Scene, in world units: strands lie side by side _SPACING apart on the
# ground, oscillate up to _HEIGHT, and run _DEPTH back into the distance.
_SPACING = 0.72
_GROUP_GAP = 1.1
_HEIGHT = 1.05
_DEPTH = 26.0
_POINTS = 375                # points per strand at most
_BANDS = 9                   # alpha steps from present to past
# The mesh: the surface joining the strands.
_SUBDIV = 6                  # rows per gap: the strand and 5 threads
_THREAD_WEIGHT = 0.40        # the threads: thin, well under the strands
_THREAD_STEP = 2             # threads at half the detail of the strands
_MESH_STEP = 3               # strip length along time, in points (~50 ms)
_MESH_RELIEF = 1.5           # seen from above the relief foreshortens: taller
_MESH_SECTION = 3            # a cross-section every so many strips (~150 ms)
_SURFACE_BANDS = 12          # steps in which the strands go on top
_GRID = "#B9B0CC"            # the cross-sections: a neutral, light grey-lilac
# Light smoothing along time for the surface (Gaussian, sigma ~20 ms at the
# default 16 ms per point): alpha keeps its shape, the bristle goes.
_SMOOTH_KERNEL = np.exp(-0.5 * (np.arange(-3, 4) / 1.25) ** 2)
_SMOOTH_KERNEL /= _SMOOTH_KERNEL.sum()
# The strands are drawn at reduced resolution and eased up, which softens
# them: about 700 lines tall whatever the window (half resolution on a
# projector, full in the app, where the view is low and halving it blurred
# the strands into ribbons).
_RENDER_LINES = 700
_MESH_LINES = 560            # the surface is soft anyway, and costs more
# Frames per second for this view: the scene moves slowly, and every frame
# drawn is time taken from the MIDI engine (same interpreter).
FRAME_MS = 33
_DRIFT_PERIOD_S = 70.0       # slow drift of the point of view
# The point of view, moved with the mouse: drag to turn the scene about its
# middle, right button (or Shift) to move it, wheel to come closer to the
# point under the pointer, double-click for the framed view again. What the
# mouse asks is reached smoothly, frame by frame, and a flick keeps turning
# for a moment.
NAVIGATION_HINT = ("trascina: ruota  ·  tasto destro: sposta  ·  "
                   "rotella: avvicina dove punti  ·  "
                   "doppio clic: inquadratura iniziale")
_TURN = 0.006                # radians per pixel dragged
_ZOOM_STEP = 1.15            # closer by this much per wheel notch
_DOLLY_RANGE = (0.18, 2.5)   # distance to the pivot, times the framed one
_SHIFT_LIMIT = 30.0          # world units the pivot may be moved, each way
_PITCH_RANGE = (math.radians(4.0), math.radians(85.0))
# Seconds to cover two thirds of the way to what the mouse asked, for yaw,
# pitch, distance and the three components of the move.
_EASE_S = np.array([0.09, 0.09, 0.13, 0.07, 0.07, 0.07])
_SPIN_DECAY_S = 0.3          # a flick slows down over this
_SPIN_MIN = 0.5              # rad/s: a slower release simply stops
_SPIN_MAX = 5.0              # rad/s


def look_at(eye: np.ndarray, target: np.ndarray) -> tuple:
    """(eye, right, up, forward): from ``eye`` toward ``target``, level."""
    f = np.asarray(target, dtype=float) - eye
    f = f / np.linalg.norm(f)
    r = np.cross([0.0, 1.0, 0.0], f)
    n = float(np.linalg.norm(r))
    r = r / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    return eye, r, np.cross(f, r), f


def project(points: np.ndarray, cam) -> tuple[np.ndarray, np.ndarray]:
    """Camera-plane coordinates of ``points`` (n, 3)."""
    eye, r, u, f = cam
    v = points - eye
    zc = np.maximum(v @ f, 1e-2)
    return (v @ r) / zc, (v @ u) / zc


def fit_frame(corners: np.ndarray, cam, w: int, h: int,
              fill: tuple) -> tuple:
    """Pixels per unit and offsets so ``corners`` fill the frame (``fill``
    shares of it), as the framed view has always placed them."""
    u, v = project(corners, cam)
    scale = min(fill[0] * w / (u.max() - u.min()),
                fill[1] * h / (v.max() - v.min()))
    return (scale, 0.52 * w - scale * (u.max() + u.min()) / 2,
            0.56 * h + scale * (v.max() + v.min()) / 2)


def pivot_of(look: np.ndarray) -> np.ndarray:
    """What the scene turns about: its middle across, on the surface, at
    the depth the framed view looks at. Turning, it stays in place."""
    return np.array([0.0, 0.0, float(look[2])])


class Navigation:
    """Where the user has moved the point of view: a camera rig turned
    about a pivot in the middle of the scene (yaw, pitch, radians), at a
    distance from it (dolly, times the framed one), the pivot moved in the
    scene (shift, world units). The mouse sets where to go; ``step``
    brings what is drawn there smoothly at every frame, whatever the rate
    of the mouse events, and lets a flick turn on for a moment. The
    defaults give the framed view. STUDIO and the projection share one:
    each frame steps it by the clock, so two views stepping it in the same
    frame move it once."""

    _FIELDS = ("yaw", "pitch", "dolly", "shift_x", "shift_y", "shift_z")
    _DEFAULT = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def __init__(self) -> None:
        self._goal = self._DEFAULT.copy()
        self._shown = self._DEFAULT.copy()
        self._spin = np.zeros(2)            # rad/s, yaw and pitch
        self._swipe = np.zeros(2)           # drag speed, for a flick
        self._moved: float | None = None    # last turn of the drag
        self._stamp: float | None = None    # last step

    # What is drawn.
    yaw = property(lambda self: float(self._shown[0]))
    pitch = property(lambda self: float(self._shown[1]))
    dolly = property(lambda self: float(self._shown[2]))
    shift = property(lambda self: self._shown[3:].copy())

    def goal(self) -> dict:
        """Where the mouse has asked to go."""
        return dict(zip(self._FIELDS, (float(v) for v in self._goal)))

    @property
    def moving(self) -> bool:
        return bool(self._spin.any()) or not np.allclose(
            self._shown, self._goal, atol=1e-4)

    def reset(self) -> None:
        """Back to the framed view, smoothly and the short way round."""
        self.hold()
        goal = self._DEFAULT.copy()
        goal[0] = self._shown[0] - math.remainder(self._shown[0],
                                                  2.0 * math.pi)
        self._goal = goal

    def settle(self) -> None:
        """Be where asked at once (loading a saved view, tests)."""
        self.hold()
        self._shown = self._goal.copy()

    def to_dict(self) -> dict:
        out = self.goal()
        out["yaw"] = math.remainder(out["yaw"], 2.0 * math.pi)
        return out

    @classmethod
    def from_dict(cls, data) -> "Navigation":
        nav = cls()
        if isinstance(data, dict):
            goal = nav._goal
            for i, name in enumerate(cls._FIELDS):
                value = _finite(data.get(name))
                if value is not None:
                    goal[i] = value
            zoom = _finite(data.get("zoom"))
            if "dolly" not in data and zoom and zoom > 0.0:
                goal[2] = 1.0 / zoom        # 0.9.10-0.9.12: a lens zoom
            goal[0] = math.remainder(goal[0], 2.0 * math.pi)
            goal[1] = min(math.pi / 2, max(-math.pi / 2, goal[1]))
            goal[2] = min(_DOLLY_RANGE[1], max(_DOLLY_RANGE[0], goal[2]))
            goal[3:] = np.clip(goal[3:], -_SHIFT_LIMIT, _SHIFT_LIMIT)
        nav.settle()
        return nav

    # ------------------------------------------------------------------ #
    # What the mouse asks                                                #
    # ------------------------------------------------------------------ #

    def turn(self, dx: float, dy: float, now: float | None = None) -> None:
        """As if the scene were grabbed: right turns it right, down tips
        its far side toward the viewer. ``now`` (a drag) keeps track of the
        speed, for a flick."""
        step = np.array([-dx * _TURN, dy * _TURN])
        self._goal[:2] += step
        if now is None:
            return
        if self._moved is not None and now > self._moved:
            dt = now - self._moved
            self._swipe += min(1.0, dt / 0.05) * (step / dt - self._swipe)
        self._moved = now

    def let_go(self, now: float) -> None:
        """The end of a drag: if it was still moving fast, it turns on."""
        speed = float(np.hypot(*self._swipe))
        if self._moved is not None and now - self._moved < 0.06 \
                and speed > _SPIN_MIN:
            self._spin = self._swipe * min(1.0, _SPIN_MAX / speed)
        self._swipe = np.zeros(2)
        self._moved = None

    def hold(self) -> None:
        """Stop a flick: the scene grabbed again."""
        self._spin = np.zeros(2)
        self._swipe = np.zeros(2)
        self._moved = None

    def slide(self, vector) -> None:
        """Move the scene by ``vector``, in world units."""
        self._goal[3:] = np.clip(self._goal[3:] + vector,
                                 -_SHIFT_LIMIT, _SHIFT_LIMIT)

    def zoom(self, notches: float, toward=None, pivot=None) -> None:
        """Closer (``notches`` > 0) or farther; toward the world point
        ``toward`` if given, which then stays under the pointer (``pivot``:
        the scene's own, unmoved)."""
        old = float(self._goal[2])
        new = min(_DOLLY_RANGE[1],
                  max(_DOLLY_RANGE[0], old / _ZOOM_STEP ** notches))
        self._goal[2] = new
        if toward is not None and pivot is not None:
            k = new / old
            self._goal[3:] = np.clip(
                k * self._goal[3:] + (1.0 - k) * (np.asarray(toward)
                                                  - np.asarray(pivot)),
                -_SHIFT_LIMIT, _SHIFT_LIMIT)

    # ------------------------------------------------------------------ #
    # What is drawn                                                      #
    # ------------------------------------------------------------------ #

    def step(self, now: float) -> None:
        """Bring what is drawn toward the goal, by the time passed."""
        if self._stamp is None:
            self._stamp = now
        dt = min(0.1, max(0.0, now - self._stamp))
        self._stamp = now
        if dt <= 0.0:
            return
        if self._spin.any():
            self._goal[:2] += self._spin * dt
            self._spin *= math.exp(-dt / _SPIN_DECAY_S)
            if float(np.hypot(*self._spin)) < 0.05:
                self._spin = np.zeros(2)
        self._shown += (1.0 - np.exp(-dt / _EASE_S)) * (self._goal
                                                         - self._shown)
        turns = round(self._shown[0] / (2.0 * math.pi))
        if turns:                    # both near zero, the same turns off
            self._goal[0] -= turns * 2.0 * math.pi
            self._shown[0] -= turns * 2.0 * math.pi

    def rig(self, eye: np.ndarray, look: np.ndarray,
            pivot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The eye and the point it looks at, as navigated: the framed ones
        turned about ``pivot``, brought closer to it, moved with it. The
        eye stays 4 to 85 degrees above the surface (the pitch is held
        there too, so dragging back answers at once)."""
        v0, w0 = eye - pivot, look - pivot
        flat = math.hypot(v0[0], v0[2])
        el0 = math.atan2(v0[1], flat)
        lo, hi = _PITCH_RANGE[0] - el0, _PITCH_RANGE[1] - el0
        for state in (self._goal, self._shown):
            state[1] = min(hi, max(lo, state[1]))
        if self._goal[1] in (lo, hi):
            self._spin[1] = 0.0
        yaw, pitch, dolly = self._shown[:3]
        axis = np.array([-v0[2], 0.0, v0[0]]) / max(flat, 1e-9)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        def turned(v: np.ndarray) -> np.ndarray:
            # Up about the line across the view, then round the vertical.
            v = v * cp + np.cross(axis, v) * sp + axis * (axis @ v) * (1 - cp)
            return np.array([v[0] * cy - v[2] * sy, v[1],
                             v[0] * sy + v[2] * cy])

        base = pivot + self._shown[3:]
        return base + dolly * turned(v0), base + dolly * turned(w0)


def _finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def navigated(nav: Navigation, eye: np.ndarray, look: np.ndarray,
              pivot: np.ndarray, fit: tuple) -> tuple:
    """The camera as navigated, and the scale and offsets to draw it with:
    the framed view's, moved so the pivot stays where the framed view shows
    it. Turning about it, the scene stays in place and keeps its size."""
    cam = look_at(*nav.rig(eye, look, pivot))
    scale, off_x, off_y = fit
    p = np.asarray(pivot, dtype=float)[None, :]
    u0, v0 = project(p, look_at(eye, look))
    u1, v1 = project(p + nav.shift, cam)
    return (cam, scale, off_x + scale * float(u0[0] - u1[0]),
            off_y - scale * float(v0[0] - v1[0]))


class NavFrame:
    """The last frame drawn, as the mouse needs it: its camera, scale and
    offsets (in drawing pixels, ``px`` per widget pixel), the pivot, and
    where the waves lie on the surface (x from, x to, depth from, to)."""

    def __init__(self, cam, scale: float, off_x: float, off_y: float,
                 pivot: np.ndarray, shift: np.ndarray, px: float,
                 bounds: tuple = (-np.inf, np.inf, -np.inf, np.inf)) -> None:
        self.cam, self.scale = cam, scale
        self.off_x, self.off_y = off_x, off_y
        self.pivot, self.shift, self.px = pivot, shift, px
        self.bounds = bounds

    def slide(self, dx: float, dy: float) -> np.ndarray:
        """The move (world units) that keeps the scene under the pointer
        dragged by ``dx``, ``dy`` widget pixels."""
        eye, r, u, f = self.cam
        depth = max(0.5, float((self.pivot + self.shift - eye) @ f))
        return (-dx * r + dy * u) * (depth * self.px / self.scale)

    def ground_at(self, x: float, y: float) -> np.ndarray | None:
        """The point of the surface under widget pixel (``x``, ``y``), or
        beside the waves the nearest point of them (coming closer to empty
        ground would lose the scene); None above the horizon."""
        eye, r, u, f = self.cam
        ray = f + (x * self.px - self.off_x) / self.scale * r \
            + (self.off_y - y * self.px) / self.scale * u
        if ray[1] > -1e-6:
            return None
        hit = eye + ray * (-eye[1] / ray[1])
        x0, x1, z0, z1 = self.bounds
        hit[0] = min(x1, max(x0, hit[0]))
        hit[2] = min(z1, max(z0, hit[2]))
        return hit if np.all(np.isfinite(hit)) else None


def nav_press(view, event) -> None:
    view._drag = (event.button(), event.modifiers(), event.position())
    view.navigation.hold()
    view.setCursor(Qt.CursorShape.ClosedHandCursor)
    event.accept()


def nav_move(view, event) -> None:
    if view._drag is None:
        return
    button, mods, last = view._drag
    pos = event.position()
    dx, dy = pos.x() - last.x(), pos.y() - last.y()
    view._drag = (button, mods, pos)
    if button == Qt.MouseButton.LeftButton \
            and not mods & Qt.KeyboardModifier.ShiftModifier:
        view.navigation.turn(dx, dy, time.perf_counter())
    elif view.nav_frame is not None:
        view.navigation.slide(view.nav_frame.slide(dx, dy))
    view.update()
    event.accept()


def nav_release(view, event) -> None:
    if view._drag is not None:
        view.navigation.let_go(time.perf_counter())
    view._drag = None
    view.unsetCursor()
    event.accept()


def nav_wheel(view, event) -> None:
    frame = view.nav_frame
    toward = None
    if frame is not None:
        pos = event.position()
        toward = frame.ground_at(pos.x(), pos.y())
    view.navigation.zoom(event.angleDelta().y() / 120.0, toward,
                         frame.pivot if frame is not None else None)
    view.update()
    event.accept()


def nav_double_click(view, event) -> None:
    """The framed view again, smoothly; where double-click means something
    else (it closes the projection), the click goes on to the window."""
    if not view.reset_on_double_click:
        event.ignore()
        return
    view.navigation.reset()
    view.update()
    event.accept()


class ViewStyle:
    """How the 3D view looks, as the user sets it (REGOLAZIONI): the
    contrast of the lines, their thickness, the seconds of EEG in view, the
    height of the waves, how much of the fabric shows. STUDIO and the
    projection share one, like the point of view."""

    RANGES = {
        "contrast": (0.4, 2.5),
        "thickness": (1.0, 3.0),
        "seconds": (2.0, 20.0),
        "gain": (0.3, 3.0),
        "fabric": (0.0, 2.0),
    }
    DEFAULTS = {"contrast": 1.0, "thickness": 1.0, "seconds": 6.0,
                "gain": 1.0, "fabric": 1.0}

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        for name, value in self.DEFAULTS.items():
            setattr(self, name, value)

    def set(self, name: str, value) -> None:
        lo, hi = self.RANGES[name]
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(value):
            setattr(self, name, min(hi, max(lo, value)))

    def to_dict(self) -> dict:
        return {name: float(getattr(self, name)) for name in self.DEFAULTS}

    @classmethod
    def from_dict(cls, data) -> "ViewStyle":
        style = cls()
        if isinstance(data, dict):
            for name in cls.DEFAULTS:
                if name in data:
                    style.set(name, data[name])
        return style


class _Stream:
    """One being's EEG: cleaned, paced like the trace view, last N s kept."""

    def __init__(self, profile: DeviceProfile, seconds: float) -> None:
        self.profile = profile
        self.fs = float(profile.sample_rate_hz)
        n = max(16, int(round(seconds * self.fs)))
        self.step = max(1, int(math.ceil(n / _POINTS)))
        self.n = (n // self.step) * self.step
        n_ch = len(profile.channel_names)
        self.visible = list(range(n_ch))
        self._cleaner = _DisplayCleaner(n_ch, self.fs)
        self._playout = _Playout(self.fs)
        self._ring = np.zeros((2 * self.n, n_ch), dtype=np.float32)
        self._head = 0
        self.scale = 0.0             # typical amplitude (uV), smoothed
        self.received = 0

    def push(self, packet: UnicornPacket, now: float) -> None:
        self._playout.push(self._cleaner.process(packet.eeg), now)

    def resized(self, seconds: float) -> "_Stream":
        """The same stream with another length in view: cleaning, pacing
        and scale go on, the newest samples stay."""
        fresh = _Stream(self.profile, seconds)
        fresh.visible = self.visible
        fresh._cleaner, fresh._playout = self._cleaner, self._playout
        fresh.scale, fresh.received = self.scale, self.received
        k = min(self.n, fresh.n, self.received)
        if k:
            newest = self._ring[self._head + self.n - k:self._head + self.n]
            for base in (0, fresh.n):           # the window ends with them
                fresh._ring[base + fresh.n - k:base + fresh.n] = newest
        return fresh

    def advance(self, now: float) -> bool:
        rows = self._playout.take(now)
        if rows is None:
            return False
        rows = rows[-self.n:]
        k = rows.shape[0]
        i = self._head
        first = min(k, self.n - i)
        for base in (0, self.n):
            self._ring[base + i:base + i + first] = rows[:first]
            if k > first:
                self._ring[base:base + k - first] = rows[first:]
        self._head = (i + k) % self.n
        self.received += k
        return True

    def strands(self, dt: float) -> np.ndarray:
        """(visible channels, points) heights in -1..1, NEWEST first."""
        points = self.n // self.step
        if not self.visible:
            return np.zeros((0, points), dtype=np.float32)
        window = self._ring[self._head:self._head + self.n, self.visible]
        values = window.reshape(points, self.step, -1).mean(axis=1).T
        # One scale per being, from the recent spread of all its visible
        # channels: relative sizes between channels stay true, and an
        # artefact is softly limited (tanh) instead of cut flat.
        spread = float(np.percentile(np.abs(values), 90)) if self.received \
            else 0.0
        if spread > 0.0:
            a = 1.0 if self.scale <= 0.0 else min(1.0, dt / 2.0)
            self.scale += a * (spread - self.scale)
        s = max(self.scale, 1e-3)
        return np.tanh(values / (2.4 * s))[:, ::-1]


class LandscapeView(QWidget):
    """Every visible channel of both beings, receding into the distance."""

    FRAME_MS = FRAME_MS

    def __init__(
        self,
        profiles: dict[str, DeviceProfile],
        seconds: float = 6.0,
        background: str = BG_DEEP,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._streams = {k: _Stream(p, seconds) for k, p in profiles.items()}
        self._seconds = float(seconds)
        self.style = ViewStyle()
        self.style.set("seconds", seconds)
        self.order = "grouped"
        self.shape = "lines"         # or "mesh": neighbouring strands joined
        self.show_labels = True
        self.drift = True
        self.navigation = Navigation()
        self.nav_frame: NavFrame | None = None     # the last frame drawn
        self.reset_on_double_click = True
        self._drag = None
        self._bg = QColor(background)
        self._accum: QImage | None = None
        self._t0 = time.perf_counter()
        self._last = self._t0
        self._fit_key = None
        self._fit = None             # (scale, off_x, off_y), pivot
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMinimumHeight(180)

    # ------------------------------------------------------------------ #
    # Feeding                                                            #
    # ------------------------------------------------------------------ #

    @property
    def gain(self) -> float:
        return self.style.gain

    @gain.setter
    def gain(self, value: float) -> None:
        self.style.set("gain", value)

    def push(self, kind: str, packet: UnicornPacket) -> None:
        stream = self._streams.get(kind)
        if stream is not None:
            stream.push(packet, time.perf_counter())

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt name
        nav_press(self, event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt name
        nav_move(self, event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt name
        nav_release(self, event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        nav_double_click(self, event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt name
        nav_wheel(self, event)

    def set_profile(self, kind: str, profile: DeviceProfile) -> None:
        """Another headset on that side: start its strands afresh."""
        old = self._streams.get(kind)
        if old is None:
            return
        visible = old.visible
        self._streams[kind] = _Stream(profile, self._seconds)
        self.set_visible_channels(kind, visible)

    def set_visible_channels(self, kind: str, indices) -> None:
        stream = self._streams.get(kind)
        if stream is None:
            return
        n_ch = len(stream.profile.channel_names)
        stream.visible = sorted({int(i) for i in indices if 0 <= i < n_ch})
        self._fit_key = None
        self.update()

    def visible_channels(self, kind: str) -> list[int]:
        return list(self._streams[kind].visible)

    def _follow_style(self) -> None:
        """Another field in view: the strands go on at that length with
        what they hold."""
        if abs(self.style.seconds - self._seconds) < 1e-6:
            return
        self._seconds = float(self.style.seconds)
        for kind, old in list(self._streams.items()):
            self._streams[kind] = old.resized(self._seconds)
        self._fit_key = None

    def advance(self) -> None:
        """Reveal what arrived since the last frame and repaint."""
        self._follow_style()
        now = time.perf_counter()
        for s in self._streams.values():
            s.advance(now)
        self.update()            # the drift and the fading move anyway

    # ------------------------------------------------------------------ #
    # Scene                                                              #
    # ------------------------------------------------------------------ #

    def _layout(self, strands: dict[str, np.ndarray]) -> list[tuple]:
        """(kind, colour, heights, x position), left to right."""
        per = {}
        for k in self._streams:
            visible = self._streams[k].visible
            colors = _palette(len(visible))     # as in the trace view
            per[k] = [(k, QColor(colors[i]), strands[k][i])
                      for i in range(len(visible))]
        human, dog = per.get("human", []), per.get("dog", [])
        if self.order == "interleaved":
            ordered = []
            for i in range(max(len(human), len(dog))):
                ordered += human[i:i + 1] + dog[i:i + 1]
            xs = [i * _SPACING for i in range(len(ordered))]
        else:
            ordered = human + dog
            xs = [i * _SPACING for i in range(len(human))]
            start = (len(human) - 1) * _SPACING + _GROUP_GAP if human else 0
            xs += [start + i * _SPACING for i in range(len(dog))]
        if not ordered:
            return []
        mid = (xs[0] + xs[-1]) / 2.0
        return [r + (x - mid,) for r, x in zip(ordered, xs)]

    def _eye(self, half_width: float, phase: float):
        """The framed eye and the point it looks at: low and to one side,
        looking along the strands into the past; higher for the surface (a
        three-quarter view from above), where a low eye made the peaks
        overlap and the grid unreadable."""
        sway = math.sin(phase)
        look = np.array([0.35 * half_width, 0.0, 0.42 * _DEPTH])
        mesh = self.shape == "mesh"
        eye = np.array([
            -half_width - 3.2 + 1.4 * sway,
            (5.2 if mesh else 2.3) + 0.35 * math.sin(phase * 0.7 + 1.0),
            -6.0 if mesh else -4.6,
        ])
        return eye, look

    _project = staticmethod(project)

    def _fit_to(self, half_width: float, w: int, h: int):
        """The framed view's scale and offsets, so the scene fills the
        frame, and the pivot it turns about. The same however the view is
        turned: turning, the scene keeps its size."""
        key = (w, h, round(half_width, 3), self.shape)
        if key == self._fit_key:
            return self._fit
        eye, look = self._eye(half_width, 0.0)
        mesh = self.shape == "mesh"
        tall = _HEIGHT * (_MESH_RELIEF if mesh else 1.0)
        corners = np.array([
            [x, y, z]
            for x in (-half_width, half_width)
            for y in (-tall, tall)
            for z in (0.0, _DEPTH * (0.9 if mesh else 0.7))
        ])
        # The surface stays inside the frame, with a margin all round.
        fill = (0.90, 0.86) if mesh else (1.0, 0.96)
        self._fit_key = key
        self._fit = (fit_frame(corners, look_at(eye, look), w, h, fill),
                     pivot_of(look))
        return self._fit

    # ------------------------------------------------------------------ #
    # Painting                                                           #
    # ------------------------------------------------------------------ #

    def _render_scale(self) -> float:
        lines = _RENDER_LINES if self.shape != "mesh" else _MESH_LINES
        return min(1.0, max(0.5, lines / max(1, self.height())))

    def _ensure_accum(self) -> QImage:
        k = self._render_scale()
        w = max(1, int(self.width() * k))
        h = max(1, int(self.height() * k))
        if self._accum is None or self._accum.width() != w \
                or self._accum.height() != h:
            self._accum = QImage(w, h,
                                 QImage.Format.Format_ARGB32_Premultiplied)
            self._accum.fill(self._bg)
        return self._accum

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt name
        now = time.perf_counter()
        dt, self._last = min(0.25, now - self._last), now
        accum = self._ensure_accum()
        labels = self._draw_strands(accum, now, dt)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawImage(self.rect(), accum)
        if self.show_labels:
            self._draw_labels(p, labels)
        p.end()

    def _draw_strands(self, img: QImage, now: float, dt: float) -> list:
        """Draw into ``img``; return where the labels go (widget pixels)."""
        strands = {k: s.strands(dt) for k, s in self._streams.items()}
        rows = self._layout(strands)
        p = QPainter(img)
        p.fillRect(img.rect(), self._bg)
        if not rows:
            p.end()
            return []
        half = max(abs(rows[0][3]), abs(rows[-1][3])) + _SPACING
        phase = 2.0 * math.pi * (now - self._t0) / _DRIFT_PERIOD_S \
            if self.drift else 0.0
        self.navigation.step(now)
        eye, look = self._eye(half, phase)
        w, h = img.width(), img.height()
        fit, pivot = self._fit_to(half, w, h)
        cam, scale, off_x, off_y = navigated(self.navigation, eye, look,
                                             pivot, fit)
        self.nav_frame = NavFrame(cam, scale, off_x, off_y, pivot,
                                  self.navigation.shift,
                                  self._render_scale(),
                                  (-half, half, 0.0, _DEPTH))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        fronts: dict[str, list[tuple[float, float]]] = {}
        drawn, world = [], []
        for kind, base, values, x in rows:
            k = values.shape[0]
            zs = np.linspace(0.0, _DEPTH, k)
            heights = _HEIGHT * self.gain * values * (
                _MESH_RELIEF if self.shape == "mesh" else 1.0)
            pts = np.column_stack([np.full(k, x), heights, zs])
            u, v = self._project(pts, cam)
            sx = off_x + scale * u
            sy = off_y - scale * v
            fronts.setdefault(kind, []).append((float(sx[0]), float(sy[0])))
            drawn.append((base, sx, sy))
            world.append((base, x, heights))
        if self.shape == "mesh" and len(world) > 1:
            self._paint_surface(p, world, drawn, cam, scale, off_x, off_y)
        else:
            self._paint_lines(p, drawn)
        p.end()
        if self.order == "interleaved":
            return []
        # UOMO / CANE under the present of each group.
        labels = []
        for kind in KINDS:
            ends = fronts.get(kind)
            if not ends:
                continue
            x = sum(e[0] for e in ends) / len(ends)
            y = max(e[1] for e in ends) + 0.03 * h
            k = self._render_scale()
            labels.append((LABELS[kind], x / k, y / k))
        return labels

    def _pen(self, base: QColor, age: float, weight: float = 1.0) -> QPen:
        """The past dissolves: with depth a strand turns toward the shared
        haze and fades out. Contrast and thickness as set."""
        c = _mix(base, _HAZE, min(1.0, 1.3 * age))
        c.setAlphaF(min(1.0, self.style.contrast * weight * 0.95
                        * (1.0 - age) ** 1.7))
        pen = QPen(c, self.style.thickness)
        pen.setCosmetic(True)
        return pen

    def _paint_lines(self, p: QPainter, drawn: list) -> None:
        for base, sx, sy in drawn:
            edges = np.linspace(0, sx.shape[0] - 1, _BANDS + 1).astype(int)
            for b in range(_BANDS):
                p.setPen(self._pen(base, b / (_BANDS - 1)))
                lo, hi = edges[b], edges[b + 1] + 1
                p.drawPath(arrayToQPath(sx[lo:hi], sy[lo:hi]))

    def _paint_surface(self, p: QPainter, world: list, drawn: list, cam,
                       scale: float, off_x: float, off_y: float) -> None:
        """All the strands woven into one surface: a single wave in three
        dimensions, made of fine lines.

        Between each pair of neighbouring strands run _SUBDIV - 1 threads,
        interpolated smoothly across the channels (Catmull-Rom), each in a
        colour part of the way from one strand to the next, so pink and
        cyan mix in the lines themselves. The strands are the brightest
        lines, heavier near the present; cross-sections at even intervals
        of time keep the structure. The surface under the lines is solid,
        the colour of the background, so the fabric hides what lies behind
        it. No shading, no gradients, nothing blended over the last frame.
        The signal is lightly smoothed along time here (the fabric should
        flow, not bristle), the strands as well, so they lie on it. Painted
        in strips from the past forward, each hiding what lies behind it.
        """
        n = min(h.shape[0] for _c, _x, h in world)

        def fit(a: np.ndarray) -> np.ndarray:
            if a.shape[0] == n:
                return a
            return np.interp(np.linspace(0, a.shape[0] - 1, n),
                             np.arange(a.shape[0]), a)

        k = len(world)
        full = np.stack([fit(h) for _c, _x, h in world])        # (k, n)
        pad = _SMOOTH_KERNEL.shape[0] // 2
        padded = np.pad(full, ((0, 0), (pad, pad)), mode="edge")
        full = np.stack([np.convolve(row, _SMOOTH_KERNEL, mode="valid")
                         for row in padded])
        xk = np.array([x for _c, x, _h in world], dtype=float)
        ck = [c for c, _x, _h in world]
        z_all = np.linspace(0.0, _DEPTH, n)

        # Rows across, at full detail: the strands and the threads between.
        t = np.arange((k - 1) * _SUBDIV + 1) / _SUBDIV
        i0 = np.minimum(np.floor(t).astype(int), k - 2)
        fr = t - i0
        f = fr[:, None]
        p0 = full[np.clip(i0 - 1, 0, k - 1)]
        p1 = full[i0]
        p2 = full[np.clip(i0 + 1, 0, k - 1)]
        p3 = full[np.clip(i0 + 2, 0, k - 1)]
        rows_h = 0.5 * (2 * p1 + (p2 - p0) * f
                        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f ** 2
                        + (3 * p1 - p0 - 3 * p2 + p3) * f ** 3)
        rows_x = xk[i0] + (xk[i0 + 1] - xk[i0]) * fr
        r_n = rows_h.shape[0]
        pts = np.stack([
            np.repeat(rows_x[:, None], n, axis=1), rows_h,
            np.repeat(z_all[None, :], r_n, axis=0),
        ], axis=-1).reshape(-1, 3)
        u, v = self._project(pts, cam)
        sx = (off_x + scale * u).reshape(r_n, n)
        sy = (off_y - scale * v).reshape(r_n, n)

        # Each thread in a colour part of the way from pink to cyan, as far
        # as it is from its two strands: a few steps, so a whole band of
        # threads of one colour is one stroke.
        strand_rows = np.arange(k) * _SUBDIV
        threads: dict[int, list[int]] = {}
        for r in range(r_n):
            if r % _SUBDIV == 0:
                continue
            start = ck[i0[r]]
            toward_cyan = fr[r] if start.red() > start.blue() else 1 - fr[r]
            step = int(round(toward_cyan * (_SUBDIV - 1)))
            threads.setdefault(step, []).append(r)
        thread_colors = {
            q: _mix(QColor(ACCENT_PINK), QColor(ACCENT_CYAN),
                    q / (_SUBDIV - 1))
            for q in threads
        }
        strand_groups: dict[str, list[int]] = {}
        for j in range(k):
            strand_groups.setdefault(ck[j].name(), []).append(j)

        cols = np.arange(0, n, _MESH_STEP)
        if cols[-1] != n - 1:
            cols = np.append(cols, n - 1)
        c_n = cols.shape[0]
        bands = np.linspace(0, c_n - 1, _SURFACE_BANDS + 1).astype(int)
        hide = QColor(self._bg)

        def stroke(rows, lo, hi, step=1):
            idx = np.arange(lo, hi + 1, step)
            if idx[-1] != hi:
                idx = np.append(idx, hi)      # bands must meet
            joins = np.ones(idx.shape[0], dtype=bool)
            joins[-1] = False                 # no line from one to the next
            path = arrayToQPath(sx[np.ix_(rows, idx)].ravel(),
                                sy[np.ix_(rows, idx)].ravel(),
                                connect=np.tile(joins, len(rows)))
            p.drawPath(path)
            return path

        for b in range(_SURFACE_BANDS - 1, -1, -1):           # past first
            for ci in range(bands[b + 1] - 1, bands[b] - 1, -1):
                if ci >= c_n - 1:
                    continue
                c, c2 = cols[ci], cols[ci + 1]
                ax, ay = sx[:, c], sy[:, c]
                bx, by = sx[:, c2], sy[:, c2]
                body = arrayToQPath(np.concatenate([ax, bx[::-1]]),
                                    np.concatenate([ay, by[::-1]]))
                body.closeSubpath()
                p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                p.fillPath(body, hide)
                p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                # The cross-section on the far edge of this strip, drawn once
                # both strips beside it are filled (else it comes out dotted).
                if (ci + 1) % _MESH_SECTION == 0:
                    p.setPen(self._grid_pen(float(z_all[c2] / _DEPTH)))
                    p.drawPath(arrayToQPath(bx, by))
            lo, hi = cols[bands[b]], cols[bands[b + 1]]
            age = float(z_all[lo] / _DEPTH)
            fabric = _THREAD_WEIGHT * self.style.fabric
            for q, rows in threads.items():
                if fabric <= 0.0:
                    break
                p.setPen(self._pen(thread_colors[q], age, fabric))
                stroke(rows, lo, hi, _THREAD_STEP)
            for _name, members in strand_groups.items():
                p.setPen(self._pen(ck[members[0]], age))
                path = stroke(list(strand_rows[members]), lo, hi)
                if age < 0.5:                 # heavier near the present
                    p.drawPath(path.translated(0.0, 0.6))
        # The present: the front edge of the surface.
        p.setPen(self._grid_pen(0.0, 1.6))
        p.drawPath(arrayToQPath(sx[:, 0], sy[:, 0]))

    def _grid_pen(self, age: float, weight: float = 1.0) -> QPen:
        """Cross-sections: neutral, thin, gone well before the strands."""
        c = QColor(_GRID)
        c.setAlphaF(min(1.0, self.style.contrast * weight * 0.52
                        * (1.0 - age) ** 2.2))
        pen = QPen(c, self.style.thickness)
        pen.setCosmetic(True)
        return pen

    def _draw_labels(self, p: QPainter, labels: list) -> None:
        font = QFont(self.font())
        font.setPointSizeF(8.0)
        font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 130)
        p.setFont(font)
        c = QColor(TEXT_DIM)
        c.setAlphaF(0.9)
        p.setPen(c)
        for text, x, y in labels:
            p.drawText(QRectF(x - 60, y, 120, 18),
                       Qt.AlignmentFlag.AlignHCenter
                       | Qt.AlignmentFlag.AlignTop, text)


__all__ = ["LandscapeView", "KINDS", "ORDERS", "SHAPES"]
