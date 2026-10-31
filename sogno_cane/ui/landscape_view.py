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

Plain QPainter with numpy projection: no OpenGL module to ship, nothing a
PC without 3D drivers (or the updater, which replaces only this package)
would miss.
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
# The strands are drawn at reduced resolution and eased up, which softens
# them: about 700 lines tall whatever the window (half resolution on a
# projector, full in the app, where the view is low and halving it blurred
# the strands into ribbons).
_RENDER_LINES = 700
_DRIFT_PERIOD_S = 70.0       # slow drift of the point of view


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

    def __init__(
        self,
        profiles: dict[str, DeviceProfile],
        seconds: float = 6.0,
        background: str = BG_DEEP,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._streams = {k: _Stream(p, seconds) for k, p in profiles.items()}
        self.order = "grouped"
        self.show_labels = True
        self.gain = 1.0
        self.drift = True
        self._bg = QColor(background)
        self._accum: QImage | None = None
        self._t0 = time.perf_counter()
        self._last = self._t0
        self._fit_key = None
        self._fit = (1.0, 0.0, 0.0)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMinimumHeight(180)

    # ------------------------------------------------------------------ #
    # Feeding                                                            #
    # ------------------------------------------------------------------ #

    def push(self, kind: str, packet: UnicornPacket) -> None:
        stream = self._streams.get(kind)
        if stream is not None:
            stream.push(packet, time.perf_counter())

    def set_profile(self, kind: str, profile: DeviceProfile) -> None:
        """Another headset on that side: start its strands afresh."""
        old = self._streams.get(kind)
        if old is None:
            return
        seconds = old.n / old.fs
        visible = old.visible
        self._streams[kind] = _Stream(profile, seconds)
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

    def advance(self) -> None:
        """Reveal what arrived since the last frame and repaint."""
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

    def _camera(self, half_width: float, phase: float):
        """Low and to one side, looking along the strands into the past."""
        sway = math.sin(phase)
        target = np.array([0.35 * half_width, 0.0, 0.42 * _DEPTH])
        pos = np.array([
            -half_width - 3.2 + 1.4 * sway,
            2.3 + 0.35 * math.sin(phase * 0.7 + 1.0),
            -4.6,
        ])
        f = target - pos
        f /= np.linalg.norm(f)
        r = np.cross([0.0, 1.0, 0.0], f)
        r /= np.linalg.norm(r)
        u = np.cross(f, r)
        return pos, r, u, f

    @staticmethod
    def _project(points: np.ndarray, cam) -> tuple[np.ndarray, np.ndarray]:
        pos, r, u, f = cam
        v = points - pos
        zc = np.maximum(v @ f, 1e-2)
        return (v @ r) / zc, (v @ u) / zc

    def _fit_to(self, half_width: float, w: int, h: int):
        """Pixels per unit and offsets so the scene fills the frame."""
        key = (w, h, round(half_width, 3))
        if key == self._fit_key:
            return self._fit
        cam = self._camera(half_width, 0.0)
        corners = np.array([
            [x, y, z]
            for x in (-half_width, half_width)
            for y in (-_HEIGHT, _HEIGHT)
            for z in (0.0, _DEPTH * 0.7)
        ])
        u, v = self._project(corners, cam)
        span_x, span_y = u.max() - u.min(), v.max() - v.min()
        scale = min(1.0 * w / span_x, 0.96 * h / span_y)
        off_x = 0.52 * w - scale * (u.max() + u.min()) / 2
        off_y = 0.56 * h + scale * (v.max() + v.min()) / 2
        self._fit_key, self._fit = key, (scale, off_x, off_y)
        return self._fit

    # ------------------------------------------------------------------ #
    # Painting                                                           #
    # ------------------------------------------------------------------ #

    def _render_scale(self) -> float:
        return min(1.0, max(0.5, _RENDER_LINES / max(1, self.height())))

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
        cam = self._camera(half, phase)
        w, h = img.width(), img.height()
        scale, off_x, off_y = self._fit_to(half, w, h)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        fronts: dict[str, list[tuple[float, float]]] = {}
        for kind, base, values, x in rows:
            k = values.shape[0]
            zs = np.linspace(0.0, _DEPTH, k)
            pts = np.column_stack([
                np.full(k, x), _HEIGHT * self.gain * values, zs,
            ])
            u, v = self._project(pts, cam)
            sx = off_x + scale * u
            sy = off_y - scale * v
            fronts.setdefault(kind, []).append((float(sx[0]), float(sy[0])))
            # The past dissolves: band by band with depth the strand turns
            # toward the shared haze and fades out.
            edges = np.linspace(0, k - 1, _BANDS + 1).astype(int)
            for b in range(_BANDS):
                age = b / (_BANDS - 1)
                c = _mix(base, _HAZE, min(1.0, 1.3 * age))
                c.setAlphaF(0.95 * (1.0 - age) ** 1.7)
                pen = QPen(c, 1.0)
                pen.setCosmetic(True)
                p.setPen(pen)
                lo, hi = edges[b], edges[b + 1] + 1
                p.drawPath(arrayToQPath(sx[lo:hi], sy[lo:hi]))
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


__all__ = ["LandscapeView", "KINDS", "ORDERS"]
