"""Scrolling multi-channel waveform display backed by pyqtgraph.

Uses a calm two-color alternation (pink / cyan) across the 8 EEG channels
with thin anti-aliased traces.

The traces scroll continuously, whatever the lumps the samples arrive in
(see :class:`_Playout`), and are drawn with 1 px pens: Qt draws those on
its fast path, ~100x cheaper than the 1.4 px antialiased lines used up to
0.9.2, which cost ~40 ms a frame for the two panels and capped the display
at ~16 frames a second.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget
import pyqtgraph as pg

from sogno_cane.eeg.profiles import DeviceProfile
from sogno_cane.eeg.unicorn_packet import UnicornPacket
from sogno_cane.ui.theme import ACCENT_CYAN, ACCENT_PINK, BG_DEEP, TEXT_DIM


class _DisplayCleaner:
    """Causal per-channel DC blocker + mains notch, for the display only.

    A real Unicorn is DC-coupled: each electrode rides on an offset of tens
    of thousands of microvolts, so raw traces land far outside the plot (it
    showed nothing at all). Dry electrodes also pick up 100-170 uV of mains
    hum. This shows what the analysis actually looks at: offset removed,
    50/100 Hz notched. Stateful and causal, so there is no edge effect; the
    blocker is seeded with the first sample, so there is no start-up spike.
    """

    def __init__(self, n_channels: int, fs: float, mains_hz: float = 50.0):
        self._r_dc = 0.995                     # ~0.2 Hz high-pass at 250 Hz
        self._x_prev: np.ndarray | None = None
        self._y_prev = np.zeros(n_channels)
        self._notches: list[tuple[float, float, float, float, float]] = []
        self._states: list[np.ndarray] = []
        if mains_hz > 0.0:
            harmonic = mains_hz
            while harmonic < fs / 2.0:
                self._notches.append(self._design_notch(harmonic, fs))
                self._states.append(np.zeros((2, n_channels)))
                harmonic += mains_hz

    @staticmethod
    def _design_notch(f0: float, fs: float, bandwidth_hz: float = 4.0):
        w0 = 2.0 * np.pi * f0 / fs
        r = 1.0 - np.pi * bandwidth_hz / fs
        c = np.cos(w0)
        gain = (1.0 - 2.0 * r * c + r * r) / (2.0 - 2.0 * c)   # unity at DC
        return (gain, -2.0 * c * gain, gain, -2.0 * r * c, r * r)

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float64)
        for i in range(x.shape[0]):
            xi = x[i].astype(np.float64)
            if self._x_prev is None:
                self._x_prev = xi.copy()
            v = xi - self._x_prev + self._r_dc * self._y_prev
            self._x_prev = xi
            self._y_prev = v
            for (b0, b1, b2, a1, a2), z in zip(self._notches, self._states):
                o = b0 * v + z[0]
                z[0] = b1 * v - a1 * o + z[1]
                z[1] = b2 * v - a2 * o
                v = o
            out[i] = v
        return out


class _Playout:
    """Reveal samples at the pace they are produced, not in the lumps they
    arrive in.

    Every source hands samples over in lumps: the simulator and file
    playback 25 at a time ten times a second, a real headset 4 at a time
    ~60 times a second (its Bluetooth packets), with the odd late lump of
    8-16. Drawing each lump the moment it lands makes the trace jump.
    Instead the display runs about one lump behind and reveals the backlog
    at the measured input rate, sped up or slowed by a few percent to keep
    that backlog steady — a jitter buffer, for the eye. Measuring the rate
    (instead of assuming 250 Hz) follows a headset whose clock runs a little
    fast and slowed-down file playback alike.
    """

    _RATE_WINDOW_S = 3.0       # arrivals used to measure rate and lump size
    _LAG_SMOOTH_S = 0.5        # backlog smoothing, so lumps don't wobble it
    _CATCH_UP_S = 1.0          # correct a backlog error over about 1 s
    _JUMP_AFTER_S = 1.0        # this far behind (a stalled UI): jump ahead

    def __init__(self, fs: float) -> None:
        self._nominal = float(fs)
        self._chunks: deque[np.ndarray] = deque()
        self.pending = 0
        self._arrivals: deque[tuple[float, int]] = deque()
        self._acc = 0.0
        self._last_t: float | None = None
        self._lag = 0.0

    def push(self, rows: np.ndarray, now: float) -> None:
        if rows.shape[0] == 0:
            return
        self._chunks.append(rows)
        self.pending += rows.shape[0]
        self._arrivals.append((now, rows.shape[0]))
        while now - self._arrivals[0][0] > self._RATE_WINDOW_S:
            self._arrivals.popleft()

    def input_rate(self) -> float:
        """Samples per second actually coming in (nominal until measured)."""
        if len(self._arrivals) >= 3:
            span = self._arrivals[-1][0] - self._arrivals[0][0]
            if span >= 0.5:
                n = sum(k for _t, k in self._arrivals) - self._arrivals[0][1]
                return max(1.0, n / span)
        return self._nominal

    def target(self, rate: float) -> float:
        """Backlog (samples) that rides out the pause before the next lump."""
        lump = max((k for _t, k in self._arrivals), default=0)
        return max(4.0, 0.7 * lump + 0.03 * rate)

    def take(self, now: float) -> np.ndarray | None:
        """The rows to reveal on this frame (``None``: nothing new)."""
        if self._last_t is None:
            self._last_t = now
        dt = min(max(0.0, now - self._last_t), 0.25)
        self._last_t = now
        if self.pending == 0:
            self._acc = 0.0
            return None
        rate = self.input_rate()
        target = self.target(rate)
        smooth = min(1.0, dt / self._LAG_SMOOTH_S)
        self._lag += smooth * (self.pending - self._lag)
        if self.pending > target + rate * self._JUMP_AFTER_S:
            n = int(self.pending - target)
            self._acc = 0.0
            self._lag = target
        else:
            excess_s = (self._lag - target) / rate
            speed = min(1.25, max(0.85, 1.0 + excess_s / self._CATCH_UP_S))
            self._acc += rate * speed * dt
            n = int(self._acc)
            if n >= self.pending:
                n = self.pending
                self._acc = 0.0          # ran dry: owe nothing
            else:
                self._acc -= n
        if n <= 0:
            return None
        return self._pop(n)

    def _pop(self, n: int) -> np.ndarray:
        out = []
        need = n
        while need > 0:
            head = self._chunks[0]
            if head.shape[0] <= need:
                out.append(self._chunks.popleft())
                need -= head.shape[0]
            else:
                out.append(head[:need])
                self._chunks[0] = head[need:]
                need = 0
        self.pending -= n
        return out[0] if len(out) == 1 else np.concatenate(out)


def _palette(n: int) -> list[str]:
    # Alternate pink / cyan with subtle brightness variations.
    out: list[str] = []
    for i in range(n):
        if i % 2 == 0:
            out.append("#FF5FA0" if i // 2 % 2 == 0 else "#FF3D8A")
        else:
            out.append("#82F1FF" if i // 2 % 2 == 0 else "#5BE9FF")
    return out


class WaveformWidget(QWidget):
    def __init__(
        self,
        profile: DeviceProfile,
        seconds_visible: float = 5.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._profile = profile
        self._fs = profile.sample_rate_hz
        self._n_visible = int(seconds_visible * self._fs)

        pg.setConfigOption("background", (10, 4, 24, 0))
        pg.setConfigOption("foreground", TEXT_DIM)
        pg.setConfigOption("antialias", True)

        self._plot = pg.PlotWidget()
        self._plot.setBackground(None)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.showGrid(x=False, y=False)
        self._plot.getAxis("left").setStyle(showValues=False)
        self._plot.getAxis("bottom").setStyle(showValues=False)
        self._plot.getAxis("left").setPen(pg.mkPen(TEXT_DIM, width=0))
        self._plot.getAxis("bottom").setPen(pg.mkPen(TEXT_DIM, width=0))
        self._plot.setXRange(0, self._n_visible)
        self._stride_uv = 260.0
        self._plot.setYRange(
            -self._stride_uv,
            self._stride_uv * len(profile.channel_names),
        )

        palette = _palette(len(profile.channel_names))
        self._curves: list[pg.PlotDataItem] = []
        for i, ch_name in enumerate(profile.channel_names):
            # 1 px: Qt's fast path (see the module notes).
            pen = pg.mkPen(palette[i], width=1.0)
            curve = self._plot.plot([], [], pen=pen)
            self._curves.append(curve)
            label = pg.TextItem(
                text=ch_name, color=TEXT_DIM, anchor=(0, 0.5),
            )
            label.setPos(2, i * self._stride_uv)
            self._plot.addItem(label)

        # The last n_visible samples, stored twice over so that the visible
        # window is always one contiguous slice: nothing is rebuilt per
        # frame (turning 16 deques into arrays 60 times a second cost ~6%
        # of a core on its own).
        n_ch = len(profile.channel_names)
        self._ring = np.zeros((2 * self._n_visible, n_ch), dtype=np.float32)
        self._head = 0                   # oldest sample = next write slot
        self._offsets = np.arange(n_ch, dtype=np.float32) * self._stride_uv
        self._x_axis = np.arange(self._n_visible)
        self._cleaner = _DisplayCleaner(len(profile.channel_names), self._fs)
        self._playout = _Playout(self._fs)
        self._shown_total = 0        # samples revealed so far (diagnostics)

        self.setMinimumHeight(160)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)

    # The playout clock is perf_counter: time.monotonic() on Windows
    # (Python 3.12) ticks every 15.6 ms, so at 60 fps some frames would see
    # no time pass at all and the trace would stall on them.

    def push_packet(self, packet: UnicornPacket) -> None:
        eeg = self._cleaner.process(packet.eeg)
        self._playout.push(eeg, time.perf_counter())

    def repaint_traces(self) -> None:
        """Advance the traces by what this frame should reveal."""
        rows = self._playout.take(time.perf_counter())
        if rows is None:
            return                   # nothing new: leave the picture alone
        self._shown_total += rows.shape[0]
        self._append(rows)
        n = self._n_visible
        window = self._ring[self._head:self._head + n]   # oldest -> newest
        for ch, curve in enumerate(self._curves):
            # The cleaner never outputs NaN/inf: skip pyqtgraph's check.
            curve.setData(self._x_axis, window[:, ch] + self._offsets[ch],
                          skipFiniteCheck=True)

    def _append(self, rows: np.ndarray) -> None:
        n = self._n_visible
        rows = rows[-n:]
        k = rows.shape[0]
        i = self._head
        first = min(k, n - i)
        for base in (0, n):                      # both copies
            self._ring[base + i:base + i + first] = rows[:first]
            if k > first:
                self._ring[base:base + k - first] = rows[first:]
        self._head = (i + k) % n
