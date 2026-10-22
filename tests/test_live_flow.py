"""The live stream flows at the headset's pace and scrolls smoothly (0.9.3).

Measured on g.tec's own raw recordings (the Recorder's DT column = ms
between two samples it received): a BCI Core headset samples at 250 Hz but
delivers over Bluetooth in bursts of 4 every ~16 ms (~5% of them late,
merged into 8), and against the PC clock it runs at ~250.65 Hz.

Up to 0.9.2 the app re-cut that stream into 25 samples per 100 ms of the PC
clock: the traces moved ten times a second in 100 ms jumps, and the latency
grew by ~0.15 s a minute (measured: +130 ms/min against a faithful emulator
of the Recorder). The fake headset below has a deliberately wrong clock so
the same drift shows within seconds.
"""
import os
import re
import socket
import threading
import time

import numpy as np
import pytest

from sogno_cane.core.engine import RealtimeEngine
from sogno_cane.eeg.live import UnicornLiveSource
from sogno_cane.eeg.unicorn_packet import UnicornPacket
from sogno_cane.midi.mapper import MappingConfig, MappingEngine

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FS = 250
DC = np.array([50384, 77483, 21621, 55072, 53106, 51083, 60829, 43080],
              dtype=np.float64)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class BurstyRecorder:
    """A BCI Core Recorder fed by a Bluetooth headset: bursts of ``burst``
    samples, one text line per datagram sent back to back, the DT column
    filled in the way the Recorder does, at ``rate_hz`` on the PC clock.
    ``skip`` = {sample index: how many counter values go missing}."""

    def __init__(self, port, rate_hz=250.0, burst=4, skip=None):
        self.port = port
        self.rate_hz = rate_hz
        self.burst = burst
        self.skip = dict(skip or {})
        self.sent = 0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        dest = ("127.0.0.1", self.port)
        period = self.burst / self.rate_hz
        t0 = time.perf_counter()
        i = cnt = 0
        while not self._stop.is_set():
            for j in range(self.burst):
                cnt += 1 + self.skip.get(i, 0)
                t = i / FS
                eeg = DC + 20.0 * np.sin(2 * np.pi * 10.0 * t + np.arange(8))
                dt_ms = period * 1000.0 if j == 0 else 0.005
                tail = [0.0, float(cnt), 70.0, 1.0, dt_ms, 0.0]
                line = ",".join(f"{v:.3f}" for v in list(eeg) + tail) + "\n"
                sock.sendto(line.encode("ascii"), dest)
                i += 1
            self.sent = i
            wake = t0 + (i // self.burst) * period
            while not self._stop.is_set():
                left = wake - time.perf_counter()
                if left <= 0:
                    break
                time.sleep(min(left, 0.005))
        sock.close()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


class _NullMidi:
    def send_note_on(self, *a, **k):
        pass

    send_note_off = send_control_change = send_pitch_bend = send_note_on


def _wait_signal(src, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not src.has_signal:
        time.sleep(0.02)
    assert src.has_signal, "the fake headset never got through"


# --------------------------------------------------------------------------- #
# The source hands samples over as they arrive                                #
# --------------------------------------------------------------------------- #

def test_bursts_are_handed_over_as_they_arrive():
    """No re-cutting into 25-sample lumps on the PC clock."""
    port = _free_port()
    src = UnicornLiveSource(port=port, host="127.0.0.1")
    src.open()
    rec = BurstyRecorder(port).start()
    try:
        _wait_signal(src)
        src.next_packet()                      # whatever piled up meanwhile
        sizes, waits = [], []
        for _ in range(40):
            t = time.perf_counter()
            pkt = src.next_packet()
            waits.append(time.perf_counter() - t)
            sizes.append(pkt.samples.shape[0])
            assert np.all(pkt.validation == 1.0), "nothing invented"
        assert np.median(sizes) <= 8, sizes    # bursts, not 25-sample lumps
        assert np.median(waits) < 0.03        # handed over within a burst
    finally:
        rec.stop()
        src.close()


@pytest.mark.parametrize("rate_hz", [235.0, 275.0])
def test_engine_follows_the_headset_clock_without_backlog(rate_hz):
    """Whatever the headset's clock, the engine consumes exactly what it
    delivers: no backlog (latency) piles up, nothing is invented."""
    port = _free_port()
    src = UnicornLiveSource(port=port, host="127.0.0.1")
    src.open()
    rec = BurstyRecorder(port, rate_hz=rate_hz).start()
    seen = []
    eng = RealtimeEngine(
        source=src, midi=_NullMidi(),
        mapping_config=MappingConfig(sample_rate_hz=FS),
        on_packet=lambda p: seen.append(
            (int(p.samples.shape[0]), int((p.validation < 0.5).sum()))),
    )
    try:
        _wait_signal(src)
        eng.start()
        time.sleep(0.5)
        start = len(seen)
        backlog = []
        for _ in range(30):                    # 3 s of streaming
            time.sleep(0.1)
            with src._lock:
                backlog.append(len(src._buf))
        run = seen[start:]
        received = sum(n for n, _bad in run)
        assert sum(bad for _n, bad in run) == 0, "held samples mid-stream"
        # 0.9.2 consumed 25 per 100 ms whatever came: at 275 Hz the backlog
        # grew by 25 samples (100 ms of latency) every second.
        assert max(backlog) <= 12, backlog
        assert received >= 0.9 * rate_hz * 3.0
    finally:
        eng.stop()
        rec.stop()
        src.close()


def test_silence_becomes_held_samples_then_the_stream_resumes():
    port = _free_port()
    src = UnicornLiveSource(port=port, host="127.0.0.1")
    src.open()
    rec = BurstyRecorder(port).start()
    try:
        _wait_signal(src)
        rec.stop()
        time.sleep(0.1)
        src.next_packet()                      # the last real samples
        held = src.next_packet()               # then the gap
        assert held.samples.shape[0] == 25 and np.all(held.validation == 0)
        assert np.all(np.abs(held.eeg[:, 0] - DC[0]) < 100.0), "held, not 0"
        rec = BurstyRecorder(port).start()     # the headset comes back
        deadline = time.monotonic() + 3.0
        pkt = src.next_packet()
        while time.monotonic() < deadline and np.all(pkt.validation == 0):
            pkt = src.next_packet()
        assert np.all(pkt.validation == 1.0), "real data again, unmixed"
    finally:
        rec.stop()
        src.close()


# --------------------------------------------------------------------------- #
# The source side, measured                                                   #
# --------------------------------------------------------------------------- #

def test_cadence_measures_rate_bursts_bluetooth_and_losses():
    port = _free_port()
    src = UnicornLiveSource(port=port, host="127.0.0.1")
    src.open()
    rec = BurstyRecorder(port, skip={300: 3}).start()
    try:
        _wait_signal(src)
        time.sleep(2.0)
        cad = src.cadence()
    finally:
        rec.stop()
        src.close()
    assert 240.0 <= cad["rate_hz"] <= 260.0, cad
    assert cad["samples_per_burst"] == 4.0, cad          # one burst, 4 lines
    assert 50.0 <= cad["bursts_per_s"] <= 75.0, cad
    assert cad["bt_burst_samples"] == 4.0, cad           # from the DT column
    assert 15.0 <= cad["bt_interval_ms"] <= 17.0, cad
    assert cad["lost_at_source"] == 3, cad               # the counter gap


def test_cadence_is_empty_before_any_data():
    src = UnicornLiveSource(port=_free_port(), host="127.0.0.1")
    cad = src.cadence()
    assert cad["rate_hz"] is None and cad["bursts_per_s"] is None
    assert cad["lost_at_source"] == 0


# --------------------------------------------------------------------------- #
# The analysis keeps its 100 ms rhythm with small packets                     #
# --------------------------------------------------------------------------- #

class _Clock:
    name = "clock"
    enabled = True
    channel = 0

    def __init__(self):
        self.t = []

    def on_window(self, eeg_window, bands, ctx):
        self.t.append(ctx.t_seconds)
        return []

    def shutdown(self):
        return []


def _packet(n, start):
    samples = np.zeros((n, 17), dtype=np.float32)
    samples[:, :8] = np.sin(np.arange(start, start + n) / 7.0)[:, None]
    samples[:, 16] = 1.0
    return UnicornPacket(samples=samples, timestamp=0.0, sample_rate=FS)


@pytest.mark.parametrize("burst", [4, 7, 25])
def test_analysis_hop_stays_100ms_whatever_the_packet_size(burst):
    """Restarting the hop count on the burst that crosses it stretched every
    hop to 28 samples with 4-sample bursts: the music 12% slow."""
    clock = _Clock()
    eng = MappingEngine(MappingConfig(sample_rate_hz=FS))
    eng.add_strategy(clock)
    done = 0
    while done < FS * 20:
        eng.process_packet(_packet(burst, done))
        done += burst
    hops = np.diff(clock.t)
    assert abs(hops.mean() - 0.1) < 0.002, hops.mean()


# --------------------------------------------------------------------------- #
# The display glides                                                          #
# --------------------------------------------------------------------------- #

def _playout():
    pytest.importorskip("PySide6")
    from sogno_cane.ui.waveform_widget import _Playout
    return _Playout(FS)


def _lumps_sim(seconds):
    """Simulator / file playback: 25 samples every 100 ms."""
    return [(k * 0.1, 25) for k in range(int(seconds * 10))]


def _lumps_headset(seconds, rate=250.65, late_every=20):
    """Headset: bursts of 4 every 16 ms, now and then one late (merged)."""
    out, t, k = [], 0.0, 0
    period = 4 / rate
    while t < seconds:
        if k % late_every == late_every - 1:
            k += 1
            t += period
            out.append((t, 8))
        else:
            out.append((t, 4))
        k += 1
        t += period
    return out


def _run(play, lumps, seconds, fps=60.0, stall=None):
    """Feed lumps and draw frames on a simulated clock; per-frame reveals."""
    frames, i, t = [], 0, 0.0
    while t < seconds:
        while i < len(lumps) and lumps[i][0] <= t:
            play.push(np.zeros((lumps[i][1], 8)), lumps[i][0])
            i += 1
        if stall is None or not (stall[0] <= t < stall[1]):
            rows = play.take(t)
            frames.append((t, 0 if rows is None else rows.shape[0],
                           play.pending))
        t += 1.0 / fps
    return frames


@pytest.mark.parametrize("lumps", ["sim", "headset"])
def test_traces_advance_on_every_frame_by_a_steady_amount(lumps):
    """0.9.2: the trace moved on 61% of the frames, by 25-50 samples."""
    play = _playout()
    feed = _lumps_sim(12) if lumps == "sim" else _lumps_headset(12)
    frames = _run(play, feed, 12.0)
    steady = [n for t, n, _p in frames if t > 2.0]
    assert min(steady) >= 2, "the trace froze on some frames"
    assert max(steady) <= 7, "the trace jumped"
    assert abs(np.mean(steady) - FS / 60.0) < 0.3


def test_display_lag_does_not_creep_with_a_fast_headset_clock():
    play = _playout()
    frames = _run(play, _lumps_headset(120, rate=250.65), 120.0)
    late = [p for t, _n, p in frames if t > 5.0]
    assert max(late) < 40, max(late)          # under 0.16 s, for 2 minutes


def test_display_jumps_ahead_after_the_window_was_stalled():
    play = _playout()
    frames = _run(play, _lumps_sim(8), 8.0, stall=(3.0, 5.0))
    after = [p for t, _n, p in frames if t > 5.2]
    assert max(after) < 40, "still dragging two seconds of backlog"


def test_display_follows_slowed_down_playback():
    play = _playout()
    feed = [(k * 0.2, 25) for k in range(60)]    # 125 samples/s
    frames = _run(play, feed, 12.0)
    steady = [n for t, n, _p in frames if t > 4.0]
    assert np.mean([n == 0 for n in steady]) < 0.05
    assert abs(np.mean(steady) - 125 / 60.0) < 0.3


# --------------------------------------------------------------------------- #
# Qt side: cheap frames, 60 fps, honest health line                           #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_traces_use_pens_on_qts_fast_path(app):
    """1.4 px antialiased lines cost ~4.7 ms each: ~40 ms a frame for the
    two panels, which held the display at ~16 frames a second."""
    from sogno_cane.eeg.profiles import HUMAN_PROFILE
    from sogno_cane.ui.waveform_widget import WaveformWidget
    w = WaveformWidget(HUMAN_PROFILE)
    assert all(c.opts["pen"].widthF() <= 1.0 for c in w._curves)


def test_the_visible_window_is_exactly_the_latest_samples(app):
    """The ring buffer behind the traces: oldest to newest, nothing lost or
    duplicated, whatever the sizes of the pieces revealed."""
    from sogno_cane.eeg.profiles import HUMAN_PROFILE
    from sogno_cane.ui.waveform_widget import WaveformWidget
    w = WaveformWidget(HUMAN_PROFILE)
    n = w._n_visible
    rng = np.random.default_rng(3)
    ramp = np.arange(4 * n + 77, dtype=np.float32)
    done = 0
    while done < ramp.size:
        k = int(rng.integers(1, 300))
        piece = ramp[done:done + k]
        w._append(np.repeat(piece[:, None], 8, axis=1))
        done += piece.size
        window = w._ring[w._head:w._head + n, 3]
        expect = ramp[max(0, done - n):done]
        assert np.array_equal(window[n - expect.size:], expect)
    w._append(np.repeat(ramp[:, None], 8, axis=1))   # more than a window
    assert np.array_equal(w._ring[w._head:w._head + n, 0], ramp[-n:])


def test_a_frame_with_nothing_new_costs_nothing(app):
    from sogno_cane.eeg.profiles import HUMAN_PROFILE
    from sogno_cane.ui.waveform_widget import WaveformWidget
    w = WaveformWidget(HUMAN_PROFILE)
    calls = []
    for c in w._curves:
        c.setData = lambda *a, **k: calls.append(1)
    w.repaint_traces()
    assert calls == []
    samples = np.zeros((25, 17), dtype=np.float32)
    w.push_packet(UnicornPacket(samples=samples, timestamp=0.0,
                                sample_rate=FS))
    time.sleep(0.05)
    w.repaint_traces()
    assert len(calls) == len(w._curves)


def test_panel_redraws_at_60_fps_on_a_precise_timer(app, tmp_path,
                                                    monkeypatch):
    from PySide6.QtCore import Qt
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    from sogno_cane.eeg.profiles import HUMAN_PROFILE
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.device_panel import DevicePanel
    panel = DevicePanel("UMANO", HUMAN_PROFILE,
                        archive=Archive(root=str(tmp_path / "rec")))
    try:
        assert panel._repaint_timer.interval() <= 17
        assert panel._repaint_timer.timerType() == Qt.TimerType.PreciseTimer
    finally:
        panel.shutdown()


@pytest.mark.parametrize("frame_s,expect_ms", [(0.016, 16), (0.040, 33)])
def test_a_pc_that_cannot_hold_60_fps_settles_for_30(
        app, tmp_path, monkeypatch, frame_s, expect_ms):
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    import sogno_cane.ui.device_panel as dp
    from sogno_cane.eeg.profiles import HUMAN_PROFILE
    from sogno_cane.io.archive import Archive
    panel = dp.DevicePanel("UMANO", HUMAN_PROFILE,
                           archive=Archive(root=str(tmp_path / "rec")))
    clock = [100.0]
    monkeypatch.setattr(dp.time, "perf_counter", lambda: clock[0])
    try:
        panel._repaint_timer.stop()            # we drive the frames
        for _ in range(130):
            panel._on_frame()
            clock[0] += frame_s
        assert panel._repaint_timer.interval() == expect_ms
    finally:
        monkeypatch.undo()
        panel.shutdown()


@pytest.mark.parametrize("rate_hz,expect", [
    (250.0, r"\b250 Hz,"),
    (125.0, r"\b12[3-7] Hz invece di 250"),   # a slow sender is called out
])
def test_health_line_shows_the_rate_measured_at_the_source(
        app, tmp_path, monkeypatch, rate_hz, expect):
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    from sogno_cane.eeg.profiles import HUMAN_PROFILE
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.device_panel import DevicePanel
    port = _free_port()
    rec = BurstyRecorder(port, rate_hz=rate_hz).start()
    panel = DevicePanel("UMANO", HUMAN_PROFILE,
                        archive=Archive(root=str(tmp_path / "rec")))
    try:
        panel._connect_live(port=port)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
        panel._update_stats()
        text = panel._stats.text()
        assert re.search(expect, text), text
        assert "campioni al secondo" in panel._stats.toolTip()
    finally:
        rec.stop()
        panel.shutdown()


def test_garbage_is_collected_by_the_gui_timer_not_by_chance(app):
    """Automatic GC off; the GUI-thread collector frees cycles when due."""
    import gc
    import weakref

    from sogno_cane.ui.gui_gc import GuiThreadGarbageCollector

    class Node:
        pass

    threshold = gc.get_threshold()
    collector = GuiThreadGarbageCollector(interval_ms=60_000)
    try:
        assert not gc.isenabled()
        a, b = Node(), Node()
        a.other, b.other = b, a                 # a cycle: refcount can't
        probe = weakref.ref(a)
        del a, b
        junk = [[] for _ in range(threshold[0] + 10)]   # make gen 0 due
        assert probe() is not None, "nothing collected behind our back"
        collector.check()
        assert probe() is None, "the GUI-thread collector freed the cycle"
        del junk
    finally:
        collector._timer.stop()


def test_a_live_recording_says_it_came_from_the_live_eeg(app, tmp_path,
                                                         monkeypatch):
    """It used to be filed as 'simulator'."""
    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    from sogno_cane.eeg.profiles import HUMAN_PROFILE
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.device_panel import DevicePanel
    port = _free_port()
    rec = BurstyRecorder(port).start()
    panel = DevicePanel("UMANO", HUMAN_PROFILE,
                        archive=Archive(root=str(tmp_path / "rec")))
    try:
        panel._connect_live(port=port)
        panel._start_record()
        assert panel._recorder.source == f"live:UDP {port}"
        panel._stop_record(save=False)
    finally:
        rec.stop()
        panel.shutdown()
