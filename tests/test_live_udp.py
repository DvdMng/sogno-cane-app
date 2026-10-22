"""Live UDP acquisition from a real Unicorn, exercised without hardware.

A fake sender reproduces, byte for byte, what g.tec's own tools put on the
wire (documented in unicorn-bi/Unicorn-Network-Interfaces-Hybrid-Black and
in the Unicorn BCI Core Recorder README):

* Unicorn UDP Interface: 17 little-endian float32 per datagram (68 bytes),
  ``EEG1..8, ACC XYZ, GYR XYZ, BAT, CNT, VALID``.
* Unicorn BCI Core Recorder: the channels enabled in Acquisition settings,
  always EEG-first, default order ``EEG1..8, SAT, CNT, BAT, VALID, DT,
  STATUS``.

The tests assert the samples survive the trip unchanged and drive the real
mapping engine into real MIDI events.
"""
import socket
import struct
import threading
import time

import numpy as np
import pytest

from sogno_cane.core.engine import RealtimeEngine
from sogno_cane.eeg.live import (
    LAYOUT_EEG_ONLY,
    LAYOUT_RECORDER_14,
    LAYOUT_UNICORN_17,
    UnicornLiveSource,
    describe_layout,
    frame_to_packet_rows,
    probe_udp_ports,
)
from sogno_cane.eeg.unicorn_packet import PACKET_VALUES
from sogno_cane.midi.mapper import MappingConfig
from sogno_cane.midi.presets import rich_vocabulary_preset


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _FakeUnicorn:
    """Sends Unicorn-shaped datagrams at the device cadence."""

    def __init__(self, port: int, n_channels: int = PACKET_VALUES,
                 rate_hz: int = 250, amplitude_uv: float = 25.0):
        self.port = port
        self.n_channels = n_channels
        self.rate_hz = rate_hz
        self.amplitude_uv = amplitude_uv
        self._stop = threading.Event()
        self._thread = None
        self.sent = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        period = 1.0 / self.rate_hz
        next_t = time.monotonic()
        i = 0
        while not self._stop.is_set():
            t = i / self.rate_hz
            vals = [0.0] * self.n_channels
            # 10 Hz alpha on every EEG channel.
            for ch in range(min(8, self.n_channels)):
                vals[ch] = self.amplitude_uv * np.sin(2 * np.pi * 10 * t)
            if self.n_channels == PACKET_VALUES:
                vals[14] = 88.0            # BAT
                vals[15] = float(i)        # CNT
                vals[16] = 1.0             # VALID
            elif self.n_channels == 14:
                vals[9] = float(i)         # CNT
                vals[10] = 73.0            # BAT
                vals[11] = 1.0             # VALID
            sock.sendto(
                struct.pack("<%df" % self.n_channels, *vals),
                ("127.0.0.1", self.port),
            )
            self.sent += 1
            i += 1
            # Keep the device's pace on average (never reset the schedule):
            # Windows wakes threads every ~16 ms, so samples go out in small
            # bursts — just like the real headset over Bluetooth.
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
        sock.close()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


# --------------------------------------------------------------------------- #
# Layout mapping                                                              #
# --------------------------------------------------------------------------- #

def test_layout_detection_by_channel_count():
    assert describe_layout(17) == LAYOUT_UNICORN_17
    assert describe_layout(14) == LAYOUT_RECORDER_14
    assert describe_layout(8) == LAYOUT_EEG_ONLY


def test_unicorn_17_layout_passes_through_untouched():
    """The UDP Interface layout is already our packet layout."""
    frame = np.arange(2 * 17, dtype=np.float32).reshape(2, 17)
    out = frame_to_packet_rows(frame)
    assert np.array_equal(out, frame)


def test_recorder_14_layout_maps_telemetry_not_flags():
    """SAT/DT/STATUS must never be mistaken for battery or EEG."""
    frame = np.zeros((1, 14), dtype=np.float32)
    frame[0, :8] = np.arange(8) + 1.0     # EEG
    frame[0, 8] = 999.0                   # SAT (must be ignored)
    frame[0, 9] = 42.0                    # CNT
    frame[0, 10] = 73.0                   # BAT
    frame[0, 11] = 1.0                    # VALID
    frame[0, 12] = 4.0                    # DT (must be ignored)
    frame[0, 13] = 7.0                    # STATUS (must be ignored)
    out = frame_to_packet_rows(frame)
    assert out.shape == (1, PACKET_VALUES)
    assert np.array_equal(out[0, :8], np.arange(8) + 1.0)
    assert out[0, 14] == 73.0             # battery
    assert out[0, 15] == 42.0             # counter
    assert out[0, 16] == 1.0              # validation
    # Accelerometer/gyroscope are absent from this sender: zero, not garbage.
    assert np.all(out[0, 8:14] == 0.0)
    assert 999.0 not in out[0]


def test_eeg_only_layout_synthesises_telemetry():
    frame = np.ones((3, 8), dtype=np.float32)
    out = frame_to_packet_rows(frame)
    assert np.all(out[:, :8] == 1.0)
    assert np.all(out[:, 14] == 100.0)    # plausible battery
    assert np.all(out[:, 16] == 1.0)      # marked valid


# --------------------------------------------------------------------------- #
# Receiving real datagrams                                                    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n_channels,expected", [
    (PACKET_VALUES, LAYOUT_UNICORN_17),
    (14, LAYOUT_RECORDER_14),
    (8, LAYOUT_EEG_ONLY),
])
def test_receives_and_decodes_each_sender_layout(n_channels, expected):
    port = _free_port()
    src = UnicornLiveSource(port=port, host="127.0.0.1", packet_samples=25)
    src.open()
    sender = _FakeUnicorn(port, n_channels=n_channels).start()
    try:
        deadline = time.monotonic() + 5.0
        got = []
        while (time.monotonic() < deadline
               and sum(p.samples.shape[0] for p in got) < 60):
            candidate = src.next_packet()
            if (candidate is not None and np.any(candidate.eeg)
                    and np.all(candidate.validation > 0.5)):
                got.append(candidate)
        assert got, "no live packet decoded"
        for pkt in got:
            # Handed over as delivered (any number of rows), canonical cols.
            assert pkt.samples.shape[0] >= 1
            assert pkt.samples.shape[1] == PACKET_VALUES
            assert pkt.samples.dtype == np.float32
            assert pkt.sample_rate == 250
        eeg = np.concatenate([p.eeg for p in got])
        assert eeg.shape[0] >= 60
        # The 10 Hz alpha survived the wire at the right amplitude.
        assert 10.0 < float(eeg[:, 0].std()) < 30.0
        st = src.stats
        assert st["layout"] == expected
        assert st["channels"] == n_channels
        assert st["datagrams"] > 0
        assert st["malformed"] == 0
        assert src.has_signal
    finally:
        sender.stop()
        src.close()


def test_dropout_yields_silence_not_engine_stop():
    """A gap on the wire must not look like 'source exhausted' -- that would
    stop the engine and kill MIDI until the user notices."""
    port = _free_port()
    src = UnicornLiveSource(port=port, host="127.0.0.1", packet_samples=25)
    src.open()
    sender = _FakeUnicorn(port).start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not src.has_signal:
            src.next_packet()
        assert src.has_signal
        sender.stop()          # device goes away mid-performance
        time.sleep(0.3)
        src.next_packet()      # what arrived before it went away
        for _ in range(5):
            pkt = src.next_packet()
            assert pkt is not None, "dropout must not end the stream"
            # Held at the nominal pace, honestly flagged as not real.
            assert pkt.samples.shape == (25, PACKET_VALUES)
            assert np.all(pkt.validation == 0.0)
        assert not src.has_signal      # honestly reported as dead
    finally:
        sender.stop()
        src.close()
    # Only once closed does the source declare itself finished.
    assert src.next_packet() is None


def test_backlog_is_bounded_so_latency_stays_flat():
    """A stalled consumer must drop old samples, not queue them forever."""
    port = _free_port()
    src = UnicornLiveSource(
        port=port, host="127.0.0.1", packet_samples=25, buffer_seconds=0.5,
    )
    src.open()
    sender = _FakeUnicorn(port).start()
    try:
        time.sleep(1.5)        # never call next_packet(): consumer stalled
        with src._lock:
            buffered = len(src._buf)
        assert buffered <= src._buf.maxlen
        assert src.stats["dropped"] > 0, "old samples must be dropped"
    finally:
        sender.stop()
        src.close()


def test_malformed_datagrams_are_counted_and_ignored():
    port = _free_port()
    src = UnicornLiveSource(port=port, host="127.0.0.1", packet_samples=5)
    src.open()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(b"hello", ("127.0.0.1", port))          # not a float frame
        s.sendto(b"\x00" * 3, ("127.0.0.1", port))       # too short
        s.close()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and src.stats["malformed"] < 2:
            time.sleep(0.02)
        assert src.stats["malformed"] >= 2
        assert src.stats["datagrams"] == 0
    finally:
        src.close()


def test_probe_finds_the_active_port():
    port = _free_port()
    sender = _FakeUnicorn(port).start()
    try:
        found = probe_udp_ports(ports=(port,), timeout=2.0, host="127.0.0.1")
        assert found, "plug-and-play detection found nothing"
        assert found[0]["port"] == port
        assert found[0]["channels"] == PACKET_VALUES
        assert found[0]["layout"] == LAYOUT_UNICORN_17
    finally:
        sender.stop()


# --------------------------------------------------------------------------- #
# End to end: real device stream -> existing translation pipeline -> MIDI     #
# --------------------------------------------------------------------------- #

class _FakeMidi:
    def __init__(self):
        self.notes = []

    def send_note_on(self, note, velocity=96, channel=0):
        self.notes.append((note, channel))

    def send_note_off(self, note, channel=0):
        pass

    def send_control_change(self, control, value, channel=0):
        pass

    def send_pitch_bend(self, value, channel=0):
        pass


def test_live_stream_drives_the_existing_midi_pipeline():
    """The whole point: a real device stream must flow through the unchanged,
    already-tested translation chain and come out as MIDI."""
    port = _free_port()
    src = UnicornLiveSource(port=port, host="127.0.0.1", packet_samples=25)
    src.open()
    sender = _FakeUnicorn(port, amplitude_uv=40.0).start()
    midi = _FakeMidi()
    eng = RealtimeEngine(
        source=src,
        midi=midi,
        mapping_config=MappingConfig(sample_rate_hz=250),
    )
    for s in rich_vocabulary_preset().as_list():
        eng.mapper.add_strategy(s)
    try:
        eng.start()
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline and len(midi.notes) < 3:
            time.sleep(0.1)
        assert eng.is_running, "engine must stay alive on a live source"
        assert len(midi.notes) >= 3, (
            f"live EEG produced only {len(midi.notes)} notes"
        )
        # Real samples genuinely crossed the wire and none were garbled.
        assert src.stats["samples"] > 200, "almost no data arrived"
        assert src.stats["malformed"] == 0
    finally:
        eng.stop()
        sender.stop()
        src.close()


def test_engine_uses_the_live_source_sample_rate():
    port = _free_port()
    src = UnicornLiveSource(port=port, host="127.0.0.1")
    eng = RealtimeEngine(source=src, midi=_FakeMidi())
    assert eng.mapper.config.sample_rate_hz == 250
    src.close()


# --------------------------------------------------------------------------- #
# Command-line diagnostic and Qt panel wiring                                 #
# --------------------------------------------------------------------------- #

def test_live_test_cli_reports_a_running_stream(capsys):
    """`python -m sogno_cane --live-test <port>` is the on-site check."""
    from sogno_cane.__main__ import _live_test

    port = _free_port()
    sender = _FakeUnicorn(port).start()
    try:
        rc = _live_test(["--live-test", str(port)])
        out = capsys.readouterr().out
    finally:
        sender.stop()
    assert rc == 0, out
    assert "live acquisition OK" in out
    assert LAYOUT_UNICORN_17 in out
    assert "electrode offset" in out
    assert "signal around it" in out


def test_live_test_cli_reports_silence_clearly(capsys):
    from sogno_cane.__main__ import _live_test

    port = _free_port()
    rc = _live_test(["--live-test", str(port)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NO DATA" in out


def test_device_panel_connects_to_a_live_stream(tmp_path, monkeypatch):
    """The panel adopts the stream, starts the engine and reports health."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    monkeypatch.setenv("SOGNO_CANE_HOME", str(tmp_path))
    from sogno_cane.eeg.profiles import HUMAN_PROFILE
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.device_panel import DevicePanel

    app = QApplication.instance() or QApplication([])
    port = _free_port()
    sender = _FakeUnicorn(port, amplitude_uv=40.0).start()
    panel = DevicePanel(
        "HUMAN", HUMAN_PROFILE, archive=Archive(root=str(tmp_path / "rec")),
    )
    try:
        panel._connect_live(port=port)       # explicit port: no dialog
        assert panel.live_source is not None
        assert panel.is_streaming, "connecting must start the engine"
        deadline = time.monotonic() + 6.0
        while (
            time.monotonic() < deadline
            and not panel.live_source.has_signal
        ):
            app.processEvents()
            time.sleep(0.05)
        assert panel.live_source.has_signal
        panel._update_stats()
        assert "EEG" in panel._stats.text()
        assert "LIVE" in panel._source_lbl.text()
        # Switching back to the simulator must free the socket.
        panel._use_simulator()
        assert panel.live_source is None
    finally:
        sender.stop()
        panel.shutdown()
    # Port is released: a fresh bind on it must succeed.
    probe = UnicornLiveSource(port=port, host="127.0.0.1")
    probe.open()
    probe.close()
