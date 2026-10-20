"""Compatibility with the real Unicorn BCI Core Recorder.

Ground truth used to write these tests (no hardware needed):

* g.tec's official receiver, ``UDP Receiver/Program.cs`` in
  unicorn-bi/Unicorn-BCI-Core-Recorder, decodes every datagram with
  ``Encoding.ASCII.GetString`` — the Recorder sends TEXT, not float32.
* The official Unicorn BCI Core-8 recordings (unicorn-bi/Unicorn-BCI-Core-8-
  EEG-Datasets) fix the exact serialisation: header
  ``EXG 1, ..., EXG 8, SATURATION, CNT, BAT, VALID, DT, TRIG``, comma
  separated, dot decimals, three decimals, LF line ends — and raw EEG values
  of tens of thousands of microvolts, because the amplifier is DC-coupled.

The datasets carry no licence, so nothing from them is copied here: the
fake Recorder below synthesises lines in the same format.

Measured on the real recordings before the fixes these tests guard: the old
analysis put 100% of the power in delta (DC leakage), and once the offset
was removed the 50 Hz mains made up 100% of gamma.
"""
import socket
import threading
import time

import numpy as np

from sogno_cane.core.engine import RealtimeEngine
from sogno_cane.eeg.bands import compute_band_powers
from sogno_cane.eeg.live import (
    ENCODING_BINARY,
    ENCODING_TEXT,
    LAYOUT_RECORDER_14,
    UnicornLiveSource,
    looks_like_text,
    parse_text_datagram,
    probe_udp_ports,
)
from sogno_cane.eeg.unicorn_packet import (
    BATTERY_INDEX,
    COUNTER_INDEX,
    PACKET_VALUES,
    VALIDATION_INDEX,
    UnicornPacket,
)
from sogno_cane.midi.mapper import MappingConfig, MappingEngine
from sogno_cane.midi.presets import rich_vocabulary_preset

FS = 250
HEADER = ("EXG 1, EXG 2, EXG 3, EXG 4, EXG 5, EXG 6, EXG 7, EXG 8, "
          "SATURATION, CNT, BAT, VALID, DT, TRIG")
# Electrode offsets of the size the real amplifier shows (uV).
DC = np.array([50384, 77483, 21621, 55072, 53106, 51083, 60829, 43080],
              dtype=np.float64)


def recorder_line(i: int, alpha_uv=20.0, mains_uv=150.0, battery=63.0,
                  valid=1.0, dc=DC) -> str:
    """One sample, serialised exactly like the BCI Core Recorder does."""
    t = i / FS
    eeg = (dc + alpha_uv * np.sin(2 * np.pi * 10.0 * t + np.arange(8))
           + mains_uv * np.sin(2 * np.pi * 50.0 * t))
    tail = [0.0, float(i + 1), battery, valid, 0.007, 0.0]
    return ",".join(f"{v:.3f}" for v in list(eeg) + tail) + "\n"


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeRecorder:
    """Streams ASCII sample lines over UDP at 250 Hz, like the Recorder."""

    def __init__(self, port, lines_per_datagram=1, send_header=True,
                 alpha_uv=20.0):
        self.port = port
        self.lines_per_datagram = lines_per_datagram
        self.send_header = send_header
        self.alpha_uv = alpha_uv
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        dest = ("127.0.0.1", self.port)
        if self.send_header:
            sock.sendto((HEADER + "\n").encode("ascii"), dest)
        i = 0
        next_t = time.monotonic()
        k = self.lines_per_datagram
        while not self._stop.is_set():
            chunk = "".join(
                recorder_line(i + j, alpha_uv=self.alpha_uv) for j in range(k)
            )
            sock.sendto(chunk.encode("ascii"), dest)
            i += k
            next_t += k / FS
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.monotonic()
        sock.close()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


# --------------------------------------------------------------------------- #
# Decoding the Recorder's text format                                         #
# --------------------------------------------------------------------------- #

def test_text_and_binary_are_told_apart():
    assert looks_like_text(recorder_line(0).encode("ascii"))
    assert looks_like_text((HEADER + "\n").encode("ascii"))
    binary = np.array([25.0] * 17, dtype="<f4").tobytes()
    assert not looks_like_text(binary)


def test_one_sample_line_decodes_exactly():
    parsed = parse_text_datagram(recorder_line(41).encode("ascii"))
    assert parsed is not None
    frame, headers = parsed
    assert headers == 0
    assert frame.shape == (1, 14)
    # EXG 1 exactly as serialised: offset + alpha + mains at sample 41.
    t = 41 / FS
    expected = (DC[0] + 20.0 * np.sin(2 * np.pi * 10.0 * t)
                + 150.0 * np.sin(2 * np.pi * 50.0 * t))
    assert abs(frame[0, 0] - expected) < 0.01
    assert frame[0, 9] == 42.0                    # CNT
    assert frame[0, 10] == 63.0                   # BAT
    assert frame[0, 11] == 1.0                    # VALID


def test_header_line_is_recognised_not_malformed():
    parsed = parse_text_datagram((HEADER + "\n").encode("ascii"))
    assert parsed is not None
    frame, headers = parsed
    assert headers == 1
    assert frame.shape[0] == 0


def test_several_lines_in_one_datagram():
    chunk = "".join(recorder_line(i) for i in range(5)).encode("ascii")
    frame, _ = parse_text_datagram(chunk)
    assert frame.shape == (5, 14)
    assert list(frame[:, 9]) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_crlf_and_nul_padding_are_tolerated():
    line = recorder_line(0).replace("\n", "\r\n").encode("ascii") + b"\x00" * 7
    src = UnicornLiveSource(port=_free_port())
    rows = src._parse(line)
    assert rows is not None and rows.shape == (1, PACKET_VALUES)


def test_semicolon_decimal_comma_variant():
    """European CSV (decimal comma, semicolon separator) must not split
    every number in two."""
    line = ";".join(["49780,490"] * 8 + ["0", "1", "63", "1", "0", "0"])
    frame, _ = parse_text_datagram((line + "\n").encode("ascii"))
    assert frame.shape == (1, 14)
    assert abs(frame[0, 0] - 49780.49) < 1e-2


def test_garbage_text_is_malformed():
    assert parse_text_datagram(b"hello world\n") is None
    assert parse_text_datagram(b"1,2,3\n") is None        # too few channels


def test_recorder_layout_maps_battery_counter_valid():
    src = UnicornLiveSource(port=_free_port())
    rows = src._parse(recorder_line(9, battery=58.0).encode("ascii"))
    assert rows.shape == (1, PACKET_VALUES)
    assert rows[0, BATTERY_INDEX] == 58.0
    assert rows[0, COUNTER_INDEX] == 10.0
    assert rows[0, VALIDATION_INDEX] == 1.0
    # The Recorder sends no accelerometer/gyroscope: zero, not SAT/DT/TRIG.
    assert np.all(rows[0, 8:14] == 0.0)
    st = src.stats
    assert st["layout"] == LAYOUT_RECORDER_14
    assert st["encoding"] == ENCODING_TEXT


def test_binary_with_trailing_zero_values_is_not_truncated():
    """0.0 is four NUL bytes: stripping NUL padding must never eat real
    float32 values (regression of a bug caught during development)."""
    vals = np.zeros(14, dtype="<f4")
    vals[:8] = 12.5
    src = UnicornLiveSource(port=_free_port())
    rows = src._parse(vals.tobytes())
    assert rows is not None and rows.shape == (1, PACKET_VALUES)
    assert src.stats["encoding"] == ENCODING_BINARY


# --------------------------------------------------------------------------- #
# Over the wire                                                               #
# --------------------------------------------------------------------------- #

def test_live_source_reads_a_text_recorder_stream():
    port = _free_port()
    src = UnicornLiveSource(port=port, host="127.0.0.1")
    src.open()
    rec = FakeRecorder(port).start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and src.stats["samples"] < 300:
            src.next_packet()
        st = src.stats
        assert st["samples"] >= 250, st
        assert st["malformed"] == 0, st
        assert st["header_lines"] == 1
        assert st["layout"] == LAYOUT_RECORDER_14
        assert st["encoding"] == ENCODING_TEXT
        assert src.has_signal
    finally:
        rec.stop()
        src.close()


def test_multi_line_datagrams_over_the_wire():
    port = _free_port()
    src = UnicornLiveSource(port=port, host="127.0.0.1")
    src.open()
    rec = FakeRecorder(port, lines_per_datagram=4).start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and src.stats["samples"] < 300:
            src.next_packet()
        assert src.stats["samples"] >= 250
        assert src.stats["malformed"] == 0
    finally:
        rec.stop()
        src.close()


def test_probe_finds_a_text_recorder():
    port = _free_port()
    rec = FakeRecorder(port, send_header=False).start()
    try:
        found = probe_udp_ports(ports=(port,), timeout=2.0, host="127.0.0.1")
        assert found, "plug-and-play detection missed the Recorder"
        assert found[0]["port"] == port
        assert found[0]["channels"] == 14
        assert found[0]["layout"] == LAYOUT_RECORDER_14
        assert found[0]["encoding"] == ENCODING_TEXT
    finally:
        rec.stop()


def test_dropout_holds_last_sample_instead_of_zeros():
    """A gap must not hand the engine zeros: next to a 50 000 uV offset
    that is a giant step that fires every voice at once."""
    port = _free_port()
    src = UnicornLiveSource(port=port, host="127.0.0.1")
    src.open()
    rec = FakeRecorder(port).start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not src.has_signal:
            src.next_packet()
        src.next_packet()
        rec.stop()
        time.sleep(0.3)
        with src._lock:
            src._buf.clear()           # whatever was in flight is gone
        pkt = src.next_packet()
        assert pkt is not None
        # Held at the electrode offset, not dropped to zero...
        assert np.all(np.abs(pkt.eeg[:, 0] - DC[0]) < 400.0)
        # ...and honestly marked invalid.
        assert np.all(pkt.validation == 0.0)
    finally:
        rec.stop()
        src.close()


# --------------------------------------------------------------------------- #
# Analysis on DC-coupled, mains-contaminated input                            #
# --------------------------------------------------------------------------- #

def _raw_window(alpha_uv=20.0, mains_uv=150.0, n=FS):
    t = np.arange(n) / FS
    return (DC[None, :]
            + alpha_uv * np.sin(2 * np.pi * 10.0 * t)[:, None]
            + mains_uv * np.sin(2 * np.pi * 50.0 * t)[:, None])


def test_dc_offset_no_longer_swamps_the_bands():
    bp = compute_band_powers(_raw_window(mains_uv=0.0), FS)
    alpha = bp.channel_mean("alpha")
    delta = bp.channel_mean("delta")
    # A 20 uV 10 Hz sine carries 200 uV^2; the 50 000 uV offset must not
    # leak into delta any more.
    assert 150.0 < alpha < 250.0, alpha
    assert delta < alpha * 0.05, (delta, alpha)


def test_dc_leakage_was_real_without_detrend():
    """Documents the failure mode the fix removes."""
    bp = compute_band_powers(_raw_window(mains_uv=0.0), FS, detrend=False)
    assert bp.channel_mean("delta") > 1000 * bp.channel_mean("alpha")


def test_mains_hum_is_excluded_from_gamma():
    clean = compute_band_powers(_raw_window(mains_uv=0.0), FS)
    hummy = compute_band_powers(_raw_window(mains_uv=150.0), FS)
    # 150 uV of 50 Hz = 11 250 uV^2; after the notch gamma barely moves.
    assert hummy.channel_mean("gamma") < clean.channel_mean("gamma") + 5.0
    unnotched = compute_band_powers(_raw_window(mains_uv=150.0), FS,
                                    mains_hz=0.0)
    assert unnotched.channel_mean("gamma") > 5000.0


def test_engine_bridges_samples_flagged_lost():
    """VALID = 0 samples (Bluetooth loss) are held, not analysed as data."""
    eng = MappingEngine(MappingConfig(sample_rate_hz=FS))
    good = np.zeros((25, PACKET_VALUES), dtype=np.float32)
    good[:, :8] = 50000.0
    good[:, VALIDATION_INDEX] = 1.0
    eng.process_packet(UnicornPacket(samples=good, timestamp=0.0,
                                     sample_rate=FS))
    lost = np.zeros((25, PACKET_VALUES), dtype=np.float32)   # zeros...
    lost[:, VALIDATION_INDEX] = 0.0                          # ...flagged lost
    eng.process_packet(UnicornPacket(samples=lost, timestamp=0.1,
                                     sample_rate=FS))
    held = eng._buffer[-1]
    assert np.all(held == 50000.0), "lost samples must hold the last value"


def test_real_format_stream_drives_the_midi_pipeline():
    """End to end: Recorder-format text over UDP, DC offsets and mains hum
    included, through the unchanged translation chain, out as MIDI."""

    class _Midi:
        def __init__(self):
            self.notes = 0

        def send_note_on(self, note, velocity=96, channel=0):
            self.notes += 1

        def send_note_off(self, note, channel=0):
            pass

        def send_control_change(self, control, value, channel=0):
            pass

        def send_pitch_bend(self, value, channel=0):
            pass

    port = _free_port()
    src = UnicornLiveSource(port=port, host="127.0.0.1")
    src.open()
    rec = FakeRecorder(port, alpha_uv=30.0).start()
    midi = _Midi()
    eng = RealtimeEngine(source=src, midi=midi,
                         mapping_config=MappingConfig(sample_rate_hz=FS))
    for s in rich_vocabulary_preset().as_list():
        eng.mapper.add_strategy(s)
    try:
        eng.start()
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline and midi.notes < 3:
            time.sleep(0.1)
        assert eng.is_running
        assert midi.notes >= 3, f"only {midi.notes} notes from a real stream"
        assert src.stats["malformed"] == 0
    finally:
        eng.stop()
        rec.stop()
        src.close()
