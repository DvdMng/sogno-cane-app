"""Regression tests for bugs found by the adversarial code review."""
import time

import numpy as np

from sogno_cane.core.engine import RealtimeEngine
from sogno_cane.core.sources import ArrayPlaybackSource
from sogno_cane.eeg.bands import compute_band_powers
from sogno_cane.midi.context import WindowContext
from sogno_cane.midi.mapper import MappingConfig, MappingEngine
from sogno_cane.midi.output import MidiOutput
from sogno_cane.midi.strategies.markov_generative import MarkovGenerativeStrategy
from sogno_cane.midi.strategies.per_channel_band import (
    PerChannelBandStrategy,
    PerChannelVoice,
)


class _CaptureMidi(MidiOutput):
    def __init__(self):
        super().__init__()
        self.notes_on = 0
        self.notes_off = 0

    def _send_raw(self, msg):
        status = msg[0] & 0xF0
        if status == 0x90 and msg[2] > 0:
            self.notes_on += 1
        elif status == 0x80 or (status == 0x90 and msg[2] == 0):
            self.notes_off += 1


def test_playback_finish_flushes_notes_no_stuck():
    """When non-looping playback ends, held notes must be released."""
    eeg = np.random.default_rng(0).normal(0, 30, size=(750, 8)).astype(np.float32)
    src = ArrayPlaybackSource(eeg, 250, packet_samples=25, loop=False)
    midi = _CaptureMidi()
    voice = PerChannelVoice(
        eeg_channel=0, midi_channel=0, band="alpha",
        min_interval_seconds=0.0, min_duration_seconds=0.0,
        change_threshold_norm=0.0, sustain=True,
    )
    eng = RealtimeEngine(source=src, midi=midi)
    eng.mapper.add_strategy(PerChannelBandStrategy(voices=[voice]))
    eng.set_time_scale(0.005)
    eng.start()
    deadline = time.monotonic() + 3.0
    while eng.is_running and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.1)
    assert not eng.is_running
    # No stuck notes: every note_on was matched by a note_off.
    assert midi.notes_on > 0
    assert midi.notes_off == midi.notes_on


def test_markov_no_duplicate_simultaneous_notes():
    """A Markov phrase must never emit the same pitch twice without release."""
    strat = MarkovGenerativeStrategy(
        scale="dorian", root="D", max_notes_per_window=8, channel=8, seed=1,
    )
    # Build high-energy bands so phrases are dense (n_notes large).
    fs = 250
    t = np.arange(fs) / fs
    eeg = np.tile((50 * np.sin(2 * np.pi * 20 * t))[:, None], (1, 8))
    bands = compute_band_powers(eeg, fs)
    ctx = WindowContext(0.0, 0.1, fs, fs)
    seen_dupe = False
    for k in range(50):
        ctx = WindowContext(t_seconds=k * 5.0, hop_seconds=0.1,
                            sample_rate_hz=fs, window_samples=fs)
        evs = strat.on_window(eeg, bands, ctx)
        ons = [e.note for e in evs if e.kind == "note_on"]
        if len(ons) != len(set(ons)):
            seen_dupe = True
    assert not seen_dupe, "Markov emitted duplicate simultaneous note_on"


def test_loader_millisecond_rate(tmp_path):
    """A millisecond time column at 250 Hz must infer ~250 Hz, not 0.25 Hz."""
    from sogno_cane.io.loaders import load_eeg
    rows = ["time,Fp1,Fp2"]
    for i in range(40):
        rows.append(f"{i*4},{i},{i*2}")   # 4 ms steps -> 250 Hz
    p = tmp_path / "ms.csv"
    p.write_text("\n".join(rows))
    lo = load_eeg(str(p))
    assert abs(lo.sample_rate_hz - 250.0) < 1.0
    assert lo.n_channels == 2


def test_loader_seconds_rate(tmp_path):
    from sogno_cane.io.loaders import load_eeg
    rows = ["time,a,b"]
    for i in range(20):
        rows.append(f"{i*0.004:.6f},{i},{i}")  # 0.004 s -> 250 Hz
    p = tmp_path / "s.csv"
    p.write_text("\n".join(rows))
    lo = load_eeg(str(p))
    assert abs(lo.sample_rate_hz - 250.0) < 1.0


def test_npz_1d_eeg(tmp_path):
    from sogno_cane.io.loaders import load_eeg
    p = tmp_path / "single.npz"
    np.savez(str(p), eeg=np.arange(100, dtype=np.float64))
    lo = load_eeg(str(p))     # must not raise
    assert lo.data.shape == (100, 1)


def test_edf_truncated_recovers(tmp_path):
    """A truncated EDF with a declared record count loads its intact records."""
    from tests.test_io_edf import _write_edf
    from sogno_cane.io.edf import read_edf
    fs = 100
    ch = (np.arange(500) % 50).astype(np.int16)  # 5 records of 100
    path = tmp_path / "full.edf"
    _write_edf(str(path), [("Fz", ch)], fs)
    raw = path.read_bytes()
    # Header is 256*(1+1)=512 bytes; keep header + 3 of 5 records.
    rec_bytes = fs * 2
    truncated = raw[: 512 + 3 * rec_bytes]
    tp = tmp_path / "trunc.edf"
    tp.write_bytes(truncated)
    edf = read_edf(str(tp))        # must not raise
    assert edf.data.shape[0] == 300   # 3 intact records recovered


def test_bands_odd_window_parseval():
    """Integrated PSD must match signal power for an ODD-length window too."""
    fs = 251  # odd window when 1 s
    n = 251
    t = np.arange(n) / fs
    eeg = (10.0 * np.sin(2 * np.pi * 10 * t))[:, None]
    bp = compute_band_powers(eeg, fs)
    total = float(bp.total_power()[0])
    assert 40.0 < total < 60.0   # ~50 uV^2; was ~2x low for odd before fix


def test_sub_hz_rate_not_coerced():
    src = ArrayPlaybackSource(np.zeros((100, 8), np.float32), 0.4, loop=False)
    assert abs(src.sample_rate_hz - 0.4) < 1e-9


def test_export_filename_sanitised_for_windows():
    """Recording names carry colons (timestamps); the export filename must
    not, or the Windows save fails with OSError [Errno 22]."""
    import pytest
    pytest.importorskip("PySide6")
    from sogno_cane.ui.archive_panel import _safe_filename
    name = "HUMAN 2026-06-21 13:50:22"
    safe = _safe_filename(name)
    for bad in '<>:"/\\|?*':
        assert bad not in safe
    assert " " not in safe
    assert safe  # non-empty


def test_export_csv_to_sanitised_path_succeeds(tmp_path):
    """End-to-end: a colon-bearing recording name exports once sanitised."""
    import pytest
    pytest.importorskip("PySide6")
    from sogno_cane.io.archive import Archive
    from sogno_cane.io.recording import Recording, RecordingMeta
    from sogno_cane.ui.archive_panel import _safe_filename
    arc = Archive(root=str(tmp_path / "rec"))
    meta = RecordingMeta(
        rec_id="", name="HUMAN 2026-06-21 13:50:22", profile="HUMAN",
        channel_names=[f"c{i}" for i in range(8)], sample_rate_hz=250.0,
        n_samples=20, created="",
    )
    rid = arc.save(Recording(eeg=np.zeros((20, 8)), meta=meta))
    dest = tmp_path / f"{_safe_filename(meta.name)}.csv"
    arc.export_csv(rid, str(dest))   # must not raise
    assert dest.exists() and dest.stat().st_size > 0
