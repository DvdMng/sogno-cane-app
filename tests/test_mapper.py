import numpy as np

from sogno_cane.eeg.bands import compute_band_powers
from sogno_cane.eeg.simulator import EEGSimulator
from sogno_cane.eeg.profiles import HUMAN_PROFILE
from sogno_cane.midi.context import WindowContext
from sogno_cane.midi.mapper import MappingConfig, MappingEngine, MappingEvent


class _CtxRecorder:
    """3-arg strategy that records the context times it sees."""
    name = "ctx"
    enabled = True
    channel = 0

    def __init__(self):
        self.times = []

    def on_window(self, eeg, bands, ctx):
        self.times.append(ctx.t_seconds)
        return []

    def shutdown(self):
        return []


class _LegacyTwoArg:
    """Old-style 2-arg strategy must still work."""
    name = "legacy"
    enabled = True
    channel = 0

    def __init__(self):
        self.calls = 0

    def on_window(self, eeg, bands):
        self.calls += 1
        return [MappingEvent(kind="cc", channel=0, control=1, value=10)]

    def shutdown(self):
        return []


def _feed(engine, n_packets=50, seed=1):
    sim = EEGSimulator(profile=HUMAN_PROFILE, seed=seed)
    out = []
    for _ in range(n_packets):
        out.extend(engine.process_packet(sim.next_packet()))
    return out


def test_config_window_hop_samples():
    cfg = MappingConfig(sample_rate_hz=250, window_seconds=1.0, hop_seconds=0.1)
    assert cfg.window_samples == 250
    assert cfg.hop_samples == 25


def test_context_time_advances():
    eng = MappingEngine(MappingConfig(sample_rate_hz=250))
    rec = _CtxRecorder()
    eng.add_strategy(rec)
    _feed(eng, n_packets=50)
    assert len(rec.times) > 0
    # Times are strictly increasing and ~hop spaced.
    assert all(b > a for a, b in zip(rec.times, rec.times[1:]))
    assert rec.times[-1] <= eng.elapsed_seconds + 1e-9


def test_legacy_two_arg_strategy_supported():
    eng = MappingEngine(MappingConfig(sample_rate_hz=250))
    legacy = _LegacyTwoArg()
    eng.add_strategy(legacy)
    events = _feed(eng, n_packets=50)
    assert legacy.calls > 0
    assert any(e.kind == "cc" for e in events)


def test_strategy_exception_is_isolated():
    class Boom:
        name = "boom"; enabled = True; channel = 0
        def on_window(self, eeg, bands, ctx):
            raise RuntimeError("kaboom")
        def shutdown(self):
            return []

    eng = MappingEngine(MappingConfig(sample_rate_hz=250))
    eng.add_strategy(Boom())
    good = _CtxRecorder()
    eng.add_strategy(good)
    _feed(eng, n_packets=30)   # must not raise
    assert len(good.times) > 0


def test_no_events_before_full_window():
    eng = MappingEngine(MappingConfig(sample_rate_hz=250, window_seconds=1.0))
    rec = _CtxRecorder()
    eng.add_strategy(rec)
    sim = EEGSimulator(profile=HUMAN_PROFILE, seed=1)
    # 5 packets * 25 samples = 125 < 250 window -> no evaluation yet.
    for _ in range(4):
        eng.process_packet(sim.next_packet())
    assert rec.times == []
