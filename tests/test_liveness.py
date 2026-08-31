"""Liveness — fix for "after a stop/start (or setup change) MIDI stops",
and for "one of the channels stops transmitting".

Strategy objects outlive engines: the UI re-adds the same live instances to
every new engine, whose simulation clock restarts at zero. Stale gate and
throttle timestamps from the previous run then block emission for up to the
previous session's whole duration. Covered here:

* ``RealtimeEngine.start()`` clears per-run strategy state,
* even without an explicit reset, gates self-heal when the clock runs
  backwards (defence in depth),
* the ``max_silence_seconds`` failsafe retriggers a voice that a stable
  signal would otherwise starve forever via the change-threshold hysteresis.
"""
import time

import numpy as np

from sogno_cane.core.engine import RealtimeEngine
from sogno_cane.eeg.bands import BandPowers
from sogno_cane.eeg.profiles import HUMAN_PROFILE
from sogno_cane.eeg.simulator import EEGSimulator
from sogno_cane.midi.context import WindowContext
from sogno_cane.midi.mapper import MappingConfig, MappingEngine
from sogno_cane.midi.pacing import NoteGate
from sogno_cane.midi.presets import rich_vocabulary_preset
from sogno_cane.midi.strategies.per_channel_band import (
    PerChannelBandStrategy,
    PerChannelVoice,
)


class _FakeMidi:
    def send_note_on(self, note, velocity=96, channel=0):
        pass

    def send_note_off(self, note, channel=0):
        pass

    def send_control_change(self, control, value, channel=0):
        pass

    def send_pitch_bend(self, value, channel=0):
        pass


def _play(engine, strategies, seconds, seed):
    sim = EEGSimulator(profile=HUMAN_PROFILE, seed=seed)
    events = []
    for _ in range(int(seconds * 10)):   # 100 ms packets
        events.extend(engine.process_packet(sim.next_packet()))
    return events


# --------------------------------------------------------------------------- #
# Restart liveness                                                            #
# --------------------------------------------------------------------------- #

def test_voices_speak_promptly_after_engine_restart():
    """Session 1 runs 60 s; the fresh engine (clock back at 0) re-uses the
    same strategies and must produce notes within a few seconds."""
    strategies = rich_vocabulary_preset().as_list()

    eng1 = MappingEngine(MappingConfig(sample_rate_hz=250))
    for s in strategies:
        eng1.add_strategy(s)
    _play(eng1, strategies, seconds=60, seed=5)

    eng2 = MappingEngine(MappingConfig(sample_rate_hz=250))
    for s in strategies:
        eng2.add_strategy(s)
    eng2.reset_strategy_state()   # what RealtimeEngine.start() does
    events = _play(eng2, strategies, seconds=6, seed=6)
    ons = [e for e in events if e.kind == "note_on"]
    assert ons, "voices must not stay mute after an engine restart"


def test_gates_self_heal_even_without_reset():
    """Defence in depth: if nothing calls reset_strategy_state(), the
    clock-regression guard inside the gates must still revive the voices."""
    strategies = rich_vocabulary_preset().as_list()

    eng1 = MappingEngine(MappingConfig(sample_rate_hz=250))
    for s in strategies:
        eng1.add_strategy(s)
    _play(eng1, strategies, seconds=60, seed=7)

    eng2 = MappingEngine(MappingConfig(sample_rate_hz=250))
    for s in strategies:
        eng2.add_strategy(s)
    # NOTE: no reset_strategy_state() here, on purpose.
    events = _play(eng2, strategies, seconds=6, seed=8)
    ons = [e for e in events if e.kind == "note_on"]
    assert ons, "stale timestamps must self-heal when the clock restarts"


def test_realtime_engine_start_resets_strategy_state():
    strat = PerChannelBandStrategy.default_for_n_channels(2)
    # Poison the runtime state as a long previous session would.
    for ch in range(2):
        strat._gate(ch).commit_on(9_999.0, 0.5)
    strat._last_notes[0] = 60
    strat._gated[0] = True

    eng = RealtimeEngine(profile=HUMAN_PROFILE, seed=1, midi=_FakeMidi())
    eng.mapper.add_strategy(strat)
    eng.set_time_scale(100.0)   # slow: no packet lands during the test
    eng.start()
    try:
        assert strat._last_notes == {}
        assert strat._gated == {}
        assert all(
            g._last_change_t == -1.0e9 for g in strat._gates.values()
        )
    finally:
        eng.stop()


def test_note_gate_clock_regression_heals():
    g = NoteGate()
    g.commit_on(3_600.0, 0.5)
    # New engine: clock restarted at ~1 s. Without healing this would block
    # for a full hour of simulated time; healing clears the stale interval
    # reference entirely, so the voice may speak right away.
    assert g.may_change(1.0, 0.9, min_interval=1.5, change_threshold=0.18), \
        "gate must be usable immediately once the clock regression heals"
    # And the healed gate behaves normally afterwards.
    g.commit_on(1.0, 0.9)
    assert not g.may_change(1.5, 0.2, min_interval=1.5, change_threshold=0.18)
    assert g.may_change(2.6, 0.2, min_interval=1.5, change_threshold=0.18)


# --------------------------------------------------------------------------- #
# Starvation failsafe                                                         #
# --------------------------------------------------------------------------- #

def _const_bands(power: float = 100.0, nc: int = 8) -> BandPowers:
    return BandPowers(
        by_band={
            b: np.full(nc, power)
            for b in ("delta", "theta", "alpha", "beta", "gamma")
        },
        sample_rate_hz=250,
        window_samples=250,
    )


def _tick(strat, t):
    ctx = WindowContext(
        t_seconds=t, hop_seconds=0.1, sample_rate_hz=250, window_samples=250,
    )
    return strat.on_window(np.zeros((250, 8)), _const_bands(), ctx)


def test_max_silence_failsafe_retriggers_stuck_voice():
    voice = PerChannelVoice(
        eeg_channel=0, midi_channel=0,
        min_interval_seconds=0.5,
        min_duration_seconds=0.0,
        change_threshold_norm=0.9,     # hysteresis a constant signal never crosses
        max_silence_seconds=2.0,
    )
    strat = PerChannelBandStrategy(voices=[voice])
    ons = []
    for i in range(300):               # 30 s of windows
        t = i * 0.1
        ons.extend(
            e for e in _tick(strat, t) if e.kind == "note_on"
        )
    # First note + a retrigger roughly every 2 s.
    assert len(ons) >= 10, f"expected periodic retriggers, got {len(ons)}"


def test_max_silence_zero_keeps_old_behaviour():
    voice = PerChannelVoice(
        eeg_channel=0, midi_channel=0,
        min_interval_seconds=0.5,
        min_duration_seconds=0.0,
        change_threshold_norm=0.9,
        max_silence_seconds=0.0,       # failsafe OFF
    )
    strat = PerChannelBandStrategy(voices=[voice])
    ons = []
    for i in range(300):
        ons.extend(
            e for e in _tick(strat, i * 0.1) if e.kind == "note_on"
        )
    assert len(ons) == 1, "with the failsafe off a stable signal stays quiet"


def test_starved_retrigger_of_gated_note_has_no_stray_note_off():
    """A same-pitch starved retrigger of a gated note must be a plain
    note_on (the engine owns the releases; a stray note_off would cut it)."""
    voice = PerChannelVoice(
        eeg_channel=0, midi_channel=0,
        min_interval_seconds=0.1,
        min_duration_seconds=0.0,
        change_threshold_norm=0.9,
        max_silence_seconds=1.0,
        note_length_seconds=0.015,
    )
    strat = PerChannelBandStrategy(voices=[voice])
    all_events = []
    for i in range(60):                # 6 s
        all_events.extend(_tick(strat, i * 0.1))
    ons = [e for e in all_events if e.kind == "note_on"]
    offs = [e for e in all_events if e.kind == "note_off"]
    assert len(ons) >= 4
    assert not offs, "gated notes are released by the engine, not the strategy"
    assert all(e.duration == 0.015 for e in ons)


def test_reset_runtime_clears_all_strategies():
    bundle = rich_vocabulary_preset()
    eng = MappingEngine(MappingConfig(sample_rate_hz=250))
    for s in bundle.as_list():
        eng.add_strategy(s)
    _play(eng, bundle.as_list(), seconds=30, seed=9)
    eng.reset_strategy_state()
    assert bundle.per_channel_band._last_notes == {}
    assert bundle.markov._last_phrase_t == -1.0e9
    assert bundle.markov._held == []
    assert bundle.threshold._active_note is None
    assert bundle.threshold._last_trigger_t == -1.0e9
    assert bundle.clips._last_fire_t == {}
    assert bundle.per_channel_cc._last_cc == {}
    assert bundle.coherence._last_t == -1.0e9


def test_stopped_engine_releases_and_restarts_clean():
    """Full engine-level stop → start cycle (what STOP/START does)."""
    bundle = rich_vocabulary_preset()
    eng = RealtimeEngine(profile=HUMAN_PROFILE, seed=3, midi=_FakeMidi())
    for s in bundle.as_list():
        eng.mapper.add_strategy(s)
    eng.set_time_scale(0.02)
    eng.start()
    time.sleep(0.4)
    eng.stop()
    n1 = eng.event_count
    assert n1 > 0
    # Second engine, same bundle (what the UI does on the next START).
    eng2 = RealtimeEngine(profile=HUMAN_PROFILE, seed=4, midi=_FakeMidi())
    for s in bundle.as_list():
        eng2.mapper.add_strategy(s)
    eng2.set_time_scale(0.02)
    eng2.start()
    time.sleep(0.4)
    eng2.stop()
    assert eng2.event_count > 0, "restarted engine must produce events"
