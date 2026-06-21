from collections import Counter

import numpy as np

from sogno_cane.eeg.profiles import HUMAN_PROFILE
from sogno_cane.eeg.simulator import EEGSimulator
from sogno_cane.midi.mapper import MappingConfig, MappingEngine
from sogno_cane.midi.presets import rich_vocabulary_preset
from sogno_cane.midi.strategies.per_channel_band import (
    PerChannelBandStrategy,
    PerChannelVoice,
)


def _run(strategies, n_packets=200, seed=2, sr=250):
    eng = MappingEngine(MappingConfig(sample_rate_hz=sr))
    for s in strategies:
        eng.add_strategy(s)
    sim = EEGSimulator(profile=HUMAN_PROFILE, seed=seed)
    events = []
    for _ in range(n_packets):
        events.extend(eng.process_packet(sim.next_packet()))
    events.extend(eng.shutdown())
    return events


def test_events_are_well_formed():
    events = _run(rich_vocabulary_preset().as_list())
    for e in events:
        assert e.kind in ("note_on", "note_off", "cc", "pitchbend")
        assert 0 <= e.channel <= 15
        assert 0 <= e.note <= 127
        assert 0 <= e.velocity <= 127
        assert 0 <= e.control <= 127
        assert 0 <= e.value <= 127


def test_pacing_reduces_density():
    """A long min interval must yield far fewer notes than a short one."""
    def count_notes(interval):
        v = PerChannelVoice(
            eeg_channel=0, midi_channel=0, band="alpha",
            min_interval_seconds=interval, min_duration_seconds=0.0,
            change_threshold_norm=0.0,
        )
        evs = _run([PerChannelBandStrategy(voices=[v])], n_packets=200)
        return sum(1 for e in evs if e.kind == "note_on")

    fast = count_notes(0.2)
    slow = count_notes(3.0)
    assert slow < fast
    assert slow <= 20 / 3.0 + 3  # ~ 20 s / 3 s interval


def test_min_interval_is_respected():
    v = PerChannelVoice(
        eeg_channel=0, midi_channel=0, band="alpha",
        min_interval_seconds=2.0, min_duration_seconds=0.0,
        change_threshold_norm=0.0,
    )
    eng = MappingEngine(MappingConfig(sample_rate_hz=250))
    eng.add_strategy(PerChannelBandStrategy(voices=[v]))
    sim = EEGSimulator(profile=HUMAN_PROFILE, seed=5)
    on_times = []
    for _ in range(200):
        for e in eng.process_packet(sim.next_packet()):
            if e.kind == "note_on":
                on_times.append(eng.elapsed_seconds)
    gaps = [b - a for a, b in zip(on_times, on_times[1:])]
    assert all(g >= 2.0 - 1e-6 for g in gaps)


def test_shutdown_flushes_note_offs():
    v = PerChannelVoice(
        eeg_channel=0, midi_channel=0, band="alpha",
        min_interval_seconds=0.0, change_threshold_norm=0.0,
        min_duration_seconds=0.0,
    )
    strat = PerChannelBandStrategy(voices=[v])
    eng = MappingEngine(MappingConfig(sample_rate_hz=250))
    eng.add_strategy(strat)
    sim = EEGSimulator(profile=HUMAN_PROFILE, seed=1)
    on = off = 0
    for _ in range(60):
        for e in eng.process_packet(sim.next_packet()):
            on += e.kind == "note_on"
            off += e.kind == "note_off"
    for e in eng.shutdown():
        off += e.kind == "note_off"
    # Every note that turned on is eventually turned off.
    assert off >= on - 1 and on > 0


def test_threshold_trigger_balanced_on_off():
    from sogno_cane.midi.strategies.threshold_trigger import (
        ThresholdTriggerStrategy,
    )
    strat = ThresholdTriggerStrategy(
        band="alpha", threshold_uv2=1.0, release_ratio=0.6, channel=9, seed=0,
    )
    events = _run([strat], n_packets=200)
    kinds = Counter(e.kind for e in events)
    # On/off must be roughly balanced (no stuck notes after shutdown flush).
    assert abs(kinds["note_on"] - kinds["note_off"]) <= 1


def test_clip_launcher_respects_lockout():
    from sogno_cane.midi.strategies.clip_launcher import (
        ClipLauncherStrategy,
        ClipRule,
    )
    rule = ClipRule(
        eeg_channel=0, band="alpha", midi_note=36, midi_channel=11,
        threshold_uv2=0.0, min_interval_seconds=2.0,
    )
    eng = MappingEngine(MappingConfig(sample_rate_hz=250))
    eng.add_strategy(ClipLauncherStrategy(rules=[rule]))
    sim = EEGSimulator(profile=HUMAN_PROFILE, seed=1)
    fire_times = []
    for _ in range(200):
        for e in eng.process_packet(sim.next_packet()):
            if e.kind == "note_on":
                fire_times.append(eng.elapsed_seconds)
    gaps = [b - a for a, b in zip(fire_times, fire_times[1:])]
    assert all(g >= 2.0 - 1e-6 for g in gaps)
