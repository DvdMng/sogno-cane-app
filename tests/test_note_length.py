"""Note length (gate) — fix for "notes stay held ~2 s and block envelopes".

Covers:
* strategies stamp ``duration`` on note_on events (and 0 = legato mode),
* the wall-clock :class:`NoteOffScheduler` actually sends the note_off
  after the requested delay (millisecond scale),
* the realtime engine schedules + flushes gated notes,
* the new field round-trips through config save/load and old configs
  (without the field) still load.
"""
import time

import numpy as np

from sogno_cane.core.engine import NoteOffScheduler, RealtimeEngine
from sogno_cane.eeg.profiles import HUMAN_PROFILE
from sogno_cane.eeg.simulator import EEGSimulator
from sogno_cane.midi.mapper import MappingConfig, MappingEngine, MappingEvent
from sogno_cane.midi.presets import rich_vocabulary_preset
from sogno_cane.midi.strategies.band_to_note import BandToNoteStrategy
from sogno_cane.midi.strategies.markov_generative import (
    MarkovGenerativeStrategy,
)
from sogno_cane.midi.strategies.per_channel_band import (
    PerChannelBandStrategy,
    PerChannelVoice,
)
from sogno_cane.midi.strategies.threshold_trigger import (
    ThresholdTriggerStrategy,
)
from sogno_cane import config as appconfig


def _run(strategies, n_packets=300, seed=2, sr=250):
    eng = MappingEngine(MappingConfig(sample_rate_hz=sr))
    for s in strategies:
        eng.add_strategy(s)
    sim = EEGSimulator(profile=HUMAN_PROFILE, seed=seed)
    events = []
    for _ in range(n_packets):
        events.extend(eng.process_packet(sim.next_packet()))
    return events


# --------------------------------------------------------------------------- #
# Strategies stamp the duration                                               #
# --------------------------------------------------------------------------- #

def test_default_note_length_is_15ms():
    assert PerChannelVoice(eeg_channel=0, midi_channel=0) \
        .note_length_seconds == 0.015
    assert BandToNoteStrategy().note_length_seconds == 0.015
    assert MarkovGenerativeStrategy().note_length_seconds == 0.015
    assert ThresholdTriggerStrategy().note_length_seconds == 0.015


def test_note_ons_carry_duration():
    events = _run(rich_vocabulary_preset().as_list())
    ons = [e for e in events if e.kind == "note_on"]
    assert ons, "expected some notes from the default preset"
    gated = [e for e in ons if e.duration > 0.0]
    # Everything except the clip launcher (which releases next window) is
    # gated at the 15 ms default.
    clip_channels = {11}
    for e in ons:
        if e.channel in clip_channels:
            continue
        assert e.duration == 0.015, (e.channel, e.duration)
    assert gated


def test_zero_means_legato():
    strat = PerChannelBandStrategy.default_for_n_channels(8)
    for v in strat.voices:
        v.note_length_seconds = 0.0
    events = _run([strat])
    for e in events:
        if e.kind == "note_on":
            assert e.duration == 0.0


# --------------------------------------------------------------------------- #
# Scheduler                                                                   #
# --------------------------------------------------------------------------- #

def test_scheduler_sends_note_off_after_delay():
    sent = []
    sched = NoteOffScheduler(lambda ev: sent.append((time.monotonic(), ev)))
    t0 = time.monotonic()
    sched.schedule(0.015, channel=3, note=60)
    deadline = time.monotonic() + 2.0
    while not sent and time.monotonic() < deadline:
        time.sleep(0.001)
    assert sent, "scheduled note_off was never sent"
    t_sent, ev = sent[0]
    assert ev.kind == "note_off" and ev.note == 60 and ev.channel == 3
    elapsed = t_sent - t0
    assert 0.010 <= elapsed < 0.25, f"note_off after {elapsed*1000:.1f} ms"
    sched.stop(flush=False)


def test_scheduler_revives_after_stop():
    """RealtimeEngine.start() after stop() must not silently drop gated
    releases (review finding: one-shot scheduler reintroduced stuck notes)."""
    sent = []
    sched = NoteOffScheduler(lambda ev: sent.append(ev.note))
    sched.schedule(0.005, channel=0, note=60)
    deadline = time.monotonic() + 2.0
    while 60 not in sent and time.monotonic() < deadline:
        time.sleep(0.001)
    sched.stop(flush=True)
    # New work after stop() must be accepted and served again.
    sched.schedule(0.005, channel=0, note=61)
    deadline = time.monotonic() + 2.0
    while 61 not in sent and time.monotonic() < deadline:
        time.sleep(0.001)
    assert 61 in sent, "scheduler must revive after stop()"
    sched.stop(flush=True)


def test_scheduler_supersedes_same_note_restrike():
    """A restrike of the same (channel, note) must cancel the older pending
    note_off, otherwise the stale off truncates the new note."""
    sent = []
    sched = NoteOffScheduler(lambda ev: sent.append(ev.note))
    sched.schedule(0.040, channel=0, note=60)   # old instance
    sched.schedule(5.0, channel=0, note=60)     # restrike, longer gate
    time.sleep(0.12)
    assert sent == [], "stale note_off must have been superseded"
    sched.stop(flush=True)
    assert sent == [60], "exactly one (the newest) release remains"


def test_engine_stop_parks_scheduler_thread():
    """After stop() the worker thread must exit (review finding: leak per
    finished playback)."""
    midi = _FakeMidi()
    eng = RealtimeEngine(midi=midi)
    eng._dispatch_event(MappingEvent(
        kind="note_on", channel=0, note=70, velocity=90, duration=30.0,
    ))
    worker = eng._noteoff._thread
    assert worker is not None and worker.is_alive()
    eng.stop()
    worker.join(timeout=2.0)
    assert not worker.is_alive(), "worker thread must exit on engine.stop()"


def test_scheduler_orders_and_flushes():
    sent = []
    sched = NoteOffScheduler(lambda ev: sent.append(ev.note))
    sched.schedule(0.030, channel=0, note=62)
    sched.schedule(0.010, channel=0, note=60)
    deadline = time.monotonic() + 2.0
    while len(sent) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert sent == [60, 62]
    # Pending notes are flushed on stop.
    sched.schedule(60.0, channel=0, note=70)
    sched.stop(flush=True)
    assert 70 in sent


# --------------------------------------------------------------------------- #
# Engine integration                                                          #
# --------------------------------------------------------------------------- #

class _FakeMidi:
    def __init__(self):
        self.msgs = []

    def send_note_on(self, note, velocity=96, channel=0):
        self.msgs.append(("on", note, channel, time.monotonic()))

    def send_note_off(self, note, channel=0):
        self.msgs.append(("off", note, channel, time.monotonic()))

    def send_control_change(self, control, value, channel=0):
        pass

    def send_pitch_bend(self, value, channel=0):
        pass


def test_engine_schedules_gated_note_off():
    midi = _FakeMidi()
    eng = RealtimeEngine(midi=midi)
    ev = MappingEvent(
        kind="note_on", channel=2, note=64, velocity=100, duration=0.015,
    )
    eng._dispatch_event(ev)
    deadline = time.monotonic() + 2.0
    while (
        not any(m[0] == "off" for m in midi.msgs)
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    ons = [m for m in midi.msgs if m[0] == "on"]
    offs = [m for m in midi.msgs if m[0] == "off"]
    assert len(ons) == 1 and len(offs) == 1
    assert offs[0][1] == 64 and offs[0][2] == 2
    gap = offs[0][3] - ons[0][3]
    assert 0.010 <= gap < 0.25, f"note released after {gap*1000:.1f} ms"
    eng.stop()


def test_engine_stop_flushes_pending_gated_notes():
    midi = _FakeMidi()
    eng = RealtimeEngine(midi=midi)
    eng._dispatch_event(MappingEvent(
        kind="note_on", channel=1, note=50, velocity=90, duration=120.0,
    ))
    eng.stop()   # long gate still pending -> must be flushed now
    offs = [m for m in midi.msgs if m[0] == "off" and m[1] == 50]
    assert offs, "engine.stop() must release pending gated notes"


def test_engine_legato_note_not_scheduled():
    midi = _FakeMidi()
    eng = RealtimeEngine(midi=midi)
    eng._dispatch_event(MappingEvent(
        kind="note_on", channel=0, note=51, velocity=90, duration=0.0,
    ))
    time.sleep(0.05)
    assert not [m for m in midi.msgs if m[0] == "off"]
    eng.stop()


# --------------------------------------------------------------------------- #
# Config round-trip / backward compatibility                                  #
# --------------------------------------------------------------------------- #

def test_note_length_roundtrips_through_config():
    b1 = rich_vocabulary_preset()
    for v in b1.per_channel_band.voices:
        v.note_length_seconds = 0.25
    b1.markov.note_length_seconds = 0.5
    b1.threshold.note_length_seconds = 0.0
    data = appconfig.serialize_bundle(b1)

    b2 = rich_vocabulary_preset()
    appconfig.apply_bundle_config(b2, data)
    assert all(
        v.note_length_seconds == 0.25 for v in b2.per_channel_band.voices
    )
    assert b2.markov.note_length_seconds == 0.5
    assert b2.threshold.note_length_seconds == 0.0


def test_old_config_without_note_length_keeps_default():
    """A 0.3/0.4 config (no note_length_seconds key) must load cleanly and
    leave the new 15 ms default in place."""
    b1 = rich_vocabulary_preset()
    data = appconfig.serialize_bundle(b1)

    def strip(d):
        if isinstance(d, dict):
            return {
                k: strip(v) for k, v in d.items()
                if k != "note_length_seconds"
            }
        if isinstance(d, list):
            return [strip(x) for x in d]
        return d

    old_data = strip(data)
    b2 = rich_vocabulary_preset()
    appconfig.apply_bundle_config(b2, old_data)
    assert all(
        v.note_length_seconds == 0.015 for v in b2.per_channel_band.voices
    )
    assert b2.markov.note_length_seconds == 0.015
