"""No note left on at the synth (0.9.6).

Field report: "every now and then a note stays on until Ableton is
restarted". What could leave one on, and what PANIC could not undo:

* PANIC, closing the port and the reconnect after a stall sent only All
  Notes Off (CC 123), which Ableton and many instruments ignore: a note
  whose note-off was lost (stalled loopMIDI port, port closed mid-note)
  stayed on for good;
* a short note struck again while its note-off was still pending went out
  as note-on, note-on, note-off: a synth that stacks voices keeps one;
* notes a voice still held when a run ended without stop() were forgotten
  at the next start, never released.
"""
import time

import numpy as np

from sogno_cane.core.engine import RealtimeEngine
from sogno_cane.core.sources import ArrayPlaybackSource
from sogno_cane.midi import output as out_mod
from sogno_cane.midi.mapper import MappingEvent
from sogno_cane.midi.output import RECONNECT_AFTER_FAILURES, MidiOutput
from sogno_cane.midi.presets import rich_vocabulary_preset


class _Wire:
    """What actually reached the other end of the port(s)."""

    def __init__(self):
        self.messages = []
        self.fail = False

    def left_on(self):
        """Notes switched on and never off (CC 123 does not count)."""
        on, doubled = set(), 0
        for m in self.messages:
            kind, key = m[0] & 0xF0, (m[0] & 0x0F, m[1])
            if kind == 0x90 and m[2] > 0:
                doubled += key in on
                on.add(key)
            elif kind == 0x80 or (kind == 0x90 and m[2] == 0):
                on.discard(key)
        return on, doubled


class _Port:
    def __init__(self, wire):
        self.wire = wire

    def send_message(self, msg):
        if self.wire.fail:
            raise RuntimeError("MidiOutWinMM::sendMessage: error sending")
        self.wire.messages.append(list(msg))

    def close_port(self):
        pass


def _output(wire, name=None):
    out = MidiOutput()
    out._port = _Port(wire)
    out._port_name = name
    return out


def _rtmidi_on(wire):
    """rtmidi stand-in whose ports all write to ``wire``."""
    class _Backend(_Port):
        def get_port_count(self):
            return 1

        def get_port_name(self, i):
            return "loopMIDI Port 1"

        def open_port(self, idx, name):
            pass

    class _Rtmidi:
        @staticmethod
        def MidiOut():
            return _Backend(wire)

    return _Rtmidi


# --------------------------------------------------------------------------- #
# The output                                                                  #
# --------------------------------------------------------------------------- #

def test_a_note_is_never_switched_on_twice_in_a_row():
    wire = _Wire()
    out = _output(wire)
    out.send_note_on(60, 100, 2)
    out.send_note_on(60, 90, 2)
    assert wire.messages == [[0x92, 60, 100], [0x82, 60, 0], [0x92, 60, 90]]
    assert out.sounding == {(2, 60)}


def test_a_lost_note_off_is_sent_again_on_release():
    wire = _Wire()
    out = _output(wire)
    out.send_note_on(64, 100, 0)
    wire.fail = True
    out.send_note_off(64, 0)                     # lost
    wire.fail = False
    assert out.sounding == {(0, 64)}
    assert out.release_all() == 1
    assert wire.left_on() == (set(), 0)


def test_closing_the_port_switches_off_what_sounds():
    wire = _Wire()
    out = _output(wire)
    out.send_note_on(60, 100, 0)
    out.send_note_on(67, 100, 8)
    out.close()
    assert wire.left_on() == (set(), 0)
    assert not out.is_open


def test_reconnect_resends_the_note_offs_lost_in_the_stall(monkeypatch):
    wire = _Wire()
    monkeypatch.setattr(out_mod, "rtmidi", _rtmidi_on(wire))
    out = _output(wire, name="loopMIDI Port")
    out.send_note_on(60, 100, 0)
    out.send_note_on(62, 100, 1)
    wire.fail = True                             # loopMIDI stalls
    out.send_note_off(60, 0)
    out.send_note_off(62, 1)
    for _ in range(RECONNECT_AFTER_FAILURES):
        out.send_control_change(20, 64, 12)
    wire.fail = False                            # the port is back
    out.send_control_change(20, 64, 12)          # triggers the reconnect
    for _ in range(RECONNECT_AFTER_FAILURES):
        out.send_control_change(20, 64, 12)
    assert out.stats["reconnects"] == 1
    assert wire.left_on() == (set(), 0)
    assert out.sounding == set()


def test_panic_sends_a_note_off_for_every_note_of_every_channel():
    wire = _Wire()
    out = _output(wire)
    out.send_note_on(60, 100, 0)
    wire.messages.clear()
    out.panic()
    offs = {(m[0] & 0x0F, m[1]) for m in wire.messages if m[0] & 0xF0 == 0x80}
    assert len(offs) == 16 * 128
    ccs = {(m[0] & 0x0F, m[1]) for m in wire.messages if m[0] & 0xF0 == 0xB0}
    assert ccs == {(ch, cc) for ch in range(16) for cc in (64, 120, 123)}
    assert out.sounding == set()


def test_panic_clears_a_note_this_output_never_knew_about():
    """A note left on by a crash: nothing remembers it, PANIC still ends it."""
    wire = _Wire()
    wire.messages.append([0x90, 57, 100])        # from a previous session
    _output(wire).panic()
    assert wire.left_on() == (set(), 0)


# --------------------------------------------------------------------------- #
# The engine                                                                  #
# --------------------------------------------------------------------------- #

def _engine(out, seconds=6.0, seed=0):
    eeg = np.random.default_rng(seed).normal(0, 20, (int(seconds * 250), 8))
    src = ArrayPlaybackSource(eeg.astype(np.float32), 250, packet_samples=25,
                              loop=True)
    return RealtimeEngine(source=src, midi=out)


def test_a_short_note_struck_again_is_closed_first():
    wire = _Wire()
    eng = _engine(_output(wire))
    for _ in range(2):
        eng._dispatch_event(MappingEvent(kind="note_on", channel=3, note=50,
                                         velocity=90, duration=5.0))
    eng.stop()
    assert wire.messages[:3] == [[0x93, 50, 90], [0x83, 50, 0],
                                 [0x93, 50, 90]]
    assert wire.left_on() == (set(), 0)


def test_start_releases_notes_a_previous_run_left_behind():
    wire = _Wire()
    out = _output(wire)
    b = rich_vocabulary_preset()
    b.markov._held.append((8, 62))               # a run that never stopped
    out.send_note_on(62, 80, 8)
    eng = _engine(out)
    eng.mapper.add_strategy(b.markov)
    eng.set_time_scale(50.0)                     # no packet during the test
    eng.start()
    eng.stop()
    assert (8, 62) not in wire.left_on()[0]


def test_stop_switches_off_even_a_note_its_voice_forgot():
    wire = _Wire()
    out = _output(wire)
    out.send_note_on(45, 80, 5)                  # nobody remembers it
    eng = _engine(out)
    eng.set_time_scale(50.0)
    eng.start()
    eng.stop()
    assert wire.left_on() == (set(), 0)


def test_live_edits_and_a_stalled_port_leave_nothing_on(monkeypatch):
    """Every voice on, HOLD and short notes mixed, channels changed and
    voices switched off while playing, and the port stalling for a while:
    at the end nothing is on and no note was ever switched on twice."""
    wire = _Wire()
    monkeypatch.setattr(out_mod, "rtmidi", _rtmidi_on(wire))
    out = _output(wire, name="loopMIDI Port")
    b = rich_vocabulary_preset()
    b.threshold.note_length_seconds = 0.0
    b.markov.note_length_seconds = 0.0
    for v in b.per_channel_band.voices[::2]:
        v.note_length_seconds = 0.0
    count = [0]

    def edit(_packet):
        count[0] += 1
        k = count[0]
        if k == 60:
            b.per_channel_band.voices[0].midi_channel = 14
            b.markov.channel = 15
        elif k == 90:
            wire.fail = True
        elif k == 110:
            wire.fail = False
        elif k == 140:
            b.per_channel_band.voices[2].enabled = False
            b.markov.enabled = False
        elif k == 170:
            b.markov.enabled = True
            b.threshold.note_length_seconds = 0.015

    eeg = np.random.default_rng(1).normal(0, 20, (60 * 250, 8))
    eng = RealtimeEngine(
        source=ArrayPlaybackSource(eeg.astype(np.float32), 250,
                                   packet_samples=25, loop=False),
        midi=out, on_packet=edit,
    )
    for s in b.as_list():
        eng.mapper.add_strategy(s)
    eng.set_time_scale(0.02)
    eng.start()
    deadline = time.monotonic() + 30.0
    while eng.is_running and time.monotonic() < deadline:
        time.sleep(0.02)
    eng.stop()
    notes = [m for m in wire.messages if m[0] & 0xF0 == 0x90]
    assert len(notes) > 50
    assert wire.left_on() == (set(), 0)
