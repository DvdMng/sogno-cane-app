"""MIDI output health — fix for "the app says it is sending but loopMIDI is
frozen".

``MidiOutput`` used to swallow every send error, so a stale port handle
(loopMIDI port re-created, driver hiccup) failed on every message while the
app looked alive. Now failures are counted, surfaced (``stats`` /
``is_stalled``) and after a few back-to-back failures the output re-opens
the same port by name and clears hanging notes.
"""
from sogno_cane.midi import output as out_mod
from sogno_cane.midi.output import (
    RECONNECT_AFTER_FAILURES,
    MidiOutput,
)


class _DeadPort:
    """Behaves like a WinMM handle whose loopMIDI port vanished."""

    def send_message(self, msg):
        raise RuntimeError("MidiOutWinMM::sendMessage: error sending message")

    def close_port(self):
        pass


class _FakeBackendPort:
    """rtmidi.MidiOut() replacement exposing one healthy loopMIDI port."""

    def __init__(self):
        self.opened = None
        self.sent = []

    def get_port_count(self):
        return 1

    def get_port_name(self, i):
        return "loopMIDI Port 1"   # rtmidi's Windows index suffix

    def open_port(self, idx, name):
        self.opened = (idx, name)

    def send_message(self, msg):
        self.sent.append(list(msg))

    def close_port(self):
        pass


def test_send_failures_are_counted_not_swallowed():
    out = MidiOutput()
    out._port = _DeadPort()
    out._port_name = None          # no name -> reconnect not possible
    for _ in range(5):
        out.send_note_on(60, 100, 0)
    st = out.stats
    assert st["failed"] == 5
    assert st["sent"] == 0
    assert st["last_error"], "the last error must be preserved for the UI"


def test_stall_flag_raises_after_consecutive_failures():
    out = MidiOutput()
    out._port = _DeadPort()
    out._port_name = None
    for _ in range(RECONNECT_AFTER_FAILURES):
        out.send_note_on(60, 100, 0)
    assert out.is_stalled
    assert out.stats["consecutive_failures"] == RECONNECT_AFTER_FAILURES


def test_auto_reconnect_swaps_to_live_port_and_cleans_notes(monkeypatch):
    backend = {}

    class _FakeRtmidi:
        @staticmethod
        def MidiOut():
            b = _FakeBackendPort()
            backend["b"] = b
            return b

    monkeypatch.setattr(out_mod, "rtmidi", _FakeRtmidi)
    out = MidiOutput()
    out._port = _DeadPort()
    out._port_name = "loopMIDI Port"

    for _ in range(RECONNECT_AFTER_FAILURES):
        out.send_note_on(60, 100, 0)

    st = out.stats
    assert st["reconnects"] == 1, "must re-open the same port by name"
    b = backend["b"]
    assert b.opened == (0, "SOGNO_CANE")
    # All-Notes-Off (CC 123) on all 16 channels right after the swap, so the
    # synth does not keep notes hanging from before the stall.
    assert len(b.sent) == 16
    assert all(m[1] == 123 and m[2] == 0 for m in b.sent)
    assert not out.is_stalled

    out.send_note_on(61, 100, 2)
    assert out.stats["sent"] == 1
    assert b.sent[-1][0] == 0x92   # note_on, channel 2


def test_failed_reconnect_keeps_retry_loop_alive(monkeypatch):
    class _EmptyRtmidi:
        @staticmethod
        def MidiOut():
            b = _FakeBackendPort()
            b.get_port_count = lambda: 0    # port disappeared entirely
            return b

    monkeypatch.setattr(out_mod, "rtmidi", _EmptyRtmidi)
    out = MidiOutput()
    out._port = _DeadPort()
    out._port_name = "loopMIDI Port"

    for _ in range(RECONNECT_AFTER_FAILURES + 5):
        out.send_note_on(60, 100, 0)

    st = out.stats
    assert st["reconnects"] == 0
    assert out.is_stalled, "still stalled: the broken handle must be kept"
    assert st["last_error"]   # later send failures overwrite the reopen error
    # Sends keep failing (and would keep triggering rate-limited retries).
    assert st["failed"] == RECONNECT_AFTER_FAILURES + 5


def test_open_resets_failure_streak(monkeypatch):
    backend = {}

    class _FakeRtmidi:
        @staticmethod
        def MidiOut():
            b = _FakeBackendPort()
            backend["b"] = b
            return b

    monkeypatch.setattr(out_mod, "rtmidi", _FakeRtmidi)
    monkeypatch.setattr(out_mod, "_HAVE_RTMIDI", True)
    out = MidiOutput()
    out._port = _DeadPort()
    out._port_name = None
    for _ in range(3):
        out.send_note_on(60, 100, 0)
    assert out.stats["consecutive_failures"] == 3
    out.open("loopMIDI Port")
    assert out.stats["consecutive_failures"] == 0
    out.send_note_on(60, 100, 0)
    assert out.stats["sent"] == 1
