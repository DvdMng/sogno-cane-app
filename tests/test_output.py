"""Test the MIDI byte construction / clamping without a real port."""
from sogno_cane.midi.output import MidiOutput, _status


class _FakePort:
    def __init__(self):
        self.messages = []

    def send_message(self, msg):
        self.messages.append(list(msg))


def _patched():
    out = MidiOutput()
    out._port = _FakePort()
    out._port_name = "fake"
    return out


def test_status_byte():
    assert _status(0x90, 0) == 0x90
    assert _status(0x90, 5) == 0x95
    assert _status(0xB0, 15) == 0xBF


def test_note_on_clamps():
    out = _patched()
    out.send_note_on(200, 200, 0)   # out-of-range -> clamped to 127
    assert out._port.messages[-1] == [0x90, 127, 127]
    out.send_note_on(-5, -5, 0)
    assert out._port.messages[-1] == [0x90, 0, 0]


def test_note_off():
    out = _patched()
    out.send_note_off(64, 3)
    assert out._port.messages[-1] == [0x83, 64, 0]


def test_control_change():
    out = _patched()
    out.send_control_change(74, 100, 1)
    assert out._port.messages[-1] == [0xB1, 74, 100]


def test_pitch_bend_center_and_extremes():
    out = _patched()
    out.send_pitch_bend(0, 0)        # center -> 8192
    assert out._port.messages[-1] == [0xE0, 0, 64]
    out.send_pitch_bend(8191, 0)     # max
    assert out._port.messages[-1] == [0xE0, 127, 127]
    out.send_pitch_bend(-8192, 0)    # min
    assert out._port.messages[-1] == [0xE0, 0, 0]


def test_all_notes_off_every_channel():
    out = _patched()
    out.all_notes_off()
    # 16 channels, each CC 123 value 0.
    assert len(out._port.messages) == 16
    assert all(m[1] == 123 and m[2] == 0 for m in out._port.messages)


def test_send_with_no_port_is_safe():
    out = MidiOutput()       # no port opened
    out.send_note_on(60, 100, 0)   # must not raise
    assert not out.is_open
