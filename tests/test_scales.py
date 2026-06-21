import pytest

from sogno_cane.midi.scales import (
    MidiNoteRange,
    SCALES,
    build_scale,
    quantize_to_scale,
)


def test_build_scale_within_range():
    notes = build_scale("major", "C", MidiNoteRange(60, 72))
    assert notes == (60, 62, 64, 65, 67, 69, 71, 72)
    assert all(60 <= n <= 72 for n in notes)


def test_build_scale_root_offset():
    c = build_scale("minor_pentatonic", "C", MidiNoteRange(48, 60))
    a = build_scale("minor_pentatonic", "A", MidiNoteRange(48, 60))
    assert c != a


def test_unknown_scale_raises():
    with pytest.raises(KeyError):
        build_scale("not_a_scale", "C")


def test_unknown_root_raises():
    with pytest.raises(KeyError):
        build_scale("major", "H")


def test_quantize_endpoints():
    notes = build_scale("major", "C", MidiNoteRange(60, 72))
    assert quantize_to_scale(0.0, notes) == notes[0]
    assert quantize_to_scale(1.0, notes) == notes[-1]
    assert notes[0] <= quantize_to_scale(0.5, notes) <= notes[-1]


def test_quantize_clamps_out_of_range():
    notes = (60, 62, 64)
    assert quantize_to_scale(-5.0, notes) == 60
    assert quantize_to_scale(5.0, notes) == 64


def test_all_scales_build():
    for name in SCALES:
        notes = build_scale(name, "C", MidiNoteRange(0, 127))
        assert len(notes) > 0
        assert notes == tuple(sorted(notes))


def test_invalid_note_range():
    with pytest.raises(ValueError):
        MidiNoteRange(80, 20)
    with pytest.raises(ValueError):
        MidiNoteRange(-1, 20)
