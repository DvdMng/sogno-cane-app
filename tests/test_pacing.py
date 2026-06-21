from sogno_cane.midi.pacing import AdaptiveNorm, NoteGate


def test_adaptive_norm_uses_full_range():
    n = AdaptiveNorm()
    vals = [n.normalize(x) for x in [0, 1, 2, 3, 0, 1, 2, 3] * 5]
    # After adapting, extremes map near 0 and 1.
    lo = n.normalize(0.0)
    hi = n.normalize(3.0)
    assert lo < 0.25
    assert hi > 0.75
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_adaptive_norm_constant_input_is_stable():
    n = AdaptiveNorm()
    for _ in range(100):
        v = n.normalize(2.0)
    assert 0.0 <= v <= 1.0


def test_note_gate_min_interval():
    g = NoteGate()
    assert g.may_change(0.0, 0.5, min_interval=1.0, change_threshold=0.0)
    g.commit_on(0.0, 0.5)
    # Too soon.
    assert not g.may_change(0.5, 0.9, min_interval=1.0, change_threshold=0.0)
    # Enough time elapsed.
    assert g.may_change(1.1, 0.9, min_interval=1.0, change_threshold=0.0)


def test_note_gate_change_threshold():
    g = NoteGate()
    g.commit_on(0.0, 0.5)
    # Value barely moved -> blocked even after interval.
    assert not g.may_change(5.0, 0.55, min_interval=1.0, change_threshold=0.2)
    # Value moved enough -> allowed.
    assert g.may_change(5.0, 0.8, min_interval=1.0, change_threshold=0.2)


def test_note_gate_min_hold():
    g = NoteGate()
    g.commit_on(0.0, 0.5)
    assert not g.may_release(0.3, min_hold=0.6)
    assert g.may_release(0.7, min_hold=0.6)
    g.commit_off()
    assert g.may_release(0.0, min_hold=0.6)  # nothing held
