"""The voices keep doing what the mapping panel says they do (0.9.5).

Found by driving each voice through the real engine with live edits,
source changes and lost samples, on real recordings and synthetic EEG:

* a held note (HOLD / legato) stuck on the synth when its MIDI channel was
  changed, or its voice switched off, while it sounded;
* EVENTI and CLIP kept the baseline of the old band, electrode or source:
  a spurious event at once, or up to ~35 s of silence;
* NOTE / CC per channel squeezed a new band into the old band's range;
* a signal loss stretched every adaptive reference: EVENTI mute for about
  a minute, CLIP for two;
* COERENZA read ~18 for independent signals, 0 in delta whatever the
  signal, the mains hum as gamma coherence, a shared rhythm in the
  neighbouring bands too;
* MARKOV read beta within 0.37-0.52 on real EEG: two notes per step at
  velocity 70-80 almost every time, its maximum density never reached.
"""
import collections

import numpy as np

from sogno_cane.eeg.bands import BandPowers
from sogno_cane.eeg.unicorn_packet import (
    PACKET_VALUES,
    VALIDATION_INDEX,
    UnicornPacket,
)
from sogno_cane.midi.context import WindowContext
from sogno_cane.midi.mapper import MappingConfig, MappingEngine, MappingEvent
from sogno_cane.midi.pacing import BurstDetector
from sogno_cane.midi.presets import rich_vocabulary_preset
from sogno_cane.midi.scales import build_scale
from sogno_cane.midi.strategies.clip_launcher import (
    ClipLauncherStrategy,
    ClipRule,
)
from sogno_cane.midi.strategies.coherence_cc import CoherenceCCStrategy
from sogno_cane.midi.strategies.markov_generative import (
    MarkovGenerativeStrategy,
)
from sogno_cane.midi.strategies.per_channel_band import (
    PerChannelBandStrategy,
    PerChannelVoice,
)
from sogno_cane.midi.strategies.per_channel_cc import (
    PerChannelCCConfig,
    PerChannelCCStrategy,
)
from sogno_cane.midi.strategies.threshold_trigger import (
    ThresholdTriggerStrategy,
)

FS = 250
BANDS = ("delta", "theta", "alpha", "beta", "gamma")
# DC offsets of a real Unicorn (tens of thousands of uV, all different).
_DC = np.array([50384, 77483, 21621, 55072, 53106, 51083, 60829, 43080.0])
_FLAT = np.zeros((FS, 8))


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _bands(**power) -> BandPowers:
    """Same power on every electrode; per band, 5 uV^2 unless given. A
    value may also be an 8-element sequence (one per electrode)."""
    by_band = {}
    for b in BANDS:
        v = power.get(b, 5.0)
        by_band[b] = np.broadcast_to(np.asarray(v, dtype=float), (8,)).copy()
    return BandPowers(by_band=by_band, sample_rate_hz=FS, window_samples=FS)


def _ctx(t: float) -> WindowContext:
    return WindowContext(t_seconds=t, hop_seconds=0.1, sample_rate_hz=FS,
                         window_samples=FS)


def _wobble(i: int) -> float:
    """Small deterministic fluctuation (+-10%), never an event at 2 sigma."""
    return 1.0 + 0.1 * np.sin(i)


def _eeg(seconds, seed=0, amp_uv=6.0, shared=(), bursts=(), hum_uv=0.0):
    """Synthetic 8-channel EEG on DC offsets: independent coloured noise,
    ``shared`` = [(t0, t1, f, amp)] identical on every electrode,
    ``bursts`` = [(t0, t1, f, amp)] on every electrode, phase-shifted."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    w = rng.normal(0.0, 1.0, (n, 8))
    x = np.empty_like(w)
    acc = np.zeros(8)
    for i in range(n):
        acc = 0.97 * acc + w[i]
        x[i] = acc
    x *= amp_uv / x.std(axis=0)
    for t0, t1, f, a in shared:
        m = (t >= t0) & (t < t1)
        x[m] += (a * np.sin(2 * np.pi * f * t[m]))[:, None]
    for t0, t1, f, a in bursts:
        m = (t >= t0) & (t < t1)
        env = np.hanning(int(m.sum()))
        for ch in range(8):
            x[m, ch] += a * env * np.sin(2 * np.pi * f * t[m] + ch)
    x += hum_uv * np.sin(2 * np.pi * 50.0 * t)[:, None]
    return x + _DC


def _packet(rows, valid=1.0) -> UnicornPacket:
    s = np.zeros((rows.shape[0], PACKET_VALUES), dtype=np.float32)
    s[:, :8] = rows
    s[:, VALIDATION_INDEX] = valid
    return UnicornPacket(samples=s, timestamp=0.0, sample_rate=FS)


def _run(strategies, eeg, valid=None, edits=None):
    """Feed ``eeg`` through a fresh engine; ``edits`` = {t: fn} applied
    live. Returns [(t, event)]."""
    eng = MappingEngine(MappingConfig(sample_rate_hz=FS))
    for s in strategies:
        eng.add_strategy(s)
    eng.reset_strategy_state()
    pending = sorted((edits or {}).items())
    out = []
    for i in range(0, eeg.shape[0], 25):
        while pending and pending[0][0] <= i / FS:
            pending.pop(0)[1]()
        v = 1.0 if valid is None else valid[i:i + 25]
        for ev in eng.process_packet(_packet(eeg[i:i + 25], v)):
            out.append((eng.elapsed_seconds, ev))
    return out


def _stuck(events):
    """Held notes (no duration) never released on their own channel."""
    held = collections.Counter()
    for _t, e in events:
        if e.kind == "note_on" and e.duration == 0:
            held[(e.channel, e.note)] += 1
        elif e.kind == "note_off" and held[(e.channel, e.note)]:
            held[(e.channel, e.note)] -= 1
    return {k: v for k, v in held.items() if v}


def _held_cc(events, t0, step=0.1):
    """The CC value in force every ``step`` s from ``t0`` on."""
    cc = [(t, e.value) for t, e in events if e.kind == "cc"]
    if not cc:
        return np.array([])
    out, j, cur = [], 0, None
    for t in np.arange(t0, cc[-1][0] + 1.0, step):
        while j < len(cc) and cc[j][0] <= t:
            cur = cc[j][1]
            j += 1
        if cur is not None:
            out.append(cur)
    return np.array(out, dtype=float)


class _Probe:
    """A voice that records what the engine asks of it."""

    name = "probe"
    channel = 0

    def __init__(self):
        self.enabled = True
        self.windows = 0
        self.shutdowns = 0
        self.resets = 0

    def on_window(self, eeg_window, bands, ctx):
        self.windows += 1
        return []

    def shutdown(self):
        self.shutdowns += 1
        return [MappingEvent(kind="note_off", channel=0, note=60)]

    def reset_runtime(self):
        self.resets += 1


# --------------------------------------------------------------------------- #
# No note left sounding                                                       #
# --------------------------------------------------------------------------- #

def test_held_event_note_is_released_where_it_was_played():
    s = ThresholdTriggerStrategy(note_length_seconds=0.0, channel=9, seed=1)
    events = []
    for i in range(300):
        if i == 202:
            s.channel = 3                     # changed while the note sounds
        p = 50.0 if 200 <= i < 205 else 5.0 * _wobble(i)
        events += s.on_window(_FLAT, _bands(beta=p), _ctx(i * 0.1))
    ons = [e for e in events if e.kind == "note_on"]
    offs = [(e.channel, e.note) for e in events if e.kind == "note_off"]
    assert len(ons) == 1 and ons[0].channel == 9
    assert offs == [(9, ons[0].note)]


def test_engine_releases_a_switched_off_voice_once_and_restarts_it():
    probe = _Probe()
    eng = MappingEngine(MappingConfig(sample_rate_hz=FS))
    eng.add_strategy(probe)
    x = _eeg(6)

    def feed(a, b):
        out = []
        for i in range(a * FS, b * FS, 25):
            out += eng.process_packet(_packet(x[i:i + 25]))
        return out

    feed(0, 2)
    seen = probe.windows
    probe.enabled = False
    offs = feed(2, 4)
    assert probe.shutdowns == 1
    assert [e.kind for e in offs] == ["note_off"]
    assert probe.windows == seen
    probe.enabled = True
    feed(4, 6)
    assert probe.resets == 1, "switched back on: starts as at START"
    assert probe.windows > seen


def test_markov_legato_notes_are_released_where_they_were_played():
    s = MarkovGenerativeStrategy(note_length_seconds=0.0, channel=8, seed=3)

    def to_channel_2():
        s.channel = 2

    def switch_off():
        assert s._held, "the test must switch off a sounding phrase"
        s.enabled = False

    events = _run([s], _eeg(20, seed=1),
                  edits={6.05: to_channel_2, 12.05: switch_off})
    channels = {e.channel for _t, e in events if e.kind == "note_on"}
    assert channels == {8, 2}
    assert not [t for t, e in events if e.kind == "note_on" and t > 12.1]
    assert _stuck(events) == {}


def test_note_voices_release_held_notes_on_every_change():
    pcb = rich_vocabulary_preset().per_channel_band
    for v in pcb.voices:
        v.note_length_seconds = 0.0
    sounding = []

    def change(action):
        def edit():
            sounding.append(dict(pcb._last_notes))
            action()
        return edit

    edits = {
        10.05: change(lambda: setattr(pcb.voices[2], "midi_channel", 9)),
        20.05: change(lambda: setattr(pcb.voices[5], "enabled", False)),
        30.05: change(lambda: setattr(pcb, "enabled", False)),
    }
    x = _eeg(40, seed=2, bursts=[(t0, t0 + 2.0, 10.0, 12.0)
                                 for t0 in range(3, 40, 5)])
    events = _run([pcb], x, edits=edits)
    assert 2 in sounding[0] and 5 in sounding[1] and sounding[2]
    assert any(e.channel == 9 for _t, e in events if e.kind == "note_on")
    assert _stuck(events) == {}


def test_clip_switched_off_right_after_a_launch_releases_it():
    clips = ClipLauncherStrategy(rules=[
        ClipRule(eeg_channel=0, band="theta", midi_note=36, midi_channel=11),
    ])
    x = _eeg(30, seed=3, bursts=[(20.0, 21.5, 6.0, 40.0)])
    eng = MappingEngine(MappingConfig(sample_rate_hz=FS))
    eng.add_strategy(clips)
    eng.reset_strategy_state()
    events = []
    for i in range(0, x.shape[0], 25):
        evs = eng.process_packet(_packet(x[i:i + 25]))
        events += [(0.0, e) for e in evs]
        if any(e.kind == "note_on" for e in evs):
            clips.enabled = False
    assert [e for _t, e in events if e.kind == "note_on"]
    assert _stuck(events) == {}


# --------------------------------------------------------------------------- #
# A new band, electrode or source is learnt from scratch                      #
# --------------------------------------------------------------------------- #

def test_event_band_change_learns_the_new_band_from_scratch():
    s = ThresholdTriggerStrategy(seed=1)                 # beta
    hits = []
    for i in range(600):
        if i == 300:
            s.band = "theta"                             # 12x the beta level
        theta = 600.0 if 450 <= i < 455 else 60.0 * _wobble(i)
        bands = _bands(beta=5.0 * _wobble(i), theta=theta)
        if any(e.kind == "note_on"
               for e in s.on_window(_FLAT, bands, _ctx(i * 0.1))):
            hits.append(i)
    assert hits and all(450 <= h < 455 for h in hits), hits


def test_clip_rule_moved_to_another_electrode_starts_fresh():
    rule = ClipRule(eeg_channel=0, band="theta", midi_note=36,
                    midi_channel=11)
    s = ClipLauncherStrategy(rules=[rule])
    hits = []
    for i in range(600):
        if i == 300:
            rule.eeg_channel = 6                         # 12x the level
        level = np.full(8, 5.0 * _wobble(i))
        level[6] = 600.0 if 450 <= i < 455 else 60.0 * _wobble(i)
        if any(e.kind == "note_on"
               for e in s.on_window(_FLAT, _bands(theta=level),
                                    _ctx(i * 0.1))):
            hits.append(i)
    assert hits and all(450 <= h < 455 for h in hits), hits


def test_a_new_run_forgets_the_previous_source():
    """What the app does on a source change: a new engine, same voices."""
    b = rich_vocabulary_preset()
    _run(b.as_list(), _eeg(30, seed=4, amp_uv=6.0))
    assert b.threshold._burst._mean is not None
    assert b.clips._bursts and b.markov._readings
    assert b.coherence._history is not None
    events = _run(b.as_list(), _eeg(30, seed=5, amp_uv=20.0))  # louder
    assert b.markov._readings and b.coherence._history is not None
    early = [t for t, e in events
             if e.kind == "note_on" and e.channel in (9, 11) and t < 3.0]
    assert early == [], "the old level must not fire events at once"


def test_note_and_cc_voices_rescale_when_their_band_changes():
    voice = PerChannelVoice(eeg_channel=0, midi_channel=0, band="alpha",
                            min_interval_seconds=0.0,
                            min_duration_seconds=0.0,
                            change_threshold_norm=0.0)
    notes = PerChannelBandStrategy(voices=[voice])
    knob = PerChannelCCConfig(eeg_channel=0, midi_channel=12, cc_number=20,
                              band="alpha", smoothing=0.0)
    ccs = PerChannelCCStrategy(voices=[knob])
    after_notes, after_cc = [], []
    for i in range(600):
        if i == 300:
            voice.band = knob.band = "gamma"             # 100x weaker
        bands = _bands(alpha=100.0 * _wobble(i), gamma=1.0 * _wobble(i + 7))
        n_ev = notes.on_window(_FLAT, bands, _ctx(i * 0.1))
        c_ev = ccs.on_window(_FLAT, bands, _ctx(i * 0.1))
        if 305 <= i < 400:
            after_notes += [e.note for e in n_ev if e.kind == "note_on"]
            after_cc += [e.value for e in c_ev]
    scale = build_scale(voice.scale, voice.root, voice.note_range)
    low_quarter = scale[len(scale) // 4]
    assert after_notes and min(after_notes) > low_quarter, after_notes
    assert after_cc and 20 < np.median(after_cc) < 107, after_cc


# --------------------------------------------------------------------------- #
# Lost samples                                                                #
# --------------------------------------------------------------------------- #

def test_burst_detector_ignores_a_flat_line():
    det = BurstDetector()
    for k in range(300):
        det.score(0.7 + 0.05 * np.sin(k), 0.1)
    state = (det._mean, det._dev, det._n)
    assert all(det.score(-6.0, 0.1) == 0.0 for _ in range(50))
    assert (det._mean, det._dev, det._n) == state


def test_windows_at_the_edges_of_a_loss_are_not_analysed():
    probe = _Probe()
    eng = MappingEngine(MappingConfig(sample_rate_hz=FS))
    eng.add_strategy(probe)
    x = _eeg(10)
    valid = np.ones(x.shape[0])
    valid[4 * FS:7 * FS] = 0.0
    skipped = []
    for i in range(0, x.shape[0], 25):
        before = probe.windows
        eng.process_packet(_packet(x[i:i + 25], valid[i:i + 25]))
        t = round(eng.elapsed_seconds, 1)
        if t >= 1.0 and probe.windows == before:
            skipped.append(t)
    edges = [round(4.1 + 0.1 * k, 1) for k in range(1, 9)]
    edges += [round(7.0 + 0.1 * k, 1) for k in range(1, 9)]
    assert skipped == edges


def test_events_carry_on_normally_after_a_signal_loss():
    x = _eeg(120, seed=6, bursts=[(t0, t0 + 1.0, 20.0, 12.0)
                                  for t0 in range(10, 120, 6)])
    valid = np.ones(x.shape[0])
    valid[60 * FS:63 * FS] = 0.0
    ref_s, lost_s = (ThresholdTriggerStrategy(seed=1) for _ in range(2))
    spread = {}

    def note(name, s):
        return lambda: spread.__setitem__(name, s._burst._dev)

    ref = _run([ref_s], x, edits={64.05: note("ref", ref_s)})
    lost = _run([lost_s], x, valid=valid,
                edits={64.05: note("lost", lost_s)})

    def ons(events, lo, hi):
        return [t for t, e in events if e.kind == "note_on" and lo <= t < hi]

    assert ons(lost, 60.0, 64.0) == [], "nothing played from the loss"
    # The baseline comes out of the loss as it went in (0.9.4: three times
    # wider, so for a minute only the strongest bursts got through).
    assert abs(spread["lost"] / spread["ref"] - 1.0) < 0.1, spread
    assert len(ons(ref, 64.0, 120.0)) >= 8
    assert abs(len(ons(lost, 64.0, 120.0)) - len(ons(ref, 64.0, 120.0))) <= 1


def test_note_voice_is_silent_without_signal_and_keeps_its_range():
    voice = PerChannelVoice(eeg_channel=0, midi_channel=0,
                            note_length_seconds=0.0,
                            min_interval_seconds=0.0,
                            min_duration_seconds=0.0,
                            change_threshold_norm=0.0)
    s = PerChannelBandStrategy(voices=[voice])
    for i in range(200):
        s.on_window(_FLAT, _bands(alpha=20.0 * _wobble(i)), _ctx(i * 0.1))
    assert s._last_notes.get(0) is not None
    rng_before = (s._norms[0].floor, s._norms[0].ceil)
    evs = []
    for i in range(200, 230):
        evs += s.on_window(_FLAT, _bands(alpha=0.0), _ctx(i * 0.1))
    assert [e.kind for e in evs] == ["note_off"]
    assert (s._norms[0].floor, s._norms[0].ceil) == rng_before


def test_cc_voice_holds_its_value_without_signal():
    s = PerChannelCCStrategy(voices=[
        PerChannelCCConfig(eeg_channel=0, midi_channel=12, cc_number=20),
    ])
    sent = []
    for i in range(130):
        p = 20.0 * _wobble(i) if i < 100 else 0.0
        sent.append(len(s.on_window(_FLAT, _bands(alpha=p), _ctx(i * 0.1))))
    assert sum(sent[:100]) > 0 and sum(sent[100:]) == 0


def test_markov_is_silent_without_signal_and_keeps_its_readings():
    s = MarkovGenerativeStrategy(seed=1)
    for i in range(200):
        s.on_window(_FLAT, _bands(beta=5.0 * _wobble(i)), _ctx(i * 0.1))
    history = {k: list(r.history) for k, r in s._readings.items()}
    flat = _bands(**{b: 0.0 for b in BANDS})
    evs = []
    for i in range(200, 260):
        evs += s.on_window(_FLAT, flat, _ctx(i * 0.1))
    assert not [e for e in evs if e.kind == "note_on"]
    assert {k: list(r.history) for k, r in s._readings.items()} == history


# --------------------------------------------------------------------------- #
# COERENZA                                                                    #
# --------------------------------------------------------------------------- #

def _coherence(band, x):
    s = CoherenceCCStrategy(band=band, channel=10, cc_number=1)
    return _run([s], x)


def test_coherence_reads_zero_for_independent_groups_even_with_hum():
    x = _eeg(40, seed=7, hum_uv=100.0)        # the hum is common to all
    for band in BANDS:
        assert np.median(_held_cc(_coherence(band, x), 5.0)) <= 3.0, band


def test_coherence_follows_a_shared_rhythm_in_its_own_band_only():
    x = _eeg(40, seed=8, hum_uv=100.0, shared=[(0, 40, 10.0, 15.0)])
    assert np.median(_held_cc(_coherence("alpha", x), 5.0)) >= 110.0
    for band in ("theta", "beta"):
        assert np.median(_held_cc(_coherence(band, x), 5.0)) <= 15.0, band


def test_coherence_measures_delta_too():
    x = _eeg(40, seed=9, shared=[(0, 40, 2.0, 15.0)])
    assert np.median(_held_cc(_coherence("delta", x), 5.0)) >= 100.0


def test_coherence_waits_for_four_seconds_and_holds_without_signal():
    events = _coherence("alpha", _eeg(10, seed=10))
    first = min(t for t, e in events if e.kind == "cc")
    assert 4.0 <= first < 4.5
    flat = np.repeat(_DC[None, :], 10 * FS, axis=0)
    assert not [e for _t, e in _coherence("alpha", flat) if e.kind == "cc"]


# --------------------------------------------------------------------------- #
# MARKOV                                                                      #
# --------------------------------------------------------------------------- #

def test_markov_reaches_both_ends_of_its_range():
    x = _eeg(180, seed=11, bursts=[(t0, t0 + 3.0, 20.0, a)
                                   for t0, a in zip(range(10, 180, 12),
                                                    (4, 16, 8, 20, 2, 12) * 3)])
    events = _run([MarkovGenerativeStrategy(seed=2)], x)
    ons = [(round(t, 3), e) for t, e in events if e.kind == "note_on"]
    sizes = collections.Counter(collections.Counter(t for t, _e in ons)
                                .values())
    velocities = [e.velocity for _t, e in ons]
    assert sizes[1] and sizes[4], sizes        # max 4 notes is reached
    assert max(velocities) - min(velocities) >= 60


def test_markov_follows_a_band_that_rises():
    n = 150 * FS
    t = np.arange(n) / FS
    env = np.clip((t - 60) / 10, 0, 1) * np.clip((100 - t) / 10, 0, 1)
    x = _eeg(150, seed=12)
    x = x + (12.0 * env * np.sin(2 * np.pi * 6.0 * t))[:, None]   # theta
    s = MarkovGenerativeStrategy(seed=4)
    eng = MappingEngine(MappingConfig(sample_rate_hz=FS))
    eng.add_strategy(s)
    eng.reset_strategy_state()
    dark = collections.defaultdict(list)
    for i in range(0, n, 25):
        eng.process_packet(_packet(x[i:i + 25]))
        sec = eng.elapsed_seconds
        part = "before" if 30 <= sec < 60 else "rise" if 62 <= sec < 80 \
            else None
        if part:
            dark[part].append(s._levels["darkness"])
    assert np.mean(dark["rise"]) > 0.9
    assert np.mean(dark["before"]) < 0.7


def test_markov_band_change_starts_a_new_reading():
    s = MarkovGenerativeStrategy(seed=1)
    for i in range(100):
        s.on_window(_FLAT, _bands(beta=5.0 * _wobble(i)), _ctx(i * 0.1))
    assert len(s._readings["energy"].history) >= 9
    s.energy_band = "gamma"
    s.on_window(_FLAT, _bands(gamma=50.0), _ctx(10.0))
    reading = s._readings["energy"]
    assert reading.band == "gamma" and len(reading.history) == 1
