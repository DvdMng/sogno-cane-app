"""Sleep architecture in the simulator — fix for "human and dog sound
identical, and the dog's ~20 min cycle never shows a REM acceleration".

The profiles now carry a :class:`SleepCycleSpec`; the simulator crossfades
band gains, band-power volatility and EMG twitch rate between deep NREM and
REM. Human (90 min cycle) and dog (20 min cycle) therefore evolve on very
different timescales and diverge immediately.
"""
import dataclasses

import numpy as np

from sogno_cane.eeg.bands import compute_band_powers
from sogno_cane.eeg.profiles import DOG_PROFILE, HUMAN_PROFILE
from sogno_cane.eeg.simulator import EEGSimulator


def test_both_profiles_have_distinct_sleep_architecture():
    assert DOG_PROFILE.sleep is not None
    assert HUMAN_PROFILE.sleep is not None
    # The dog cycles ~4-5x faster than the human — the core audible contrast.
    assert DOG_PROFILE.sleep.cycle_seconds * 4 <= \
        HUMAN_PROFILE.sleep.cycle_seconds
    assert DOG_PROFILE.sleep.cycle_seconds == 1200.0   # ~20 min, as reported


def test_rem_weight_geometry_dog():
    sim = EEGSimulator(profile=DOG_PROFILE, seed=0)
    spec = DOG_PROFILE.sleep
    cyc = spec.cycle_seconds
    # Starts in deep NREM.
    assert sim.sleep_stage_rem_weight(0.0) == 0.0
    # Centre of the REM tail is full REM, and the cycle repeats.
    mid_rem = (1.0 - spec.rem_fraction / 2.0) * cyc
    assert sim.sleep_stage_rem_weight(mid_rem) == 1.0
    assert sim.sleep_stage_rem_weight(mid_rem + 3 * cyc) == 1.0
    # The NREM->REM edge is a smooth ramp, not a step.
    ramp_t = (1.0 - spec.rem_fraction) * cyc - spec.transition_seconds / 2.0
    w = sim.sleep_stage_rem_weight(ramp_t)
    assert 0.0 < w < 1.0
    # Weight defaults to the simulator's own elapsed time.
    assert sim.sleep_stage_rem_weight() == sim.sleep_stage_rem_weight(0.0)


def test_profile_without_sleep_spec_stays_stationary():
    prof = dataclasses.replace(DOG_PROFILE, sleep=None)
    sim = EEGSimulator(profile=prof, seed=0)
    for t in (0.0, 123.0, 4567.0):
        assert sim.sleep_stage_rem_weight(t) == 0.0


def _band_powers_at(profile, t0: float, seed: int, seconds: float = 4.0):
    sim = EEGSimulator(profile=profile, seed=seed)
    sim._elapsed_s = t0    # jump straight to the target cycle position
    big = np.concatenate(
        [sim.next_packet().eeg.astype(float)
         for _ in range(int(seconds * 10))],
        axis=0,
    )
    return compute_band_powers(big, profile.sample_rate_hz)


def test_dog_rem_accelerates_fast_bands_and_drops_delta():
    spec = DOG_PROFILE.sleep
    cyc = spec.cycle_seconds
    nrem = _band_powers_at(DOG_PROFILE, cyc * 0.30, seed=11)
    rem = _band_powers_at(
        DOG_PROFILE, cyc * (1.0 - spec.rem_fraction / 2.0), seed=11,
    )
    # Dreaming: fast bands surge...
    assert rem.channel_mean("beta") > nrem.channel_mean("beta") * 2.0
    assert rem.channel_mean("gamma") > nrem.channel_mean("gamma") * 1.5
    assert rem.channel_mean("theta") > nrem.channel_mean("theta")
    # ...and slow-wave delta recedes.
    assert nrem.channel_mean("delta") > rem.channel_mean("delta") * 1.5


def test_human_and_dog_diverge_within_minutes():
    """At t=6 min the dog is deep in NREM delta while the human, on its slow
    cycle, still keeps its alpha character — the streams must not have the
    same spectral shape (the 'they sound identical' complaint)."""
    t0 = 360.0
    h = _band_powers_at(HUMAN_PROFILE, t0, seed=21)
    d = _band_powers_at(DOG_PROFILE, t0, seed=21)
    h_ratio = h.channel_mean("alpha") / max(h.channel_mean("delta"), 1e-12)
    d_ratio = d.channel_mean("alpha") / max(d.channel_mean("delta"), 1e-12)
    assert h_ratio > d_ratio * 2.0


def test_seeded_determinism_survives_sleep_modulation():
    a = EEGSimulator(profile=DOG_PROFILE, seed=42)
    b = EEGSimulator(profile=DOG_PROFILE, seed=42)
    for _ in range(5):
        assert np.array_equal(a.next_packet().eeg, b.next_packet().eeg)


def test_reset_returns_to_cycle_start():
    sim = EEGSimulator(profile=DOG_PROFILE, seed=1)
    for _ in range(20):
        sim.next_packet()
    sim.reset()
    assert sim.elapsed_seconds == 0.0
    assert sim.sleep_stage_rem_weight() == 0.0
    for state in sim._band_states:
        assert np.all(state.amplitude_env == 1.0)
