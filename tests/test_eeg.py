import numpy as np
import pytest

from sogno_cane.eeg.profiles import ALL_PROFILES, DOG_PROFILE, HUMAN_PROFILE
from sogno_cane.eeg.simulator import EEGSimulator
from sogno_cane.eeg.unicorn_packet import PACKET_VALUES, UnicornPacket


def test_packet_shape_and_dtype():
    sim = EEGSimulator(profile=HUMAN_PROFILE, seed=0)
    pkt = sim.next_packet()
    assert isinstance(pkt, UnicornPacket)
    assert pkt.samples.shape == (sim.packet_samples, PACKET_VALUES)
    assert pkt.samples.dtype == np.float32
    assert pkt.eeg.shape[1] == len(HUMAN_PROFILE.channel_names)


def test_determinism_with_seed():
    a = EEGSimulator(profile=HUMAN_PROFILE, seed=42)
    b = EEGSimulator(profile=HUMAN_PROFILE, seed=42)
    for _ in range(5):
        pa = a.next_packet().eeg
        pb = b.next_packet().eeg
        assert np.array_equal(pa, pb)


def test_different_seeds_differ():
    a = EEGSimulator(profile=HUMAN_PROFILE, seed=1).next_packet().eeg
    b = EEGSimulator(profile=HUMAN_PROFILE, seed=2).next_packet().eeg
    assert not np.array_equal(a, b)


def test_invalid_packet_size():
    with pytest.raises(ValueError):
        EEGSimulator(profile=HUMAN_PROFILE, packet_samples=0)


def test_counter_is_monotonic_and_advances():
    sim = EEGSimulator(profile=HUMAN_PROFILE, seed=0)
    p1 = sim.next_packet()
    p2 = sim.next_packet()
    assert int(p1.counter[0]) == 0
    assert int(p2.counter[0]) == sim.packet_samples


def test_battery_drains_monotonically():
    sim = EEGSimulator(profile=HUMAN_PROFILE, seed=0)
    first = float(sim.next_packet().battery[0])
    for _ in range(50):
        last = float(sim.next_packet().battery[-1])
    assert last <= first


def test_human_alpha_dominates_dog_theta():
    """Profiles must be spectrally distinct in the documented way."""
    from sogno_cane.eeg.bands import compute_band_powers

    def mean_bands(profile):
        sim = EEGSimulator(profile=profile, seed=3)
        big = np.concatenate(
            [sim.next_packet().eeg.astype(float) for _ in range(40)], axis=0
        )
        return compute_band_powers(big, profile.sample_rate_hz)

    h = mean_bands(HUMAN_PROFILE)
    d = mean_bands(DOG_PROFILE)
    # Human: alpha is the strongest narrow band.
    assert h.channel_mean("alpha") > h.channel_mean("theta")
    # Dog: theta is much stronger than alpha (documented difference).
    assert d.channel_mean("theta") > d.channel_mean("alpha")


def test_reset_restores_state():
    sim = EEGSimulator(profile=HUMAN_PROFILE, seed=7)
    for _ in range(10):
        sim.next_packet()
    sim.reset()
    assert sim.elapsed_seconds == 0.0


def test_all_profiles_registered():
    assert set(ALL_PROFILES) == {"HUMAN", "DOG"}
    for p in ALL_PROFILES.values():
        assert len(p.channel_names) == 8
