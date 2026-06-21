import numpy as np

from sogno_cane.core.sources import ArrayPlaybackSource, PacketSource
from sogno_cane.eeg.simulator import EEGSimulator
from sogno_cane.eeg.unicorn_packet import PACKET_VALUES


def test_playback_yields_all_samples_then_stops():
    eeg = np.arange(500 * 8, dtype=np.float32).reshape(500, 8)
    src = ArrayPlaybackSource(eeg, 250, packet_samples=25, loop=False)
    total = 0
    while True:
        p = src.next_packet()
        if p is None:
            break
        assert p.samples.shape[1] == PACKET_VALUES
        total += p.n_samples
    assert total == 500
    assert src.is_finished


def test_playback_loops():
    eeg = np.zeros((100, 8), dtype=np.float32)
    src = ArrayPlaybackSource(eeg, 250, packet_samples=25, loop=True)
    count = 0
    for _ in range(20):  # 500 samples > 100 -> must wrap
        p = src.next_packet()
        assert p is not None
        count += p.n_samples
    assert count == 500
    assert not src.is_finished


def test_channel_fit_pads_and_truncates():
    # 3 channels -> padded to 8.
    src = ArrayPlaybackSource(np.ones((50, 3), np.float32), 250)
    p = src.next_packet()
    assert p.eeg.shape[1] == 8
    assert np.allclose(p.eeg[:, :3], 1.0)
    assert np.allclose(p.eeg[:, 3:], 0.0)
    # 12 channels -> truncated to 8.
    src2 = ArrayPlaybackSource(np.ones((50, 12), np.float32), 250)
    assert src2.next_packet().eeg.shape[1] == 8


def test_reset():
    src = ArrayPlaybackSource(np.zeros((100, 8), np.float32), 250, loop=False)
    src.next_packet()
    src.reset()
    assert src.position_seconds == 0.0
    assert not src.is_finished


def test_simulator_satisfies_protocol():
    sim = EEGSimulator(seed=0)
    assert isinstance(sim, PacketSource)
