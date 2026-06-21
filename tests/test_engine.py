import time

import numpy as np

from sogno_cane.core.engine import RealtimeEngine
from sogno_cane.core.sources import ArrayPlaybackSource
from sogno_cane.eeg.profiles import HUMAN_PROFILE
from sogno_cane.midi.output import MidiOutput
from sogno_cane.midi.presets import rich_vocabulary_preset


class _FakeMidi(MidiOutput):
    def __init__(self):
        super().__init__()
        self.sent = []

    def _send_raw(self, msg):
        self.sent.append(list(msg))


def test_engine_threaded_run_and_stop():
    eng = RealtimeEngine(
        profile=HUMAN_PROFILE, seed=1, midi=_FakeMidi(),
    )
    eng.set_time_scale(0.05)  # run fast
    for s in rich_vocabulary_preset().as_list():
        eng.mapper.add_strategy(s)
    eng.start()
    time.sleep(0.4)
    assert eng.is_running
    eng.stop()
    assert not eng.is_running
    assert eng.event_count > 0


def test_engine_playback_finishes_and_callbacks():
    eeg = np.random.default_rng(0).normal(0, 20, size=(250, 8)).astype(np.float32)
    src = ArrayPlaybackSource(eeg, 250, packet_samples=25, loop=False)
    finished = []
    eng = RealtimeEngine(
        source=src, midi=_FakeMidi(),
        on_finished=lambda: finished.append(True),
    )
    eng.set_time_scale(0.01)
    eng.start()
    time.sleep(0.5)
    assert finished == [True]
    assert not eng.is_running


def test_engine_uses_source_sample_rate():
    src = ArrayPlaybackSource(np.zeros((400, 8), np.float32), 200, loop=False)
    eng = RealtimeEngine(source=src, midi=_FakeMidi())
    assert eng.mapper.config.sample_rate_hz == 200


def test_simulator_property():
    eng = RealtimeEngine(profile=HUMAN_PROFILE, seed=1)
    assert eng.simulator is not None
    src = ArrayPlaybackSource(np.zeros((10, 8), np.float32), 250)
    eng2 = RealtimeEngine(source=src)
    assert eng2.simulator is None
