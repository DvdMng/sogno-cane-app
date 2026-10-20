"""Readiness for real Unicorn BCI Core EEG (0.9.0).

Measured on the official BCI Core recordings before these fixes:

* the event strategies were calibrated on the simulator's absolute scale:
  EVENTI (beta > 80 uV^2) fired 0 times a minute — on real data AND on the
  simulator — and CLIP (> 60 uV^2) 1.6 times a minute on real data;
* Recorder files loaded as 14 "channels" (technical columns included) and
  played only because the EEG happens to come first;
* the waveform drew raw DC-coupled values and showed nothing on screen.
"""
import numpy as np
import pytest

from sogno_cane import config as appconfig
from sogno_cane.eeg.bands import BandPowers
from sogno_cane.io.loaders import load_eeg
from sogno_cane.midi.context import WindowContext
from sogno_cane.midi.pacing import BurstDetector
from sogno_cane.midi.presets import rich_vocabulary_preset
from sogno_cane.midi.strategies.clip_launcher import (
    ClipLauncherStrategy,
    ClipRule,
)
from sogno_cane.midi.strategies.threshold_trigger import (
    ThresholdTriggerStrategy,
)

BANDS = ("delta", "theta", "alpha", "beta", "gamma")


def _bands(power: float) -> BandPowers:
    return BandPowers(
        by_band={b: np.full(8, power) for b in BANDS},
        sample_rate_hz=250, window_samples=250,
    )


def _ctx(t: float) -> WindowContext:
    return WindowContext(t_seconds=t, hop_seconds=0.1, sample_rate_hz=250,
                         window_samples=250)


def _power_series(scale: float, seconds: float = 120.0, seed: int = 0):
    """Log-normal 'resting' fluctuation with three clear 10x bursts.

    Band power comes from 1 s windows hopped every 0.1 s, so consecutive
    values share 90% of their samples: the fluctuation is modelled as a
    1 s moving average, not as independent draws (which would make random
    threshold crossings far more frequent than on real EEG).
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * 10)
    raw = rng.normal(0.0, 0.08 * np.sqrt(10.0), n + 9)
    logp = np.convolve(raw, np.ones(10) / 10.0, mode="valid")
    for at in (300, 700, 1000):
        logp[at:at + 8] += 1.0
    return scale * 10 ** logp


# --------------------------------------------------------------------------- #
# Burst detector                                                              #
# --------------------------------------------------------------------------- #

def test_burst_detector_scores_bursts_high_and_rest_low():
    det = BurstDetector()
    series = np.log10(_power_series(5.0))
    scores = [det.score(float(x), 0.1) for x in series]
    rest = np.abs(scores[50:290])
    assert np.percentile(rest, 95) < 3.0
    assert max(scores[300:308]) > 5.0


def test_burst_is_scored_against_the_past_not_itself():
    det = BurstDetector()
    for _ in range(300):
        det.score(0.0, 0.1)
    assert det.score(3.0, 0.1) > 10.0


# --------------------------------------------------------------------------- #
# Event strategies are scale-free in adaptive mode                            #
# --------------------------------------------------------------------------- #

BURSTS = (300, 700, 1000)


def _event_windows(strategy, series):
    hits = []
    for i, p in enumerate(series):
        evs = strategy.on_window(np.zeros((250, 8)), _bands(float(p)),
                                 _ctx(i * 0.1))
        if any(e.kind == "note_on" for e in evs):
            hits.append(i)
    return hits


def _count_threshold_events(strategy, series):
    return len(_event_windows(strategy, series))


def test_threshold_events_do_not_depend_on_absolute_scale():
    """Real beta ~5 uV^2, simulator ~80, a gel montage could be anything:
    the same relative activity must give exactly the same events."""
    runs = {
        scale: _event_windows(ThresholdTriggerStrategy(seed=1),
                              _power_series(scale))
        for scale in (5.0, 80.0, 5000.0)
    }
    assert runs[5.0] == runs[80.0] == runs[5000.0]
    hits = runs[5.0]
    for start in BURSTS:            # every burst is caught at its onset
        assert any(start <= h < start + 8 for h in hits), (start, hits)


def test_absolute_mode_still_available_and_unchanged():
    s = ThresholdTriggerStrategy(adaptive=False, threshold_uv2=80.0, seed=1)
    # Real-scale signal never reaches the old absolute threshold...
    assert _count_threshold_events(s, _power_series(5.0)) == 0
    # ...while a simulator-scale signal crossing it fires as before.
    s2 = ThresholdTriggerStrategy(adaptive=False, threshold_uv2=80.0, seed=1)
    assert _count_threshold_events(s2, _power_series(30.0)) == 3


def test_clip_rules_fire_on_relative_bursts():
    def run(scale):
        strat = ClipLauncherStrategy(rules=[
            ClipRule(eeg_channel=0, band="theta", midi_note=36),
        ])
        return _event_windows(strat, _power_series(scale))
    small, large = run(5.0), run(5000.0)
    assert small == large
    for start in BURSTS:
        assert any(start <= h < start + 8 for h in small), (start, small)
    # The stricter clip default stays rare between bursts.
    assert len(small) <= len(BURSTS) + 2


def test_defaults_are_the_calibrated_ones():
    assert ThresholdTriggerStrategy().relative_threshold == 2.0
    assert ClipLauncherStrategy().relative_threshold == 3.0
    b = rich_vocabulary_preset()
    assert b.threshold.adaptive and b.clips.adaptive


# --------------------------------------------------------------------------- #
# Configurations                                                              #
# --------------------------------------------------------------------------- #

def test_adaptive_settings_round_trip_through_config():
    b1 = rich_vocabulary_preset()
    b1.threshold.adaptive = False
    b1.threshold.relative_threshold = 2.75
    b1.clips.relative_threshold = 3.5
    data = appconfig.serialize_bundle(b1)
    b2 = rich_vocabulary_preset()
    appconfig.apply_bundle_config(b2, data)
    assert b2.threshold.adaptive is False
    assert b2.threshold.relative_threshold == 2.75
    assert b2.clips.relative_threshold == 3.5


def test_old_config_switches_to_adaptive_events():
    """A 0.8 config has no 'adaptive' key: it loads with the adaptive
    default, which is what makes EVENTI/CLIP work on real EEG."""
    b1 = rich_vocabulary_preset()
    data = appconfig.serialize_bundle(b1)
    for key in ("threshold", "clips"):
        data[key].pop("adaptive", None)
        data[key].pop("relative_threshold", None)
    b2 = rich_vocabulary_preset()
    appconfig.apply_bundle_config(b2, data)
    assert b2.threshold.adaptive and b2.clips.adaptive


# --------------------------------------------------------------------------- #
# Recorder files                                                              #
# --------------------------------------------------------------------------- #

def _write_recorder_csv(path, n=500, lost=()):
    header = ("EXG 1, EXG 2, EXG 3, EXG 4, EXG 5, EXG 6, EXG 7, EXG 8, "
              "SATURATION, CNT, BAT, VALID, DT, TRIG")
    lines = [header]
    for i in range(n):
        eeg = [50000.0 + k * 1000.0 + 10.0 * np.sin(i / 7.0) for k in range(8)]
        valid = 0.0 if i in lost else 1.0
        if i in lost:
            eeg = [0.0] * 8               # what a lost sample may contain
        tail = [0.0, float(i + 1), 63.0, valid, 0.007, 0.0]
        lines.append(",".join(f"{v:.3f}" for v in eeg + tail))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_recorder_csv_loads_only_the_eeg(tmp_path):
    p = tmp_path / "UnicornRawDataRecorder_test.csv"
    _write_recorder_csv(p)
    lo = load_eeg(str(p))
    assert lo.n_channels == 8
    assert lo.channel_names == [f"EXG {k}" for k in range(1, 9)]
    assert lo.sample_rate_hz == 250.0
    assert lo.meta["gtec_recorder"] is True
    assert "VALID" in [c.strip() for c in lo.meta["technical_columns"]]


def test_recorder_csv_bridges_lost_samples(tmp_path):
    p = tmp_path / "rec_lost.csv"
    _write_recorder_csv(p, lost=set(range(200, 210)))
    lo = load_eeg(str(p))
    assert lo.meta["lost_samples_bridged"] == 10
    # Held at the last valid value, never dropped to zero.
    assert np.all(lo.data[200:210] == lo.data[199])
    assert np.all(lo.data[200:210, 0] > 40000.0)


def test_plain_csv_is_untouched(tmp_path):
    p = tmp_path / "plain.csv"
    p.write_text("Fp1,Fp2\n1,2\n3,4\n5,6\n7,8\n9,10\n11,12\n13,14\n15,16\n")
    lo = load_eeg(str(p))
    assert lo.n_channels == 2
    assert "gtec_recorder" not in lo.meta


# --------------------------------------------------------------------------- #
# Display                                                                     #
# --------------------------------------------------------------------------- #

def test_display_cleaner_centres_dc_and_kills_mains():
    pytest.importorskip("pyqtgraph")
    from sogno_cane.ui.waveform_widget import _DisplayCleaner

    fs = 250
    t = np.arange(fs * 10) / fs
    alpha = 20.0 * np.sin(2 * np.pi * 10 * t)
    x = (50000.0 + alpha + 150.0 * np.sin(2 * np.pi * 50 * t))[:, None]
    x = np.repeat(x, 8, axis=1)
    cl = _DisplayCleaner(8, fs)
    y = np.vstack([cl.process(x[i:i + 25]) for i in range(0, x.shape[0], 25)])
    # No start-up spike from the 50 000 uV offset.
    assert np.max(np.abs(y[:, 0])) < 400.0
    steady = y[fs * 5:, 0]
    assert abs(float(steady.mean())) < 2.0              # centred
    # What is left is the alpha, not the hum.
    assert 10.0 < float(steady.std()) < 18.0
