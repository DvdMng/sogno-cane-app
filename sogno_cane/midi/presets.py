"""Pre-built mapping configurations for common scenarios.

The "Rich Vocabulary" preset is the application default: every EEG channel
emits BOTH a distinct MIDI note (on its own MIDI channel) AND a distinct
continuous CC value, in parallel. With 8 EEG channels per device this gives
8 simultaneous melodic voices + 8 macro knobs per Unicorn (= 16 voices and
16 CCs total when both human and dog devices are streaming).

Channel allocation (per device):

* MIDI ch 0..7 : per-channel band-to-note voices (one EEG channel each)
* MIDI ch 8    : Markov generative melody (whole-stream)
* MIDI ch 9    : threshold "events" (bursts)
* MIDI ch 10   : coherence CC (inter-channel)
* MIDI ch 11   : clip-launcher triggers
* MIDI ch 12   : per-channel CC stream (8 distinct CCs on CC 20..27)

The dog device offsets MIDI channels by 0 since the user is expected to
route human and dog into TWO loopMIDI ports (one each), keeping all 16 MIDI
channels free for each animal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sogno_cane.midi.scales import MidiNoteRange
from sogno_cane.midi.strategies import (
    BandToNoteStrategy,
    ClipLauncherStrategy,
    ClipRule,
    CoherenceCCStrategy,
    MarkovGenerativeStrategy,
    PerChannelBandStrategy,
    PerChannelCCConfig,
    PerChannelCCStrategy,
    PerChannelVoice,
    ThresholdTriggerStrategy,
)


@dataclass
class PresetBundle:
    """A complete set of strategies."""

    per_channel_band: PerChannelBandStrategy
    per_channel_cc: PerChannelCCStrategy
    threshold: ThresholdTriggerStrategy
    coherence: CoherenceCCStrategy
    markov: MarkovGenerativeStrategy
    clips: ClipLauncherStrategy

    def as_list(self) -> list:
        return [
            self.per_channel_band,
            self.per_channel_cc,
            self.threshold,
            self.coherence,
            self.markov,
            self.clips,
        ]


def rich_vocabulary_preset(
    n_eeg_channels: int = 8,
    scale: str = "minor_pentatonic",
    root: str = "A",
) -> PresetBundle:
    """Default preset: every channel emits a distinct note AND a distinct CC.

    Both ``PerChannelBandStrategy`` and ``PerChannelCCStrategy`` are enabled
    in parallel, plus three "global" strategies for variety.
    """
    # --- per-channel band->note: 8 voices on MIDI channels 0..7 -----------
    note_voices = [
        PerChannelVoice(
            eeg_channel=i,
            midi_channel=i % 16,
            band="alpha" if i < 4 else "beta",
            scale=scale,
            root=root,
            note_range=MidiNoteRange(
                lo=48 - (i % 2) * 12,
                hi=84 - (i % 2) * 12,
            ),
            sustain=True,
        )
        for i in range(n_eeg_channels)
    ]
    per_channel_band = PerChannelBandStrategy(voices=note_voices)

    # --- per-channel CCs: 8 CCs (20..27) all on MIDI channel 12 -----------
    cc_voices = [
        PerChannelCCConfig(
            eeg_channel=i,
            midi_channel=12,
            cc_number=20 + i,
            band="alpha",
            smoothing=0.5,
        )
        for i in range(n_eeg_channels)
    ]
    per_channel_cc = PerChannelCCStrategy(voices=cc_voices)

    # --- global threshold (events on beta) on MIDI ch 9 -------------------
    threshold = ThresholdTriggerStrategy.from_scale(
        scale="major_pentatonic",
        root="C",
        octave_low=4,
        octave_high=5,
        band="beta",
        threshold_uv2=80.0,
        channel=9,
    )

    # --- coherence CC on MIDI ch 10, CC 1 (mod wheel) ---------------------
    coherence = CoherenceCCStrategy(
        band="alpha",
        channels_a=(0, 1, 2, 3),
        channels_b=(4, 5, 6, 7),
        cc_number=1,
        channel=10,
    )

    # --- Markov on MIDI ch 8 ---------------------------------------------
    markov = MarkovGenerativeStrategy(
        scale="dorian",
        root="D",
        channel=8,
    )

    # --- clip launcher on MIDI ch 11 --------------------------------------
    clip_rules = [
        ClipRule(
            eeg_channel=i,
            band="theta" if i % 2 == 0 else "gamma",
            midi_note=36 + i,           # C2, C#2, ... = standard clip-row notes
            midi_channel=11,
            threshold_uv2=60.0,
        )
        for i in range(n_eeg_channels)
    ]
    clips = ClipLauncherStrategy(rules=clip_rules)

    return PresetBundle(
        per_channel_band=per_channel_band,
        per_channel_cc=per_channel_cc,
        threshold=threshold,
        coherence=coherence,
        markov=markov,
        clips=clips,
    )
