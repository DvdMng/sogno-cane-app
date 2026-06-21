"""Built-in EEG -> MIDI strategies."""
from sogno_cane.midi.strategies.band_to_note import BandToNoteStrategy
from sogno_cane.midi.strategies.clip_launcher import ClipLauncherStrategy, ClipRule
from sogno_cane.midi.strategies.coherence_cc import CoherenceCCStrategy
from sogno_cane.midi.strategies.markov_generative import MarkovGenerativeStrategy
from sogno_cane.midi.strategies.per_channel_band import (
    PerChannelBandStrategy,
    PerChannelVoice,
)
from sogno_cane.midi.strategies.per_channel_cc import (
    PerChannelCCConfig,
    PerChannelCCStrategy,
)
from sogno_cane.midi.strategies.threshold_trigger import ThresholdTriggerStrategy

__all__ = [
    "BandToNoteStrategy",
    "ThresholdTriggerStrategy",
    "CoherenceCCStrategy",
    "MarkovGenerativeStrategy",
    "PerChannelBandStrategy",
    "PerChannelVoice",
    "PerChannelCCStrategy",
    "PerChannelCCConfig",
    "ClipLauncherStrategy",
    "ClipRule",
]
