"""MIDI translation engine for EEG -> MIDI."""
from sogno_cane.midi.mapper import MappingEngine, MappingEvent, MappingConfig
from sogno_cane.midi.output import MidiOutput, list_output_ports
from sogno_cane.midi.scales import SCALES, build_scale, quantize_to_scale

__all__ = [
    "MappingEngine",
    "MappingConfig",
    "MappingEvent",
    "MidiOutput",
    "list_output_ports",
    "SCALES",
    "build_scale",
    "quantize_to_scale",
]
