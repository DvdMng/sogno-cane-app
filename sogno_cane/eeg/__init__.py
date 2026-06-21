"""EEG simulation package. Models the g.tec Unicorn Hybrid Black stream."""
from sogno_cane.eeg.profiles import DOG_PROFILE, HUMAN_PROFILE, DeviceProfile
from sogno_cane.eeg.simulator import EEGSimulator
from sogno_cane.eeg.unicorn_packet import (
    BATTERY_INDEX,
    COUNTER_INDEX,
    EEG_CHANNELS,
    PACKET_VALUES,
    VALIDATION_INDEX,
    UnicornPacket,
)

__all__ = [
    "EEGSimulator",
    "DeviceProfile",
    "HUMAN_PROFILE",
    "DOG_PROFILE",
    "UnicornPacket",
    "EEG_CHANNELS",
    "PACKET_VALUES",
    "BATTERY_INDEX",
    "COUNTER_INDEX",
    "VALIDATION_INDEX",
]
