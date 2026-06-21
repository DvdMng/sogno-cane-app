"""File I/O: load external EEG files, record live streams, manage an archive."""
from sogno_cane.io.archive import Archive
from sogno_cane.io.loaders import LoadedEEG, SUPPORTED_EXTENSIONS, load_eeg
from sogno_cane.io.recording import Recorder, Recording, RecordingMeta

__all__ = [
    "Archive",
    "LoadedEEG",
    "load_eeg",
    "SUPPORTED_EXTENSIONS",
    "Recorder",
    "Recording",
    "RecordingMeta",
]
