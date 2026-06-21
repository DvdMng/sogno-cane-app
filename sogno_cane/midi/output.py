"""MIDI output backed directly by ``python-rtmidi``.

We don't depend on ``mido`` at runtime: it's a nice abstraction but it
adds a dependency that proved fragile inside the Python embeddable
bundle. rtmidi exposes everything we need with a tiny byte-level API.

The :py:class:`MidiOutput` interface is unchanged so the rest of the
codebase keeps working.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterable

try:
    import rtmidi  # type: ignore[import-untyped]
    _HAVE_RTMIDI = True
except Exception:  # pragma: no cover
    rtmidi = None  # type: ignore[assignment]
    _HAVE_RTMIDI = False


# Raw MIDI status byte builders.
def _status(kind: int, channel: int) -> int:
    return (kind & 0xF0) | (channel & 0x0F)


def _strip_rtmidi_suffix(name: str, idx: int) -> str:
    """rtmidi on Windows appends ``" <idx>"`` to disambiguate duplicate port
    names. We strip that suffix for display because it confuses users who
    only see ``"loopMIDI Port 1"`` in loopMIDI itself.
    """
    suffix = f" {idx}"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def list_output_ports() -> list[str]:
    """Return cleaned-up names of MIDI output ports available on the system."""
    if not _HAVE_RTMIDI:
        return []
    try:
        out = rtmidi.MidiOut()
        try:
            return [
                _strip_rtmidi_suffix(out.get_port_name(i), i)
                for i in range(out.get_port_count())
            ]
        finally:
            try:
                out.close_port()
            except Exception:
                pass
            try:
                del out
            except Exception:
                pass
    except Exception:
        return []


class MidiOutput:
    """Thread-safe MIDI output. Opens a single port and queues writes."""

    def __init__(self) -> None:
        self._port = None
        self._port_name: str | None = None
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return self._port is not None

    @property
    def port_name(self) -> str | None:
        return self._port_name

    def open(self, port_name: str) -> None:
        if not _HAVE_RTMIDI:
            raise RuntimeError(
                "python-rtmidi not installed; reinstall the bundle"
            )
        with self._lock:
            self._close_locked()
            out = rtmidi.MidiOut()
            n = out.get_port_count()
            idx = -1
            # 1. Exact match after stripping rtmidi's trailing index.
            for i in range(n):
                if _strip_rtmidi_suffix(out.get_port_name(i), i) == port_name:
                    idx = i
                    break
            # 2. Exact match on raw rtmidi name.
            if idx < 0:
                for i in range(n):
                    if out.get_port_name(i) == port_name:
                        idx = i
                        break
            # 3. Substring fallback.
            if idx < 0:
                for i in range(n):
                    if port_name in out.get_port_name(i):
                        idx = i
                        break
            if idx < 0:
                raise RuntimeError(f"MIDI port {port_name!r} not found")
            out.open_port(idx, "SOGNO_CANE")
            self._port = out
            self._port_name = port_name

    def open_virtual(self, name: str = "SOGNO_CANE") -> None:
        """Open a virtual port. Supported on macOS/Linux only. On Windows
        you must use loopMIDI (or equivalent) and :py:meth:`open` its port.
        """
        if not _HAVE_RTMIDI:
            raise RuntimeError("python-rtmidi not installed")
        with self._lock:
            self._close_locked()
            out = rtmidi.MidiOut()
            out.open_virtual_port(name)
            self._port = out
            self._port_name = name

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._port is None:
            return
        # Best-effort: All Notes Off on every channel before closing.
        try:
            for ch in range(16):
                self._port.send_message([_status(0xB0, ch), 123, 0])
        except Exception:
            pass
        try:
            self._port.close_port()
        except Exception:
            pass
        try:
            del self._port
        except Exception:
            pass
        self._port = None
        self._port_name = None

    # ------------------------------------------------------------------ #
    # Sending                                                            #
    # ------------------------------------------------------------------ #

    def send_note_on(
        self, note: int, velocity: int = 96, channel: int = 0
    ) -> None:
        self._send_raw([
            _status(0x90, channel),
            max(0, min(127, int(note))),
            max(0, min(127, int(velocity))),
        ])

    def send_note_off(self, note: int, channel: int = 0) -> None:
        self._send_raw([
            _status(0x80, channel),
            max(0, min(127, int(note))),
            0,
        ])

    def send_control_change(
        self, control: int, value: int, channel: int = 0
    ) -> None:
        self._send_raw([
            _status(0xB0, channel),
            max(0, min(127, int(control))),
            max(0, min(127, int(value))),
        ])

    def send_pitch_bend(self, value: int, channel: int = 0) -> None:
        # value is -8192..8191; MIDI pitch wheel uses 0..16383 with 8192=center
        v14 = max(0, min(16383, int(value) + 8192))
        self._send_raw([
            _status(0xE0, channel),
            v14 & 0x7F,
            (v14 >> 7) & 0x7F,
        ])

    def all_notes_off(self, channel: int | None = None) -> None:
        if channel is None:
            for ch in range(16):
                self.send_control_change(123, 0, channel=ch)
        else:
            self.send_control_change(123, 0, channel=channel)

    def _send_raw(self, msg: list[int]) -> None:
        with self._lock:
            if self._port is None:
                return
            try:
                self._port.send_message(msg)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Context manager                                                    #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "MidiOutput":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@contextmanager
def open_port(name: str):
    """Convenience context manager."""
    out = MidiOutput()
    out.open(name)
    try:
        yield out
    finally:
        out.close()


def send_messages(out: MidiOutput, msgs: Iterable) -> None:
    """Send an iterable of objects with .type attribute (mido-like).

    Kept for backwards compatibility with older code that used mido messages.
    """
    for m in msgs:
        kind = getattr(m, "type", None)
        if kind == "note_on":
            out.send_note_on(m.note, m.velocity, m.channel)
        elif kind == "note_off":
            out.send_note_off(m.note, m.channel)
        elif kind == "control_change":
            out.send_control_change(m.control, m.value, m.channel)
        elif kind == "pitchwheel":
            out.send_pitch_bend(m.pitch, m.channel)
