"""MIDI output backed directly by ``python-rtmidi``.

We don't depend on ``mido`` at runtime: it's a nice abstraction but it
adds a dependency that proved fragile inside the Python embeddable
bundle. rtmidi exposes everything we need with a tiny byte-level API.

The :py:class:`MidiOutput` interface is unchanged so the rest of the
codebase keeps working.
"""
from __future__ import annotations

import threading
import time
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


# After this many back-to-back send failures the output is considered
# stalled and a reconnect to the same port name is attempted.
RECONNECT_AFTER_FAILURES = 8
# Minimum seconds between two reconnect attempts (a dead port must not spin).
RECONNECT_MIN_INTERVAL_S = 2.0

# Controller numbers of the channel-mode messages a panic sends.
_CC_SUSTAIN = 64
_CC_ALL_SOUND_OFF = 120
_CC_ALL_NOTES_OFF = 123


class MidiOutput:
    """Thread-safe MIDI output. Opens a single port and queues writes.

    Send failures are counted instead of vanishing: a stale handle (e.g.
    a loopMIDI port that was re-created, or a driver hiccup) makes every
    send raise while the app looks perfectly alive — the user sees the
    event monitor move while loopMIDI's counter is frozen. After a few
    consecutive failures the output automatically re-opens the same port
    by name and releases whatever was left hanging.

    It also keeps the notes it has switched on and not yet off. All Notes
    Off (CC 123) alone does not release them reliably: Ableton and many
    instruments ignore it, so a note whose note-off was lost (a stalled
    port, a port closed mid-note) stayed on until Ableton was restarted.
    Every release here sends the explicit note-offs, and a note is never
    switched on twice in a row: a synth that stacks voices would keep one
    of the two after the single note-off.
    """

    def __init__(self) -> None:
        self._port = None
        self._port_name: str | None = None
        # Re-entrant: a note-on holds it across the note-off it may send
        # first and its own send, so no other thread can come in between.
        self._lock = threading.RLock()
        # Health bookkeeping (all guarded by _lock).
        self._sent_count = 0
        self._failed_count = 0
        self._consec_failures = 0
        self._reconnects = 0
        self._last_error = ""
        self._last_reconnect_attempt = -1.0e9
        # (channel, note) switched on and not yet off (guarded by _lock).
        self._sounding: set[tuple[int, int]] = set()
        # Of those, the ones whose note-off failed: sent again as soon as
        # the port takes messages again.
        self._lost_offs: set[tuple[int, int]] = set()

    @property
    def is_open(self) -> bool:
        return self._port is not None

    @property
    def port_name(self) -> str | None:
        return self._port_name

    @staticmethod
    def _find_port_index(out, port_name: str) -> int:
        n = out.get_port_count()
        # 1. Exact match after stripping rtmidi's trailing index.
        for i in range(n):
            if _strip_rtmidi_suffix(out.get_port_name(i), i) == port_name:
                return i
        # 2. Exact match on raw rtmidi name.
        for i in range(n):
            if out.get_port_name(i) == port_name:
                return i
        # 3. Substring fallback.
        for i in range(n):
            if port_name in out.get_port_name(i):
                return i
        return -1

    def open(self, port_name: str) -> None:
        if not _HAVE_RTMIDI:
            raise RuntimeError(
                "python-rtmidi not installed; reinstall the bundle"
            )
        with self._lock:
            previous = self._port_name
            self._close_locked()
            out = rtmidi.MidiOut()
            idx = self._find_port_index(out, port_name)
            if idx < 0:
                raise RuntimeError(f"MIDI port {port_name!r} not found")
            out.open_port(idx, "SOGNO_CANE")
            if port_name != previous:
                # Whatever the old port could not release belongs to it.
                self._sounding.clear()
                self._lost_offs.clear()
            self._port = out
            self._port_name = port_name
            self._consec_failures = 0
            self._last_error = ""

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
        # Release what is sounding before the handle goes: a note-off sent
        # after closing would never arrive.
        self._release_locked()
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
        key = (int(channel) & 0x0F, max(0, min(127, int(note))))
        velocity = max(0, min(127, int(velocity)))
        with self._lock:
            if velocity == 0:                   # a note-off in MIDI terms
                if self._send_raw([_status(0x90, key[0]), key[1], 0]):
                    self._sounding.discard(key)
                return
            if key in self._sounding:
                # Already on: close it first, or a synth that stacks
                # voices keeps one after the single note-off to come.
                self._off_locked(key)
            if self._send_raw([_status(0x90, key[0]), key[1], velocity]):
                self._sounding.add(key)

    def send_note_off(self, note: int, channel: int = 0) -> None:
        key = (int(channel) & 0x0F, max(0, min(127, int(note))))
        with self._lock:
            self._off_locked(key)

    def _off_locked(self, key: tuple[int, int]) -> None:
        # Forgotten only once the note-off is out: if it fails, it is sent
        # again as soon as the port works (and by any release).
        if self._send_raw([_status(0x80, key[0]), key[1], 0]):
            self._sounding.discard(key)
            self._lost_offs.discard(key)
        elif key in self._sounding:
            self._lost_offs.add(key)

    def _resend_lost_offs_locked(self) -> None:
        for key in sorted(self._lost_offs):
            try:
                self._port.send_message([_status(0x80, key[0]), key[1], 0])
            except Exception:
                return
            self._sent_count += 1
            self._lost_offs.discard(key)
            self._sounding.discard(key)

    @property
    def sounding(self) -> set[tuple[int, int]]:
        """(channel, note) switched on and not yet off."""
        with self._lock:
            return set(self._sounding)

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
        """Release the notes this output switched on, then All Notes Off."""
        with self._lock:
            if channel is None:
                self._release_locked()
                return
            ch = int(channel) & 0x0F
            for key in sorted(k for k in self._sounding if k[0] == ch):
                self._off_locked(key)
            self._send_raw([_status(0xB0, ch), _CC_ALL_NOTES_OFF, 0])

    def release_all(self) -> int:
        """Explicit note-off for every note still on; how many there were."""
        with self._lock:
            n = len(self._sounding)
            self._release_locked()
            return n

    def _release_locked(self) -> None:
        for key in sorted(self._sounding):
            self._off_locked(key)
        for ch in range(16):
            self._send_raw([_status(0xB0, ch), _CC_ALL_NOTES_OFF, 0])

    def panic(self) -> None:
        """Silence everything on the port, whoever switched it on.

        A note-off for every note on every channel, then Sustain off, All
        Sound Off and All Notes Off: it also clears notes left on by a
        previous session or by a crash, which this output knows nothing
        about. One channel at a time, so a send never waits long.
        """
        for ch in range(16):
            with self._lock:
                for note in range(128):
                    self._send_raw([_status(0x80, ch), note, 0])
                    self._sounding.discard((ch, note))
                    self._lost_offs.discard((ch, note))
                for cc in (_CC_SUSTAIN, _CC_ALL_SOUND_OFF, _CC_ALL_NOTES_OFF):
                    self._send_raw([_status(0xB0, ch), cc, 0])
            time.sleep(0.001)

    def _send_raw(self, msg: list[int]) -> bool:
        """The one place a message reaches the port; False if it did not."""
        with self._lock:
            if self._port is None:
                return False
            try:
                self._port.send_message(msg)
            except Exception as exc:
                self._failed_count += 1
                self._consec_failures += 1
                self._last_error = str(exc) or type(exc).__name__
                self._maybe_reconnect_locked()
                return False
            self._sent_count += 1
            self._consec_failures = 0
            if self._lost_offs:
                self._resend_lost_offs_locked()
            return True

    # ------------------------------------------------------------------ #
    # Health / auto-reconnect                                            #
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> dict:
        """Send health: what actually reached the port vs what failed."""
        with self._lock:
            return {
                "sent": self._sent_count,
                "failed": self._failed_count,
                "consecutive_failures": self._consec_failures,
                "reconnects": self._reconnects,
                "last_error": self._last_error,
            }

    @property
    def is_stalled(self) -> bool:
        """True while sends are failing back-to-back (data not flowing)."""
        with self._lock:
            return (
                self._port is not None
                and self._consec_failures >= RECONNECT_AFTER_FAILURES
            )

    def _maybe_reconnect_locked(self) -> None:
        """Re-open the same port by name after repeated send failures.

        Called with the lock held, from the send path. Rate-limited so a
        genuinely missing port cannot make every send pay a scan cost.
        """
        if self._consec_failures < RECONNECT_AFTER_FAILURES:
            return
        name = self._port_name
        if not name:
            return
        now = time.monotonic()
        if now - self._last_reconnect_attempt < RECONNECT_MIN_INTERVAL_S:
            return
        self._last_reconnect_attempt = now
        try:
            out = rtmidi.MidiOut()
            idx = self._find_port_index(out, name)
            if idx < 0:
                raise RuntimeError(f"MIDI port {name!r} not found")
            out.open_port(idx, "SOGNO_CANE")
        except Exception as exc:
            # Keep the broken handle: every further send keeps failing,
            # which keeps this retry loop alive (rate-limited) until the
            # port reappears or the user disconnects.
            self._last_error = str(exc) or type(exc).__name__
            return
        # Swap handles only now that the replacement is alive.
        old, self._port = self._port, out
        try:
            old.close_port()
        except Exception:
            pass
        self._reconnects += 1
        self._consec_failures = 0
        # Note-offs lost during the stall left those notes on at the other
        # end: send them again, explicitly (or at the first send that gets
        # through, if the port is not taking messages yet).
        self._resend_lost_offs_locked()
        try:
            for ch in range(16):
                self._port.send_message(
                    [_status(0xB0, ch), _CC_ALL_NOTES_OFF, 0]
                )
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
