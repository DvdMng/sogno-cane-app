"""Live acquisition from a real Unicorn device over the network.

The app never talks to the Bluetooth link itself: g.tec already ships
applications that own the device and forward its samples over the network,
and we consume that stream. This keeps the portable bundle dependency-free
(UDP needs nothing but the standard library) and works with every Unicorn
generation, because g.tec's own tool handles the hardware.

Supported senders (both auto-detected, per datagram)
----------------------------------------------------
* **Unicorn BCI Core Recorder** (g.tec Suite 2024, Settings -> Network ->
  "Sending data via UDP", default port 1001) sends **ASCII text**: g.tec's
  own receiver example (unicorn-bi/Unicorn-BCI-Core-Recorder,
  ``UDP Receiver/Program.cs``) does nothing but
  ``Encoding.ASCII.GetString(datagram)``. Each line is one sample in the
  same serialisation as the Recorder's CSV files, e.g.::

      49780.490,76867.610,...,44277.860,0.000,1.000,63.000,1.000,0.000,0.000

  i.e. ``EXG 1..8, SATURATION, CNT, BAT, VALID, DT, TRIG`` — comma
  separated, dot decimals, newline terminated. Columns not enabled in the
  Recorder's Acquisition settings are omitted, but the 8 EEG channels always
  come first. A header line (``EXG 1, EXG 2, ...``) is recognised and skipped.
* **Unicorn UDP Interface** (Unicorn Suite Hybrid Black -> DevTools ->
  ``UnicornUDP.exe``) sends one sample per datagram as **17 little-endian
  float32** = ``EEG1..8, ACC X/Y/Z, GYR X/Y/Z, BAT, CNT, VALID`` (68 bytes),
  byte-for-byte the layout of
  :class:`~sogno_cane.eeg.unicorn_packet.UnicornPacket`.

Text versus binary is told apart by the bytes themselves: a float32 payload
is never made only of digits, separators and newlines. The column layout is
then inferred from the channel count.

Two headsets: streams are identified by SENDER, not by port
-----------------------------------------------------------
Every Recorder instance sends to UDP 1001 unless someone changes it, so a
human + dog rig usually means two senders on one port. Identifying a stream
by its port alone (0.7–0.9) mixed the two brains into one panel, and since
both panels could listen on the same port, the second panel ended up
mirroring the first. Now:

* :class:`UdpHub` owns **one exclusively bound socket per port** for the
  whole app (``SO_EXCLUSIVEADDRUSE`` on Windows, where ``SO_REUSEADDR``
  lets a second socket silently share a port), and routes every datagram by
  its sender address — each Recorder sends from its own socket.
* A :class:`UnicornLiveSource` (one per panel) claims **one sender**. With no
  explicit sender it adopts the first one nobody else has claimed, so two
  panels on the same port split the two headsets between them.
* Discovery reports every (port, sender) stream and which panel already uses
  it, so the second panel is never offered the first panel's signal.
* If a claimed sender falls silent and a new unclaimed one appears on the
  port (a Recorder restarted its network output), the panel adopts it.

The headset's clock paces the engine, not the PC's
--------------------------------------------------
g.tec's own raw recordings (the Recorder's DT column: milliseconds between
two received samples) show how the samples really travel: 250 Hz sampled,
but delivered over Bluetooth in bursts of 4 every ~16 ms (~5% late, merged
into 8), and against the PC clock the device runs at ~250.65 Hz, not 250.

Up to 0.9.2 the engine re-cut this stream into 25 samples every 100 ms of
the PC clock. Two visible faults followed: the traces moved ten times a
second in 100 ms jumps, and since the headset delivers ~0.65 samples a
second more than were consumed, a backlog — i.e. latency — grew by ~0.15 s
a minute until the bounded buffer overflowed. Now
:meth:`UnicornLiveSource.next_packet` hands over everything that has
arrived the moment it arrives (``self_clocked``): the engine waits on the
device instead of a timer, so there is no drift and no extra latency. The
display smooths the bursts on its own side.

Design notes
------------
* A bounded buffer per consumer: a slow or stalled consumer can never grow
  latency without bound; the oldest samples are dropped (and counted).
* :meth:`UnicornLiveSource.next_packet` never returns ``None`` while the
  source is open. Silence on the wire for :data:`GAP_SECONDS` yields
  packets that *hold the last sample* and are flagged invalid — not zeros:
  on DC-coupled hardware a jump from tens of thousands of microvolts to zero
  is a giant step that would fire every voice at once.
* :meth:`UnicornLiveSource.cadence` measures how the stream actually
  arrives (rate, bursts, pauses, samples lost at the source), so the source
  side can be checked on site instead of assumed.
"""
from __future__ import annotations

import socket
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from sogno_cane.eeg.unicorn_packet import (
    BATTERY_INDEX,
    COUNTER_INDEX,
    EEG_CHANNELS,
    PACKET_VALUES,
    VALIDATION_INDEX,
    UnicornPacket,
)

# Default UDP ports of the g.tec senders. (1000 is deliberately absent: it
# is the Recorder's own trigger INPUT port, never a data output.)
DEFAULT_UDP_PORT: int = 1001          # BCI Core Recorder "send data" default
ALT_UDP_PORTS: tuple[int, ...] = (1001, 1002, 1003, 1004, 5001)

# Channel counts we know how to interpret.
LAYOUT_UNICORN_17 = "unicorn-17"      # UDP Interface / Hybrid Black
LAYOUT_RECORDER_14 = "recorder-14"    # BCI Core Recorder, full default set
LAYOUT_EEG_ONLY = "eeg-only"          # any sender trimmed to the EEG channels

ENCODING_TEXT = "testo"
ENCODING_BINARY = "binario"

# BCI Core Recorder default column order (README "Acquisition settings",
# confirmed on the official recordings: EXG 1..8, SATURATION, CNT, BAT,
# VALID, DT, TRIG).
_REC_SAT, _REC_CNT, _REC_BAT, _REC_VALID, _REC_DT, _REC_STATUS = range(8, 14)

# Bytes that may appear in a text sample line (plus header letters).
_TEXT_OK = frozenset(range(0x20, 0x7F)) | {0x09, 0x0A, 0x0D}

# A claimed sender silent for this long may be replaced by a new unclaimed
# sender on the same port (the Recorder restarted its network output).
ADOPT_AFTER_SECONDS = 1.5

# Silence on the wire longer than this is a dropout, not a late burst (the
# longest Bluetooth pause in g.tec's recordings is ~66 ms): from then on the
# engine is fed held samples, flagged invalid, until the data comes back.
GAP_SECONDS = 0.25

# Datagrams this close together arrived as one burst (one sample per
# datagram is sent back to back).
_SAME_BURST_SECONDS = 0.002

Sender = tuple  # (ip, port) of the datagram's source


def describe_layout(n_channels: int) -> str:
    if n_channels == PACKET_VALUES:
        return LAYOUT_UNICORN_17
    if n_channels == 14:
        return LAYOUT_RECORDER_14
    return LAYOUT_EEG_ONLY


def frame_to_packet_rows(frame: np.ndarray) -> np.ndarray:
    """Map ``(n_samples, n_channels)`` sender rows onto canonical 17-col rows.

    Only the first 8 columns (always EEG, in every documented sender layout)
    are required. Battery/counter/validation are carried across when the
    sender's layout pins them down; otherwise they are synthesised, because
    a plausible placeholder is better than mislabelling an unrelated channel
    (a saturation flag read as a battery percentage, say).
    """
    n, nc = frame.shape
    out = np.zeros((n, PACKET_VALUES), dtype=np.float32)
    k = min(nc, EEG_CHANNELS)
    out[:, 0:k] = frame[:, 0:k]

    layout = describe_layout(nc)
    if layout == LAYOUT_UNICORN_17:
        # Exact same layout: carry accelerometer, gyroscope and telemetry.
        out[:] = frame.astype(np.float32, copy=False)
    elif layout == LAYOUT_RECORDER_14:
        out[:, BATTERY_INDEX] = frame[:, _REC_BAT]
        out[:, COUNTER_INDEX] = frame[:, _REC_CNT]
        out[:, VALIDATION_INDEX] = frame[:, _REC_VALID]
    else:
        out[:, BATTERY_INDEX] = 100.0
        out[:, VALIDATION_INDEX] = 1.0
    return out


# --------------------------------------------------------------------------- #
# Datagram decoding                                                           #
# --------------------------------------------------------------------------- #

def looks_like_text(payload: bytes) -> bool:
    """True if the payload is plain ASCII text (Recorder), not float32."""
    return bool(payload) and all(b in _TEXT_OK for b in payload)


def _split_fields(line: str) -> list[str]:
    # Semicolon-separated lines are the European CSV convention, where the
    # comma is the decimal mark; otherwise the comma separates fields (what
    # the Recorder actually writes: dot decimals, comma separators).
    if ";" in line:
        return [f.strip().replace(",", ".") for f in line.split(";")]
    if "," in line:
        return [f.strip() for f in line.split(",")]
    if "\t" in line:
        return [f.strip() for f in line.split("\t")]
    return line.split()


def parse_text_datagram(payload: bytes) -> Optional[tuple[np.ndarray, int]]:
    """Decode a Recorder text datagram.

    Returns ``(frame, header_lines)`` where ``frame`` is
    ``(n_samples, n_channels)`` float32 (possibly zero rows when the datagram
    only carried a header), or ``None`` if it is not a usable sample line.
    """
    text = payload.replace(b"\x00", b"").decode("ascii", errors="strict")
    rows: list[list[float]] = []
    headers = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = [f for f in _split_fields(line) if f != ""]
        if len(fields) < EEG_CHANNELS:
            return None              # too few fields to be EEG at all
        try:
            rows.append([float(f) for f in fields])
        except ValueError:
            # A labelled line ("EXG 1, EXG 2, ...") is the header, not junk.
            if any(c.isalpha() for c in line):
                headers += 1
                continue
            return None
    if not rows:
        return (np.zeros((0, EEG_CHANNELS), dtype=np.float32), headers) \
            if headers else None
    # Keep the dominant channel count (a torn line must not poison the rest).
    n_ch = Counter(len(r) for r in rows).most_common(1)[0][0]
    frame = np.asarray([r for r in rows if len(r) == n_ch], dtype=np.float32)
    if not np.all(np.isfinite(frame[:, :EEG_CHANNELS])):
        return None
    return frame, headers


def parse_binary_datagram(
    payload: bytes, locked_channels: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Decode a float32 datagram (Unicorn UDP Interface) into a frame."""
    if len(payload) < 4 or len(payload) % 4 != 0:
        return None
    n_floats = len(payload) // 4
    n_ch = locked_channels
    if n_ch is None or n_floats % n_ch != 0:
        n_ch = None
        for candidate in (PACKET_VALUES, 14, EEG_CHANNELS):
            if n_floats % candidate == 0:
                n_ch = candidate
                break
        if n_ch is None:
            if n_floats < EEG_CHANNELS:
                return None
            n_ch = n_floats          # single sample, unusual channel count
    flat = np.frombuffer(payload, dtype="<f4")
    if flat.size % n_ch != 0:
        return None
    frame = flat.reshape(-1, n_ch)
    if not np.all(np.isfinite(frame[:, :EEG_CHANNELS])):
        return None
    return frame


def decode_datagram(
    data: bytes, locked_channels: Optional[int] = None,
) -> Optional[tuple[np.ndarray, str, int]]:
    """``(frame, encoding, header_lines)`` for any supported datagram.

    The frame may have zero rows (header-only datagram). ``None`` means the
    datagram is not a Unicorn stream at all.
    """
    # NUL padding is only stripped to *recognise* text: in a float32
    # datagram trailing zero bytes are real values (0.0 is four NULs),
    # so binary decoding must always see the original bytes.
    text_view = data.rstrip(b"\x00")
    if looks_like_text(text_view):
        try:
            parsed = parse_text_datagram(text_view)
        except UnicodeDecodeError:
            return None
        if parsed is None:
            return None
        frame, headers = parsed
        return frame, ENCODING_TEXT, headers
    frame = parse_binary_datagram(data, locked_channels)
    if frame is None:
        return None
    return frame, ENCODING_BINARY, 0


def _battery_of(frame: np.ndarray) -> Optional[float]:
    """Battery percentage carried by a frame, when its layout has one."""
    if frame.shape[0] == 0:
        return None
    layout = describe_layout(frame.shape[1])
    if layout == LAYOUT_RECORDER_14:
        return float(frame[-1, _REC_BAT])
    if layout == LAYOUT_UNICORN_17:
        return float(frame[-1, BATTERY_INDEX])
    return None


def _exclusive_udp_socket(host: str, port: int) -> socket.socket:
    """Bind ``port`` so that no other socket can share it.

    On Windows ``SO_REUSEADDR`` would let a second socket bind the same
    port and silently split (or, for broadcast, duplicate) the traffic —
    exactly how a second panel came to mirror the first. Exclusive use
    turns that into a clear "port in use" error instead.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
    if exclusive is not None:
        sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    except OSError:
        pass
    try:
        sock.bind((host, port))
    except OSError:
        sock.close()
        raise
    sock.settimeout(0.2)
    return sock


# --------------------------------------------------------------------------- #
# Per-port hub: one socket, many senders, many consumers                      #
# --------------------------------------------------------------------------- #

@dataclass
class _SenderState:
    first_seen: float
    last_seen: float
    datagrams: int = 0
    last_data: bytes = b""


@dataclass(frozen=True)
class StreamInfo:
    """One Unicorn stream as seen on the wire: a sender on a port."""

    port: int
    sender: Sender
    channels: int
    layout: str
    encoding: str
    battery: Optional[float]
    claimed_by: Optional[str]
    seconds_since_data: float

    def as_dict(self) -> dict:
        return {
            "port": self.port,
            "sender": self.sender,
            "channels": self.channels,
            "layout": self.layout,
            "encoding": self.encoding,
            "battery": self.battery,
            "claimed_by": self.claimed_by,
        }


class UdpHub:
    """The one socket listening on a UDP port, shared by every consumer.

    Datagrams are routed by sender address: a sender claimed by a consumer
    goes to that consumer only; an unclaimed sender is offered to a consumer
    still waiting for one (and claimed if its data decodes); everything is
    recorded so discovery can list who is talking on the port.
    """

    _registry: dict[int, "UdpHub"] = {}
    _registry_lock = threading.Lock()

    def __init__(self, port: int, host: str) -> None:
        self.port = int(port)
        self.host = host
        self._sock = _exclusive_udp_socket(host, self.port)
        self._lock = threading.Lock()
        self._consumers: list["UnicornLiveSource"] = []
        self._claims: dict[Sender, "UnicornLiveSource"] = {}
        self._senders: dict[Sender, _SenderState] = {}
        self._holds = 0                 # temporary holds taken by discovery
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._rx_loop, name=f"SOGNO_CANE-udp-{self.port}",
            daemon=True,
        )
        self._thread.start()

    # -- registry --------------------------------------------------------- #

    @classmethod
    def acquire(cls, port: int, host: str = "") -> "UdpHub":
        """The hub for ``port``, binding it first if nobody has yet."""
        with cls._registry_lock:
            hub = cls._registry.get(int(port))
            if hub is None or hub._stop.is_set():
                hub = cls(port, host)
                cls._registry[int(port)] = hub
            return hub

    @classmethod
    def existing(cls, port: int) -> Optional["UdpHub"]:
        with cls._registry_lock:
            hub = cls._registry.get(int(port))
            return None if hub is None or hub._stop.is_set() else hub

    def _maybe_close(self) -> None:
        with self._lock:
            idle = not self._consumers and self._holds <= 0
        if not idle:
            return
        with UdpHub._registry_lock:
            if UdpHub._registry.get(self.port) is self:
                del UdpHub._registry[self.port]
        self._stop.set()
        try:
            self._sock.close()
        except Exception:
            pass
        if self._thread.is_alive() and \
                threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    # -- discovery holds -------------------------------------------------- #

    def hold(self) -> None:
        with self._lock:
            self._holds += 1

    def release_hold(self) -> None:
        with self._lock:
            self._holds -= 1
        self._maybe_close()

    # -- consumers -------------------------------------------------------- #

    def subscribe(self, consumer: "UnicornLiveSource",
                  sender: Optional[Sender]) -> None:
        with self._lock:
            if sender is not None:
                owner = self._claims.get(sender)
                if owner is not None and owner is not consumer:
                    raise OSError(
                        f"the stream from {sender[0]}:{sender[1]} on UDP "
                        f"{self.port} is already used by {owner.label}"
                    )
                self._claims[sender] = consumer
            if consumer not in self._consumers:
                self._consumers.append(consumer)

    def unsubscribe(self, consumer: "UnicornLiveSource") -> None:
        with self._lock:
            if consumer in self._consumers:
                self._consumers.remove(consumer)
            for s in [s for s, c in self._claims.items() if c is consumer]:
                del self._claims[s]
        self._maybe_close()

    def streams(self, max_age: float = 3.0) -> list[StreamInfo]:
        """Every sender heard on this port within ``max_age`` seconds."""
        now = time.monotonic()
        with self._lock:
            snapshot = [
                (s, st.last_seen, st.last_data, self._claims.get(s))
                for s, st in self._senders.items()
                if now - st.last_seen <= max_age
            ]
        out = []
        for sender, last_seen, data, owner in snapshot:
            decoded = decode_datagram(data)
            if decoded is None or decoded[0].shape[0] == 0:
                continue
            frame, encoding, _ = decoded
            out.append(StreamInfo(
                port=self.port,
                sender=sender,
                channels=int(frame.shape[1]),
                layout=describe_layout(int(frame.shape[1])),
                encoding=encoding,
                battery=_battery_of(frame),
                claimed_by=owner.label if owner is not None else None,
                seconds_since_data=now - last_seen,
            ))
        return sorted(out, key=lambda s: (s.sender[0], s.sender[1]))

    # -- receive loop ----------------------------------------------------- #

    def _rx_loop(self) -> None:
        sock = self._sock
        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            sender = (addr[0], int(addr[1]))
            now = time.monotonic()
            with self._lock:
                st = self._senders.get(sender)
                if st is None:
                    st = _SenderState(first_seen=now, last_seen=now)
                    self._senders[sender] = st
                st.last_seen = now
                st.datagrams += 1
                st.last_data = data
                owner = self._claims.get(sender)
                candidates = [] if owner is not None else [
                    c for c in self._consumers if c._wants_sender(now)
                ]
                # Forget long-gone senders so the table cannot grow forever.
                if len(self._senders) > 64:
                    for s in [s for s, v in self._senders.items()
                              if now - v.last_seen > 30.0
                              and s not in self._claims]:
                        del self._senders[s]
            if owner is not None:
                owner._ingest(data)
                continue
            for consumer in candidates:
                if consumer._try_adopt(data, sender):
                    with self._lock:
                        if sender in self._claims:   # lost a race: ignore
                            break
                        for s in [s for s, c in self._claims.items()
                                  if c is consumer]:
                            del self._claims[s]
                        self._claims[sender] = consumer
                    break


# --------------------------------------------------------------------------- #
# Consumer: one per panel                                                     #
# --------------------------------------------------------------------------- #

class UnicornLiveSource:
    """A :class:`~sogno_cane.core.sources.PacketSource` fed by a UDP stream.

    Parameters
    ----------
    port
        UDP port to listen on (the port configured in the g.tec sender).
    host
        Interface to bind when this source is the first on its port. ``""``
        binds every interface, which is what you want when the Unicorn tool
        runs on another machine.
    sample_rate_hz
        Nominal device rate; the Unicorn family streams at 250 Hz.
    packet_samples
        Size of the held packets that stand in for the signal during a
        dropout (25 = 100 ms). Real samples are handed over as they arrive,
        however many came in.
    buffer_seconds
        Bound on the receive backlog. Older samples are dropped rather than
        queued, so latency stays flat if the consumer stalls.
    sender
        ``(ip, port)`` of the one sender to listen to. ``None`` adopts the
        first sender on the port that no other source has claimed.
    label
        Who is listening (the panel name), shown to other panels.
    """

    # The engine must not pace this source with its own timer: the source
    # blocks until the device delivers, so the device's clock rules.
    self_clocked = True

    def __init__(
        self,
        port: int = DEFAULT_UDP_PORT,
        host: str = "",
        sample_rate_hz: int = 250,
        packet_samples: int = 25,
        buffer_seconds: float = 2.0,
        sender: Optional[Sender] = None,
        label: str = "",
    ) -> None:
        self.sample_rate_hz = int(sample_rate_hz)
        self.packet_samples = max(1, int(packet_samples))
        self.port = int(port)
        self.host = host
        self.label = label or f"UDP {self.port}"
        self._sender: Optional[Sender] = (
            (sender[0], int(sender[1])) if sender is not None else None
        )

        maxlen = max(self.packet_samples * 4,
                     int(buffer_seconds * self.sample_rate_hz))
        self._buf: deque = deque(maxlen=maxlen)
        # The latest rows received, kept regardless of what the engine has
        # consumed: lets diagnostics see "whose signal is this" at any time.
        self._recent: deque = deque(maxlen=50)
        self._lock = threading.Lock()
        self._data_ready = threading.Event()

        self._hub: Optional[UdpHub] = None
        self._closed = False

        # Telemetry the UI surfaces.
        self._datagrams = 0
        self._samples = 0
        self._dropped = 0
        self._malformed = 0
        self._header_lines = 0
        self._sender_changes = 0
        self._last_rx_monotonic: Optional[float] = None
        self._layout: Optional[str] = None
        self._layout_channels: Optional[int] = None
        self._encoding: Optional[str] = None
        self._last_error = ""
        self._counter = 0
        # Last row handed to the engine: held across gaps on the wire.
        self._last_row: Optional[np.ndarray] = None
        self._in_gap = False

        # Cadence, as measured: (arrival time, samples) per datagram, the
        # Recorder's DT column (ms between samples it received from the
        # headset) and gaps in the sample counter (samples lost upstream).
        self._arrivals: deque = deque(maxlen=4096)
        self._dt_recent: deque = deque(maxlen=1500)
        self._frame_dt: Optional[np.ndarray] = None
        self._last_cnt: Optional[float] = None
        self._lost_at_source = 0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def open(self) -> None:
        """Join the hub of our port. Raises if the port is taken by another
        application or the requested sender is already in use."""
        if self._hub is not None:
            return
        hub = UdpHub.acquire(self.port, self.host)
        try:
            hub.subscribe(self, self._sender)
        except OSError:
            hub._maybe_close()
            raise
        self._hub = hub
        self._closed = False

    def close(self) -> None:
        self._closed = True
        self._data_ready.set()
        hub, self._hub = self._hub, None
        if hub is not None:
            hub.unsubscribe(self)

    def reset(self) -> None:
        with self._lock:
            self._buf.clear()
        self._counter = 0
        self._in_gap = False

    # ------------------------------------------------------------------ #
    # Telemetry                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_open(self) -> bool:
        return self._hub is not None

    @property
    def sender(self) -> Optional[Sender]:
        return self._sender

    @property
    def seconds_since_data(self) -> Optional[float]:
        """Age of the newest datagram, or None if nothing ever arrived."""
        if self._last_rx_monotonic is None:
            return None
        return time.monotonic() - self._last_rx_monotonic

    @property
    def has_signal(self) -> bool:
        """True while samples arrived recently enough to be considered live."""
        age = self.seconds_since_data
        return age is not None and age < 1.0

    @property
    def stats(self) -> dict:
        return {
            "port": self.port,
            "sender": self._sender,
            "datagrams": self._datagrams,
            "samples": self._samples,
            "dropped": self._dropped,
            "malformed": self._malformed,
            "header_lines": self._header_lines,
            "sender_changes": self._sender_changes,
            "layout": self._layout or "(in attesa di dati)",
            "channels": self._layout_channels,
            "encoding": self._encoding or "",
            "seconds_since_data": self.seconds_since_data,
            "last_error": self._last_error,
            "lost_at_source": self._lost_at_source,
        }

    def cadence(self, window_seconds: float = 5.0) -> dict:
        """How the samples are really arriving over the last few seconds.

        ``rate_hz``            samples per second actually received
        ``bursts_per_s``       how often data lands (datagrams sent back to
                               back count as one burst)
        ``samples_per_burst``  typical burst size
        ``max_pause_ms``       longest silence between two bursts
        ``datagrams_per_s``    UDP datagrams per second
        ``bt_burst_samples``   from the Recorder's DT column: how many
        ``bt_interval_ms``     samples the headset delivers per Bluetooth
                               packet, and how often
        ``lost_at_source``     gaps in the sample counter since connection

        Values that cannot be measured yet are ``None``.
        """
        now = time.perf_counter()
        with self._lock:
            arrivals = [(t, n) for t, n in self._arrivals
                        if now - t <= window_seconds]
            dts = np.asarray(self._dt_recent, dtype=np.float64)
        out = {
            "rate_hz": None, "bursts_per_s": None, "samples_per_burst": None,
            "max_pause_ms": None, "datagrams_per_s": None,
            "bt_burst_samples": None, "bt_interval_ms": None,
            "lost_at_source": self._lost_at_source,
        }
        if len(arrivals) >= 2:
            t = np.asarray([a[0] for a in arrivals])
            n = np.asarray([a[1] for a in arrivals])
            span = t[-1] - t[0]
            if span >= 0.5:
                out["rate_hz"] = float(n[1:].sum() / span)
                out["datagrams_per_s"] = (len(t) - 1) / span
                starts = np.flatnonzero(np.concatenate(
                    [[True], np.diff(t) > _SAME_BURST_SECONDS]))
                sizes = np.add.reduceat(n, starts)
                bt = t[starts]
                out["samples_per_burst"] = float(np.median(sizes))
                if len(bt) >= 2 and bt[-1] > bt[0]:
                    out["bursts_per_s"] = (len(bt) - 1) / (bt[-1] - bt[0])
                    out["max_pause_ms"] = float(np.diff(bt).max() * 1000.0)
        if dts.size >= 20:
            starts = np.flatnonzero(dts >= 1.0)
            if len(starts) >= 3:
                out["bt_burst_samples"] = float(np.median(np.diff(starts)))
                out["bt_interval_ms"] = float(np.median(dts[starts]))
        return out

    # ------------------------------------------------------------------ #
    # Called by the hub (its thread)                                     #
    # ------------------------------------------------------------------ #

    def _wants_sender(self, now: float) -> bool:
        """Free to take a new sender: none yet, or ours went quiet.

        A sender we hold but have not heard from yet is *not* "quiet": it
        was chosen (by the user, or by discovery) and is simply about to
        speak. Treating it as replaceable let whichever headset happened to
        send first at start-up override the user's choice — the panels came
        out swapped about half the time (0.9.1).
        """
        if self._closed:
            return False
        if self._sender is None:
            return True
        last = self._last_rx_monotonic
        if last is None:
            return False
        return now - last >= ADOPT_AFTER_SECONDS

    def _try_adopt(self, data: bytes, sender: Sender) -> bool:
        """Claim ``sender`` if its datagram is real Unicorn data."""
        rows = self._parse(data)
        if rows is None:
            self._malformed += 1
            return False
        if rows.shape[0] == 0:
            return False             # header only: wait for real samples
        if self._sender is not None and self._sender != sender:
            self._sender_changes += 1
        self._sender = sender
        self._accept(rows)
        return True

    def _ingest(self, data: bytes) -> None:
        rows = self._parse(data)
        if rows is None:
            self._malformed += 1
            return
        if rows.shape[0] == 0:
            return                   # header-only datagram: nothing to play
        self._accept(rows)

    def _accept(self, rows: np.ndarray) -> None:
        now = time.monotonic()
        self._datagrams += 1
        self._last_rx_monotonic = now
        dt, self._frame_dt = self._frame_dt, None
        with self._lock:
            free = self._buf.maxlen - len(self._buf)
            overflow = max(0, rows.shape[0] - free)
            if overflow:
                self._dropped += overflow
            for row in rows:
                self._buf.append(row)
                self._recent.append(row)
            self._samples += rows.shape[0]
            # perf_counter, not monotonic: on Windows (Python 3.12) the
            # latter ticks every 15.6 ms — too coarse to see 16 ms bursts.
            self._arrivals.append((time.perf_counter(), rows.shape[0]))
            if dt is not None:
                self._dt_recent.extend(dt.tolist())
            self._count_lost_at_source(rows)
        self._data_ready.set()

    def _count_lost_at_source(self, rows: np.ndarray) -> None:
        """Count holes in the sample counter (lost before reaching us)."""
        if self._layout not in (LAYOUT_UNICORN_17, LAYOUT_RECORDER_14):
            return
        cnt = rows[:, COUNTER_INDEX].astype(np.float64)
        if self._last_cnt is not None:
            cnt = np.concatenate([[self._last_cnt], cnt])
        steps = np.diff(cnt)
        jumps = steps[steps > 1.0]
        # A step backwards is a sender that restarted, not a loss.
        self._lost_at_source += int(round(float((jumps - 1.0).sum())))
        self._last_cnt = float(cnt[-1])

    def recent_rows(self) -> list:
        """Copies of the latest received rows (canonical 17 columns)."""
        with self._lock:
            return [r.copy() for r in self._recent]

    def _parse(self, data: bytes) -> Optional[np.ndarray]:
        """Decode one datagram into canonical 17-column float32 rows.

        Returns ``None`` for an undecodable datagram and an empty array for
        one that only carried a header line.
        """
        self._frame_dt = None
        decoded = decode_datagram(data, self._layout_channels)
        if decoded is None:
            return None
        frame, encoding, headers = decoded
        self._header_lines += headers
        if frame.shape[0] == 0:
            return np.zeros((0, PACKET_VALUES), dtype=np.float32)
        n_ch = frame.shape[1]
        if self._layout_channels != n_ch or self._encoding != encoding:
            self._layout_channels = n_ch
            self._layout = describe_layout(n_ch)
            self._encoding = encoding
        if self._layout == LAYOUT_RECORDER_14:
            # The Recorder's own timing of the Bluetooth link, picked up by
            # _accept for cadence(): it is not part of the canonical packet.
            self._frame_dt = frame[:, _REC_DT].astype(np.float64)
        return frame_to_packet_rows(frame)

    # ------------------------------------------------------------------ #
    # PacketSource                                                       #
    # ------------------------------------------------------------------ #

    def next_packet(self) -> Optional[UnicornPacket]:
        """Everything that has arrived, as soon as at least one sample has.

        The packet is as big as what the device delivered (typically one
        Bluetooth burst of 4 samples): it is never padded or cut to a fixed
        size on the PC's clock — that is what made the traces jump and the
        latency creep up (see the module notes).

        Returns ``None`` only once the source is closed. While it is open,
        silence on the wire for :data:`GAP_SECONDS` produces packets that
        hold the last sample, flagged invalid, one every ``packet_samples``
        worth of time, rather than stopping the engine: playback survives a
        dropout and resumes on its own.
        """
        if self._closed and self._hub is None:
            return None
        hold = self.packet_samples
        patience = (hold / float(self.sample_rate_hz) if self._in_gap
                    else GAP_SECONDS)
        deadline = time.monotonic() + patience
        while True:
            # Clear BEFORE looking, so a datagram landing in between still
            # wakes the wait below instead of being slept through.
            self._data_ready.clear()
            with self._lock:
                have = len(self._buf)
            if have or self._closed:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._data_ready.wait(timeout=remaining)

        with self._lock:
            rows = list(self._buf)
            self._buf.clear()

        if rows:
            self._in_gap = False
            samples = np.asarray(rows, dtype=np.float32)
            self._last_row = samples[-1].copy()
        else:
            self._in_gap = True
            samples = self._hold_rows(hold)
        n = samples.shape[0]

        # Keep a monotonic counter even when the sender omits one.
        if describe_layout(self._layout_channels or 0) == LAYOUT_EEG_ONLY:
            counter_vals = (self._counter + np.arange(n)) % (1 << 32)
            samples[:, COUNTER_INDEX] = counter_vals.astype(np.float32)
        self._counter = int((self._counter + n) % (1 << 32))

        return UnicornPacket(
            samples=samples,
            timestamp=time.monotonic(),
            sample_rate=self.sample_rate_hz,
        )

    def _hold_rows(self, n: int) -> np.ndarray:
        """Rows for a gap: the last real sample, held flat, marked invalid."""
        if self._last_row is not None:
            rows = np.repeat(self._last_row[None, :], n, axis=0)
        else:
            rows = np.zeros((n, PACKET_VALUES), dtype=np.float32)
        rows = rows.astype(np.float32, copy=True)
        rows[:, VALIDATION_INDEX] = 0.0
        return rows


# --------------------------------------------------------------------------- #
# Discovery                                                                   #
# --------------------------------------------------------------------------- #

def discover_streams(
    ports: tuple[int, ...] = ALT_UDP_PORTS,
    timeout: float = 1.5,
    host: str = "",
) -> list[StreamInfo]:
    """Every Unicorn stream (sender on a port) heard within ``timeout``.

    Ports already in use by a panel are not re-bound: their hub has been
    listening all along and simply reports its senders — including which
    panel claimed each one. Ports held by another application are skipped.
    """
    held: list[UdpHub] = []
    for p in ports:
        hub = UdpHub.existing(p)
        if hub is None:
            try:
                hub = UdpHub.acquire(p, host)
            except OSError:
                continue
        hub.hold()
        held.append(hub)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.05)
        found: list[StreamInfo] = []
        for hub in held:
            found.extend(hub.streams(max_age=timeout + 0.5))
        return sorted(found, key=lambda s: (s.port, s.sender))
    finally:
        for hub in held:
            hub.release_hold()


def probe_udp_ports(
    ports: tuple[int, ...] = ALT_UDP_PORTS,
    timeout: float = 1.5,
    host: str = "",
) -> list[dict]:
    """Dict view of :func:`discover_streams` (one entry per stream)."""
    return [s.as_dict() for s in discover_streams(ports, timeout, host)]
