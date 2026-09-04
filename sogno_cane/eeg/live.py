"""Live acquisition from a real Unicorn device over the network.

The app never talks to the Bluetooth dongle itself: g.tec already ships
applications that own the device and forward its samples over the network,
and we consume that stream. This keeps the portable bundle dependency-free
(UDP needs nothing but the standard library) and works with every Unicorn
generation, because g.tec's own tool handles the hardware.

Supported senders
-----------------
* **Unicorn UDP Interface** (Unicorn Suite -> DevTools -> ``UnicornUDP.exe``)
  sends one sample per datagram as **17 little-endian float32** =
  ``EEG1..8, ACC X/Y/Z, GYR X/Y/Z, BAT, CNT, VALID`` (68 bytes). That is
  byte-for-byte the layout of
  :class:`~sogno_cane.eeg.unicorn_packet.UnicornPacket`, so those datagrams
  need no translation at all.
* **Unicorn BCI Core Recorder** (g.tec Suite 2024, "Network settings" ->
  "Sending data via UDP", default port 1001) sends the channels enabled in
  its Acquisition settings, always starting with the 8 EEG channels, in the
  documented default order ``EEG1..8, SAT, CNT, BAT, VALID, DT, STATUS``
  (up to 14 floats). We map those onto the canonical packet.

Because both senders lead with the 8 EEG channels as float32, the layout is
inferred from the datagram size alone -- the user only has to pick a port.

Design notes
------------
* A daemon thread owns the socket and fills a bounded buffer, so a slow or
  stalled consumer can never grow latency without bound: the oldest samples
  are dropped (and counted) instead.
* :meth:`UnicornLiveSource.next_packet` never returns ``None`` while the
  source is open. A gap in the wire yields a silent packet instead, so the
  engine keeps running and picks the signal back up by itself when the
  device returns -- losing MIDI because of a two-second dropout is exactly
  the failure mode this project keeps fighting.
"""
from __future__ import annotations

import socket
import threading
import time
from collections import deque
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

# Default UDP ports of the two g.tec senders.
DEFAULT_UDP_PORT: int = 1001          # BCI Core Recorder "send data" default
ALT_UDP_PORTS: tuple[int, ...] = (1001, 1002, 1003, 1004, 5001, 1000)

# Channel counts we know how to interpret.
LAYOUT_UNICORN_17 = "unicorn-17"      # UDP Interface / Hybrid Black
LAYOUT_RECORDER_14 = "recorder-14"    # BCI Core Recorder, full default set
LAYOUT_EEG_ONLY = "eeg-only"          # any sender trimmed to the EEG channels

# BCI Core Recorder default column order (README "Acquisition settings").
_REC_SAT, _REC_CNT, _REC_BAT, _REC_VALID, _REC_DT, _REC_STATUS = range(8, 14)


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


class UnicornLiveSource:
    """A :class:`~sogno_cane.core.sources.PacketSource` fed by a UDP stream.

    Parameters
    ----------
    port
        UDP port to listen on (the port configured in the g.tec sender).
    host
        Interface to bind. ``""`` binds every interface, which is what you
        want when the Unicorn tool runs on another machine.
    sample_rate_hz
        Nominal device rate; the Unicorn family streams at 250 Hz.
    packet_samples
        Samples handed to the engine per call (25 = the usual 100 ms hop).
    buffer_seconds
        Bound on the receive backlog. Older samples are dropped rather than
        queued, so latency stays flat if the consumer stalls.
    """

    def __init__(
        self,
        port: int = DEFAULT_UDP_PORT,
        host: str = "",
        sample_rate_hz: int = 250,
        packet_samples: int = 25,
        buffer_seconds: float = 2.0,
    ) -> None:
        self.sample_rate_hz = int(sample_rate_hz)
        self.packet_samples = max(1, int(packet_samples))
        self.port = int(port)
        self.host = host

        maxlen = max(self.packet_samples * 4,
                     int(buffer_seconds * self.sample_rate_hz))
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._data_ready = threading.Event()

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._closed = False

        # Telemetry the UI surfaces.
        self._datagrams = 0
        self._samples = 0
        self._dropped = 0
        self._malformed = 0
        self._last_rx_monotonic: Optional[float] = None
        self._layout: Optional[str] = None
        self._layout_channels: Optional[int] = None
        self._last_error = ""
        self._counter = 0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def open(self) -> None:
        """Bind the socket and start receiving. Raises if the port is taken."""
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # A generous receive buffer absorbs scheduling hiccups at 250 Hz.
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError:
            pass
        sock.bind((self.host, self.port))
        sock.settimeout(0.2)
        self._sock = sock
        self._closed = False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._rx_loop, name=f"SOGNO_CANE-udp-{self.port}",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._closed = True
        self._stop.set()
        self._data_ready.set()
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        t, self._thread = self._thread, None
        if t is not None and t.is_alive():
            t.join(timeout=1.0)

    def reset(self) -> None:
        with self._lock:
            self._buf.clear()
        self._counter = 0

    # ------------------------------------------------------------------ #
    # Telemetry                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_open(self) -> bool:
        return self._sock is not None

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
            "datagrams": self._datagrams,
            "samples": self._samples,
            "dropped": self._dropped,
            "malformed": self._malformed,
            "layout": self._layout or "(in attesa di dati)",
            "channels": self._layout_channels,
            "seconds_since_data": self.seconds_since_data,
            "last_error": self._last_error,
        }

    # ------------------------------------------------------------------ #
    # Receive loop                                                       #
    # ------------------------------------------------------------------ #

    def _rx_loop(self) -> None:
        sock = self._sock
        while not self._stop.is_set() and sock is not None:
            try:
                data, _addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                # Socket closed underneath us, or a transient network error.
                if self._stop.is_set():
                    return
                continue
            rows = self._parse(data)
            if rows is None:
                self._malformed += 1
                continue
            self._datagrams += 1
            self._last_rx_monotonic = time.monotonic()
            with self._lock:
                free = self._buf.maxlen - len(self._buf)
                overflow = max(0, rows.shape[0] - free)
                if overflow:
                    self._dropped += overflow
                for row in rows:
                    self._buf.append(row)
                self._samples += rows.shape[0]
            self._data_ready.set()

    def _parse(self, data: bytes) -> Optional[np.ndarray]:
        """Decode one datagram into canonical 17-column float32 rows."""
        if len(data) < 4 or len(data) % 4 != 0:
            return None
        n_floats = len(data) // 4
        # Infer the channel count: reuse the layout already locked in, else
        # prefer an exact known layout that divides the payload evenly.
        n_ch = self._layout_channels
        if n_ch is None or n_floats % n_ch != 0:
            n_ch = None
            for candidate in (PACKET_VALUES, 14, EEG_CHANNELS):
                if n_floats % candidate == 0:
                    n_ch = candidate
                    break
            if n_ch is None:
                if n_floats < EEG_CHANNELS:
                    return None
                n_ch = n_floats      # single sample, unusual channel count
        try:
            flat = np.frombuffer(data, dtype="<f4")
        except Exception:
            return None
        if flat.size % n_ch != 0:
            return None
        frame = flat.reshape(-1, n_ch)
        if not np.all(np.isfinite(frame[:, :EEG_CHANNELS])):
            return None
        if self._layout_channels != n_ch:
            self._layout_channels = n_ch
            self._layout = describe_layout(n_ch)
        return frame_to_packet_rows(frame)

    # ------------------------------------------------------------------ #
    # PacketSource                                                       #
    # ------------------------------------------------------------------ #

    def next_packet(self) -> Optional[UnicornPacket]:
        """Return the next chunk, waiting briefly for the device to speak.

        Returns ``None`` only once the source is closed. While it is open a
        gap on the wire produces a silent packet rather than stopping the
        engine, so playback survives a dropout and resumes on its own.
        """
        if self._closed and self._sock is None:
            return None
        want = self.packet_samples
        deadline = time.monotonic() + (want / float(self.sample_rate_hz)) * 2.0
        while True:
            with self._lock:
                have = len(self._buf)
            if have >= want or self._stop.is_set():
                break
            if time.monotonic() >= deadline:
                break
            self._data_ready.clear()
            self._data_ready.wait(timeout=0.01)

        with self._lock:
            take = min(want, len(self._buf))
            rows = [self._buf.popleft() for _ in range(take)]

        if rows:
            samples = np.asarray(rows, dtype=np.float32)
            if samples.shape[0] < want:
                pad = np.zeros(
                    (want - samples.shape[0], PACKET_VALUES), dtype=np.float32
                )
                samples = np.vstack([samples, pad])
        else:
            # Dropout: hand the engine silence, honestly marked invalid.
            samples = np.zeros((want, PACKET_VALUES), dtype=np.float32)

        # Keep a monotonic counter even when the sender omits one.
        if describe_layout(self._layout_channels or 0) == LAYOUT_EEG_ONLY:
            counter_vals = (self._counter + np.arange(want)) % (1 << 32)
            samples[:, COUNTER_INDEX] = counter_vals.astype(np.float32)
        self._counter = int((self._counter + want) % (1 << 32))

        return UnicornPacket(
            samples=samples,
            timestamp=time.monotonic(),
            sample_rate=self.sample_rate_hz,
        )


def probe_udp_ports(
    ports: tuple[int, ...] = ALT_UDP_PORTS,
    timeout: float = 1.5,
    host: str = "",
) -> list[dict]:
    """Listen briefly on candidate ports and report which ones carry data.

    This is the plug-and-play helper: start the g.tec sender, hit DETECT,
    and the app finds the port and channel layout by itself.
    """
    found: list[dict] = []
    socks: dict = {}
    for p in ports:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, p))
            s.setblocking(False)
            socks[p] = s
        except OSError:
            # Port already in use (often by our own live source) or denied.
            continue
    try:
        deadline = time.monotonic() + timeout
        seen: dict = {}
        while time.monotonic() < deadline and len(seen) < len(socks):
            for p, s in socks.items():
                if p in seen:
                    continue
                try:
                    data, _ = s.recvfrom(65535)
                except (BlockingIOError, OSError):
                    continue
                if len(data) >= 32 and len(data) % 4 == 0:
                    seen[p] = len(data) // 4
            time.sleep(0.01)
        for p, n_floats in seen.items():
            n_ch = next(
                (c for c in (PACKET_VALUES, 14, EEG_CHANNELS)
                 if n_floats % c == 0),
                n_floats,
            )
            found.append({
                "port": p,
                "channels": n_ch,
                "layout": describe_layout(n_ch),
            })
    finally:
        for s in socks.values():
            try:
                s.close()
            except Exception:
                pass
    return sorted(found, key=lambda d: d["port"])
