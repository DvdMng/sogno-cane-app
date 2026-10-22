"""Two headsets, two Recorders, two panels (0.9.1).

Field report: with two BCI Core Recorder sessions running, SOGNO_CANE read
only one EEG and showed it on both the UMANO and CANE panels. Reproduced:
streams were identified by UDP port only, so

* two Recorders on the same port (the default: every Recorder sends to
  1001) were mixed sample by sample into one panel;
* SO_REUSEADDR let both panels bind the same port, so the second panel got
  nothing (unicast) or an exact copy of the first (broadcast);
* the second panel's discovery offered — and pre-selected — the stream the
  first panel was already using.

Each fake Recorder below carries a distinctive electrode offset (A = 10 000
uV, B = 90 000 uV) so a source's mean EEG says whose signal it hears.
"""
import os
import socket
import threading
import time

import numpy as np
import pytest

from sogno_cane.eeg import live as live_mod
from sogno_cane.eeg.live import (
    ADOPT_AFTER_SECONDS,
    ALT_UDP_PORTS,
    UdpHub,
    UnicornLiveSource,
    discover_streams,
)

A_OFFSET, B_OFFSET = 10000.0, 90000.0


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeRecorder(threading.Thread):
    """Recorder-format text lines at 250 Hz from its own socket."""

    def __init__(self, port, offset, dest="127.0.0.1", broadcast=False,
                 battery=80.0):
        super().__init__(daemon=True)
        self.port, self.offset, self.dest = port, offset, dest
        self.broadcast, self.battery = broadcast, battery
        self.stop_evt = threading.Event()
        self.sender_port = None
        self.ready = threading.Event()

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self.broadcast:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.bind(("127.0.0.1", 0))
        self.sender_port = s.getsockname()[1]
        self.ready.set()
        i, t0 = 0, time.monotonic()
        while not self.stop_evt.is_set():
            eeg = [self.offset + 20 * np.sin(2 * np.pi * 10 * i / 250 + k)
                   for k in range(8)]
            line = ",".join(f"{v:.3f}" for v in
                            eeg + [0, i + 1, self.battery, 1, 0.007, 0]) + "\n"
            try:
                s.sendto(line.encode(), (self.dest, self.port))
            except OSError:
                pass
            i += 1
            d = t0 + i / 250 - time.monotonic()
            if d > 0:
                time.sleep(d)
        s.close()

    def start(self):
        super().start()
        self.ready.wait(2.0)
        return self

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=2.0)


def _whose(src, seconds=2.0):
    """'A', 'B', 'MIX' or None, from samples received from now on.

    Reads the source's recent-rows history (not its buffer, which a running
    engine keeps draining) and only after fresh samples have arrived.
    """
    start = src.stats["samples"]
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and src.stats["samples"] < start + 50:
        time.sleep(0.02)
    if src.stats["samples"] < start + 50:
        return None
    rows = src.recent_rows()
    if not rows:
        return None
    first = np.array([r[0] for r in rows])
    if np.all(np.abs(first - A_OFFSET) < 2000):
        return "A"
    if np.all(np.abs(first - B_OFFSET) < 2000):
        return "B"
    return "MIX"


@pytest.fixture(autouse=True)
def _no_leftover_hubs():
    yield
    for hub in list(UdpHub._registry.values()):
        hub._stop.set()
        try:
            hub._sock.close()
        except Exception:
            pass
    UdpHub._registry.clear()


# --------------------------------------------------------------------------- #
# The reported bug, at source level                                           #
# --------------------------------------------------------------------------- #

def test_two_recorders_on_the_same_port_are_not_mixed():
    """Both Recorders on the default port: each source gets ONE headset."""
    port = _free_port()
    a = FakeRecorder(port, A_OFFSET).start()
    b = FakeRecorder(port, B_OFFSET).start()
    s1 = UnicornLiveSource(port=port, host="127.0.0.1", label="UMANO")
    s2 = UnicornLiveSource(port=port, host="127.0.0.1", label="CANE")
    try:
        s1.open()
        s2.open()                    # same port: shares the hub, no error
        got = {_whose(s1), _whose(s2)}
        assert got == {"A", "B"}, got
        assert s1.sender != s2.sender
    finally:
        for x in (s1, s2):
            x.close()
        a.stop()
        b.stop()


def test_second_panel_never_mirrors_the_first():
    """Only one Recorder streaming: the second source must stay empty
    rather than receive a copy of the first one's signal."""
    port = _free_port()
    a = FakeRecorder(port, A_OFFSET).start()
    s1 = UnicornLiveSource(port=port, host="127.0.0.1", label="UMANO")
    s2 = UnicornLiveSource(port=port, host="127.0.0.1", label="CANE")
    try:
        s1.open()
        assert _whose(s1) == "A"
        s2.open()
        time.sleep(0.6)
        assert s2.stats["samples"] == 0, "second panel mirrored the first"
    finally:
        s1.close()
        s2.close()
        a.stop()


def test_broadcast_does_not_duplicate_into_both_panels():
    port = _free_port()
    try:
        a = FakeRecorder(port, A_OFFSET, dest="255.255.255.255",
                         broadcast=True).start()
        b = FakeRecorder(port, B_OFFSET, dest="255.255.255.255",
                         broadcast=True).start()
    except OSError:
        pytest.skip("broadcast not available on this machine")
    s1 = UnicornLiveSource(port=port, label="UMANO")
    s2 = UnicornLiveSource(port=port, label="CANE")
    try:
        s1.open()
        s2.open()
        got = [_whose(s1), _whose(s2)]
        if got == [None, None]:
            pytest.skip("broadcast datagrams are not delivered here")
        assert sorted(got) == ["A", "B"], got
    finally:
        for x in (s1, s2):
            x.close()
        a.stop()
        b.stop()


# --------------------------------------------------------------------------- #
# Ports and senders are owned, not shared                                      #
# --------------------------------------------------------------------------- #

def test_port_is_bound_exclusively():
    """No other socket may silently share a port SOGNO_CANE listens on."""
    port = _free_port()
    s1 = UnicornLiveSource(port=port, host="127.0.0.1", label="UMANO")
    s1.open()
    try:
        intruder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        intruder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with pytest.raises(OSError):
            intruder.bind(("127.0.0.1", port))
        intruder.close()
    finally:
        s1.close()


def test_explicit_sender_cannot_be_claimed_twice():
    port = _free_port()
    a = FakeRecorder(port, A_OFFSET).start()
    sender = ("127.0.0.1", a.sender_port)
    s1 = UnicornLiveSource(port=port, host="127.0.0.1", sender=sender,
                           label="UMANO")
    s2 = UnicornLiveSource(port=port, host="127.0.0.1", sender=sender,
                           label="CANE")
    try:
        s1.open()
        with pytest.raises(OSError, match="UMANO"):
            s2.open()
        assert _whose(s1) == "A"
    finally:
        s1.close()
        s2.close()
        a.stop()


def test_port_is_released_when_the_last_panel_leaves():
    port = _free_port()
    s1 = UnicornLiveSource(port=port, host="127.0.0.1")
    s1.open()
    s1.close()
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", port))          # must succeed now
    probe.close()


# --------------------------------------------------------------------------- #
# Discovery                                                                   #
# --------------------------------------------------------------------------- #

def test_discovery_reports_each_sender_and_who_uses_it():
    port = _free_port()
    a = FakeRecorder(port, A_OFFSET, battery=63.0).start()
    b = FakeRecorder(port, B_OFFSET, battery=41.0).start()
    s1 = UnicornLiveSource(port=port, host="127.0.0.1", label="UMANO")
    try:
        s1.open()
        assert _whose(s1) in ("A", "B")
        found = discover_streams(ports=(port,), timeout=1.0,
                                 host="127.0.0.1")
        assert len(found) == 2
        used = [f for f in found if f.claimed_by == "UMANO"]
        free = [f for f in found if f.claimed_by is None]
        assert len(used) == 1 and len(free) == 1
        assert used[0].sender == s1.sender
        assert {f.battery for f in found} == {63.0, 41.0}
    finally:
        s1.close()
        a.stop()
        b.stop()


def test_recorder_trigger_port_is_never_probed():
    """1000 is where the Recorder LISTENS for triggers."""
    assert 1000 not in ALT_UDP_PORTS


def test_discovery_skips_a_port_held_by_another_application():
    port = _free_port()
    other = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    excl = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
    if excl is not None:
        other.setsockopt(socket.SOL_SOCKET, excl, 1)
    other.bind(("", port))
    try:
        assert discover_streams(ports=(port,), timeout=0.3) == []
    finally:
        other.close()


# --------------------------------------------------------------------------- #
# A Recorder restarting its network output (new sender address)               #
# --------------------------------------------------------------------------- #

def test_panel_follows_its_recorder_after_a_network_restart():
    port = _free_port()
    a = FakeRecorder(port, A_OFFSET).start()
    b = FakeRecorder(port, B_OFFSET).start()
    s1 = UnicornLiveSource(port=port, host="127.0.0.1", label="UMANO")
    s2 = UnicornLiveSource(port=port, host="127.0.0.1", label="CANE")
    try:
        s1.open()
        s2.open()
        first = {"UMANO": _whose(s1), "CANE": _whose(s2)}
        holder_of_a = s1 if first["UMANO"] == "A" else s2
        other = s2 if holder_of_a is s1 else s1
        old_sender = holder_of_a.sender
        a.stop()                                   # Recorder A restarts...
        time.sleep(ADOPT_AFTER_SECONDS + 0.3)
        a2 = FakeRecorder(port, A_OFFSET).start()  # ...from a new socket
        try:
            deadline = time.monotonic() + 3.0
            while (time.monotonic() < deadline
                   and holder_of_a.sender == old_sender):
                time.sleep(0.05)
            assert holder_of_a.sender == ("127.0.0.1", a2.sender_port)
            assert holder_of_a.stats["sender_changes"] == 1
            assert _whose(holder_of_a) == "A"
            assert _whose(other) == "B", "the other panel must not move"
        finally:
            a2.stop()
    finally:
        s1.close()
        s2.close()
        b.stop()


# --------------------------------------------------------------------------- #
# The two real panels, through EEG DAL VIVO                                   #
# --------------------------------------------------------------------------- #

@pytest.fixture
def qt_panels(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

    from sogno_cane.eeg.profiles import DOG_PROFILE, HUMAN_PROFILE
    from sogno_cane.io.archive import Archive
    from sogno_cane.ui.device_panel import DevicePanel

    QApplication.instance() or QApplication([])
    asked = []
    monkeypatch.setattr(
        QInputDialog, "getItem",
        staticmethod(lambda *a, **k: (asked.append(a[3]) or (a[3][0], True))),
    )
    monkeypatch.setattr(
        QInputDialog, "getInt",
        staticmethod(lambda *a, **k: (a[3], True)),
    )
    told = []
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: told.append(a[2])),
    )
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: told.append(a[2])),
    )
    arc = Archive(root=str(tmp_path / "rec"))
    h = DevicePanel("UMANO", HUMAN_PROFILE, archive=arc)
    d = DevicePanel("CANE", DOG_PROFILE, archive=arc)
    yield h, d, asked, told
    h.shutdown()
    d.shutdown()


def _scan_only(port, monkeypatch):
    monkeypatch.setattr(
        live_mod, "ALT_UDP_PORTS", (port,),
    )
    import sogno_cane.ui.device_panel as dp
    monkeypatch.setattr(dp, "ALT_UDP_PORTS", (port,))


def test_panels_split_two_recorders_sharing_the_default_port(
    qt_panels, monkeypatch,
):
    h, d, asked, told = qt_panels
    port = _free_port()
    _scan_only(port, monkeypatch)
    a = FakeRecorder(port, A_OFFSET).start()
    b = FakeRecorder(port, B_OFFSET).start()
    try:
        h._connect_live()
        d._connect_live()
        assert h.live_source is not None and d.live_source is not None
        got = {_whose(h.live_source), _whose(d.live_source)}
        assert got == {"A", "B"}, got
        # The second panel was offered only the free stream (no dialog).
        assert len(asked) == 1 and len(asked[0]) == 2
    finally:
        a.stop()
        b.stop()


def test_second_panel_is_told_instead_of_mirroring(qt_panels, monkeypatch):
    h, d, asked, told = qt_panels
    port = _free_port()
    _scan_only(port, monkeypatch)
    a = FakeRecorder(port, A_OFFSET).start()
    try:
        h._connect_live()
        assert _whose(h.live_source) == "A"
        d._connect_live()
        assert d.live_source is None, "the dog panel must not copy UMANO"
        assert told and "già collegato al pannello UMANO" in told[-1]
    finally:
        a.stop()


def test_scambia_swaps_the_two_headsets(qt_panels, monkeypatch):
    pytest.importorskip("PySide6")
    h, d, asked, told = qt_panels
    port = _free_port()
    _scan_only(port, monkeypatch)
    a = FakeRecorder(port, A_OFFSET).start()
    b = FakeRecorder(port, B_OFFSET).start()
    try:
        h._connect_live()
        d._connect_live()
        before = (_whose(h.live_source), _whose(d.live_source))
        h_live = h.detach_live_source()
        d_live = d.detach_live_source()
        h.adopt_live_source(d_live)
        d.adopt_live_source(h_live)
        after = (_whose(h.live_source), _whose(d.live_source))
        assert after == (before[1], before[0])
        assert h.is_streaming and d.is_streaming
        assert h.live_source.label == "UMANO"
    finally:
        a.stop()
        b.stop()
