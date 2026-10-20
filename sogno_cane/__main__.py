"""Entry point for SOGNO_CANE.

Usage:
    python -m sogno_cane             # launch the GUI
    python -m sogno_cane --selftest  # headless self-test (no window)
    python -m sogno_cane --live-test # look for a real Unicorn UDP stream
    python -m sogno_cane --version

The GUI path installs a last-resort exception hook so that, in the portable
``pythonw.exe`` build (which has no console), an unhandled error surfaces in a
dialog and is written to a log file instead of vanishing silently.
"""
from __future__ import annotations

import os
import sys
import traceback


def _log_path() -> str:
    base = os.environ.get("SOGNO_CANE_HOME") or os.path.join(
        os.path.expanduser("~"), ".sogno_cane"
    )
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        base = os.path.expanduser("~")
    return os.path.join(base, "last_error.log")


def _selftest() -> int:
    """Exercise the full non-GUI pipeline; print OK or the failing error."""
    import numpy as np

    from sogno_cane.core.engine import RealtimeEngine
    from sogno_cane.core.sources import ArrayPlaybackSource
    from sogno_cane.eeg.profiles import DOG_PROFILE, HUMAN_PROFILE
    from sogno_cane.eeg.simulator import EEGSimulator
    from sogno_cane.midi.mapper import MappingConfig, MappingEngine
    from sogno_cane.midi.output import MidiOutput, list_output_ports
    from sogno_cane.midi.presets import rich_vocabulary_preset

    print(f"python {sys.version.split()[0]}")
    print(f"numpy {np.__version__}")
    print("MIDI output ports:", list_output_ports() or "(none)")

    for profile in (HUMAN_PROFILE, DOG_PROFILE):
        mapper = MappingEngine(
            MappingConfig(sample_rate_hz=profile.sample_rate_hz)
        )
        for s in rich_vocabulary_preset().as_list():
            mapper.add_strategy(s)
        sim = EEGSimulator(profile=profile, seed=1)
        n_events = 0
        for _ in range(100):  # 10 s of simulation
            n_events += len(mapper.process_packet(sim.next_packet()))
        print(f"  {profile.name}: {n_events} MIDI events / 10 s simulated")

    # Playback path.
    eeg = np.zeros((500, 8), dtype=np.float32)
    src = ArrayPlaybackSource(eeg, 250, loop=False)
    eng = RealtimeEngine(source=src, midi=MidiOutput())
    assert eng.source is src
    print("selftest OK")
    return 0


def _live_test(argv: list[str]) -> int:
    """Listen for a real Unicorn stream and report what arrives.

    On-site troubleshooting without opening the GUI: answers the only two
    questions that matter when nothing is heard — is anything reaching this
    machine, and on which port?

        python -m sogno_cane --live-test          # scan the usual ports
        python -m sogno_cane --live-test 1002     # listen on one port
    """
    import time

    from sogno_cane.eeg.live import (
        ALT_UDP_PORTS,
        UnicornLiveSource,
        probe_udp_ports,
    )

    port = None
    for i, a in enumerate(argv):
        if a == "--live-test" and i + 1 < len(argv):
            nxt = argv[i + 1]
            if nxt.isdigit():
                port = int(nxt)
            break

    if port is None:
        print(f"Scanning UDP ports {list(ALT_UDP_PORTS)} for 3 s...")
        found = probe_udp_ports(ports=ALT_UDP_PORTS, timeout=3.0)
        if not found:
            print(
                "No Unicorn stream found.\n"
                "Start g.tec's sender first:\n"
                "  Unicorn BCI Core Recorder -> Settings -> Network ->\n"
                "    'Sending data via UDP'   (default port 1001), or\n"
                "  Unicorn Suite -> DevTools -> UnicornUDP.exe\n"
                "If only 'Sending data via LSL' is on in the Recorder, also\n"
                "turn on 'Sending data via UDP': SOGNO_CANE listens on UDP."
            )
            return 1
        for f in found:
            print(
                f"  found: UDP {f['port']} - {f['channels']} channels "
                f"({f['layout']}, {f.get('encoding', '?')})"
            )
        port = int(found[0]["port"])
        print(f"Listening on the first one (UDP {port})...")

    src = UnicornLiveSource(port=port)
    try:
        src.open()
    except OSError as exc:
        print(f"Cannot listen on UDP {port}: {exc}")
        return 1
    try:
        import numpy as np

        deadline = time.monotonic() + 5.0
        chunks = []
        while time.monotonic() < deadline:
            pkt = src.next_packet()
            if pkt is not None:
                chunks.append(pkt.samples.copy())
        st = src.stats
        print(
            f"UDP {st['port']}: {st['datagrams']} datagrams, "
            f"{st['samples']} samples, layout {st['layout']} "
            f"({st['channels']} channels, {st['encoding'] or '?'}), "
            f"dropped {st['dropped']}, malformed {st['malformed']}"
        )
        if st["samples"] == 0:
            print("NO DATA - the sender is not transmitting to this port.")
            return 1
        data = np.vstack(chunks)
        live = data[data[:, 16] >= 0.5]          # samples flagged valid
        lost = 1.0 - live.shape[0] / max(1, data.shape[0])
        if live.shape[0] >= 8:
            eeg = live[:, :8].astype(np.float64)
            offset = eeg.mean(axis=0)
            centred = eeg - offset
            rms = centred.std(axis=0)
            # Unicorns are DC-coupled: the raw level is the electrode
            # offset; the brain signal is the (much smaller) part around it.
            print("electrode offset (uV): "
                  + ", ".join(f"{v:.0f}" for v in offset))
            print("signal around it, rms (uV): "
                  + ", ".join(f"{v:.1f}" for v in rms))
            # Separate the mains hum (what the analysis ignores) from EEG,
            # so a noisy electrode is told apart from a strong signal.
            spec = np.fft.rfft(centred, axis=0)
            freqs = np.fft.rfftfreq(centred.shape[0], 1.0 / src.sample_rate_hz)
            hum = np.zeros_like(freqs, dtype=bool)
            for h in (50.0, 100.0):
                hum |= np.abs(freqs - h) <= 2.0
            eeg_only = spec.copy()
            eeg_only[hum] = 0.0
            clean = np.fft.irfft(eeg_only, n=centred.shape[0], axis=0)
            print("of which EEG without 50 Hz hum, rms (uV): "
                  + ", ".join(f"{v:.1f}" for v in clean.std(axis=0)))
        print(f"samples lost or not yet received: {lost * 100:.0f}%")
        print("live acquisition OK")
        return 0
    finally:
        src.close()


def _install_excepthook(app) -> None:
    from PySide6.QtWidgets import QMessageBox

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            with open(_log_path(), "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass
        try:
            QMessageBox.critical(
                None, "SOGNO_CANE — error",
                f"An unexpected error occurred:\n\n{exc}\n\n"
                f"Details were written to:\n{_log_path()}",
            )
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = hook


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    if "--version" in argv:
        from sogno_cane import __version__
        print(f"SOGNO_CANE {__version__}")
        return 0
    if "--selftest" in argv or "--check" in argv:
        try:
            return _selftest()
        except Exception:
            traceback.print_exc()
            return 1
    if "--live-test" in argv:
        try:
            return _live_test(argv)
        except Exception:
            traceback.print_exc()
            return 1

    from PySide6.QtWidgets import QApplication

    from sogno_cane.ui.main_window import MainWindow
    from sogno_cane.ui.theme import apply_theme

    app = QApplication(argv)
    app.setApplicationName("SOGNO_CANE")
    app.setOrganizationName("SOGNO_CANE")
    _install_excepthook(app)
    apply_theme(app)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
