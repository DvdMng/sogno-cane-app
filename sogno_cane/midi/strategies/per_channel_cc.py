"""Per-channel CC stream: one continuous CC per EEG channel.

Acts like 8 "potentiometers" that Ableton (or any DAW) can map to plugin
parameters. The user picks one band (or "broadband") per channel; the band
power is converted to a smoothed 0..127 CC value.

CC output is throttled (``min_interval_seconds``) and dead-banded
(``min_delta``) so the stream stays responsive without flooding the MIDI
bus with redundant values.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sogno_cane.eeg.bands import BandPowers
from sogno_cane.midi.context import WindowContext
from sogno_cane.midi.mapper import MappingEvent
from sogno_cane.midi.pacing import AdaptiveNorm


@dataclass
class PerChannelCCConfig:
    eeg_channel: int
    midi_channel: int
    cc_number: int
    band: str = "alpha"           # or "broadband" for total power
    log_floor: float = -1.0
    log_ceiling: float = 3.0
    smoothing: float = 0.5
    invert: bool = False
    adaptive: bool = True
    min_interval_seconds: float = 0.04
    min_delta: int = 1
    enabled: bool = True


@dataclass
class PerChannelCCStrategy:
    voices: list[PerChannelCCConfig] = field(default_factory=list)
    channel: int = 0   # not used; per-voice channel used instead
    name: str = "PerChannelCC"
    enabled: bool = True

    _smoothed: dict[int, float] = field(
        default_factory=dict, init=False, repr=False,
    )
    _last_cc: dict[tuple[int, int], int] = field(
        default_factory=dict, init=False, repr=False,
    )
    _last_t: dict[tuple[int, int], float] = field(
        default_factory=dict, init=False, repr=False,
    )
    _norms: dict[int, AdaptiveNorm] = field(
        default_factory=dict, init=False, repr=False,
    )

    @classmethod
    def default_for_n_channels(
        cls,
        n_channels: int = 8,
        midi_channel: int = 4,
        cc_start: int = 20,
        band: str = "alpha",
    ) -> "PerChannelCCStrategy":
        voices = [
            PerChannelCCConfig(
                eeg_channel=i,
                midi_channel=midi_channel,
                cc_number=cc_start + i,
                band=band,
            )
            for i in range(n_channels)
        ]
        return cls(voices=voices)

    def _norm(self, key: int) -> AdaptiveNorm:
        n = self._norms.get(key)
        if n is None:
            n = AdaptiveNorm()
            self._norms[key] = n
        return n

    def on_window(
        self,
        eeg_window: np.ndarray,
        bands: BandPowers,
        ctx: WindowContext,
    ) -> list[MappingEvent]:
        events: list[MappingEvent] = []
        for voice in self.voices:
            if not voice.enabled:
                continue
            events.extend(self._emit_for_voice(voice, bands, ctx))
        return events

    def _emit_for_voice(
        self, voice: PerChannelCCConfig, bands: BandPowers, ctx: WindowContext
    ) -> list[MappingEvent]:
        if voice.band == "broadband":
            total = bands.total_power()
            if not 0 <= voice.eeg_channel < total.shape[0]:
                return []
            p = float(total[voice.eeg_channel])
        else:
            try:
                arr = bands.by_band[voice.band]
            except KeyError:
                return []
            if not 0 <= voice.eeg_channel < arr.shape[0]:
                return []
            p = float(arr[voice.eeg_channel])

        log_p = float(np.log10(max(p, 1e-6)))
        if voice.adaptive:
            norm = self._norm(voice.eeg_channel).normalize(log_p)
        else:
            span = max(voice.log_ceiling - voice.log_floor, 1e-9)
            norm = float(np.clip((log_p - voice.log_floor) / span, 0.0, 1.0))
        if voice.invert:
            norm = 1.0 - norm

        prev = self._smoothed.get(voice.eeg_channel, norm)
        smoothed = voice.smoothing * prev + (1.0 - voice.smoothing) * norm
        self._smoothed[voice.eeg_channel] = smoothed

        cc_val = int(round(smoothed * 127.0))
        cc_val = max(0, min(127, cc_val))
        key = (voice.midi_channel, voice.cc_number)

        last = self._last_cc.get(key)
        last_t = self._last_t.get(key, -1.0e9)
        if last is not None:
            if abs(cc_val - last) < max(1, voice.min_delta):
                return []
            if ctx.t_seconds - last_t < max(0.0, voice.min_interval_seconds):
                return []
        self._last_cc[key] = cc_val
        self._last_t[key] = ctx.t_seconds
        return [MappingEvent(
            kind="cc",
            channel=voice.midi_channel,
            control=voice.cc_number,
            value=cc_val,
        )]

    def shutdown(self) -> list[MappingEvent]:
        return []
