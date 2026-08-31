"""Per-channel band -> note routing.

Each EEG channel becomes an independent musical voice: its own MIDI channel,
its own scale, its own root, its own velocity dynamics. This yields up to
8 simultaneous melodic streams per device (16 total when human + dog are
both running).

Each voice is *paced*: it respects a minimum interval between notes, a
minimum hold time, and a hysteresis threshold on the driving value, so the
output is sparse and musical instead of retriggering on every analysis hop.
Each voice also normalises its band power with a slowly-adapting AGC window
(:class:`~sogno_cane.midi.pacing.AdaptiveNorm`) so the full pitch range is
used regardless of the signal's absolute scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sogno_cane.eeg.bands import BandPowers
from sogno_cane.midi.context import WindowContext
from sogno_cane.midi.mapper import MappingEvent
from sogno_cane.midi.pacing import AdaptiveNorm, NoteGate
from sogno_cane.midi.scales import (
    MidiNoteRange,
    build_scale,
    quantize_to_scale,
)


@dataclass
class PerChannelVoice:
    """Voice config for a single EEG channel.

    Pacing controls (``min_interval_seconds``, ``min_duration_seconds``,
    ``change_threshold_norm``) are honoured live and are what the mapping
    panel edits.
    """

    eeg_channel: int
    midi_channel: int
    band: str = "alpha"
    scale: str = "minor_pentatonic"
    root: str = "A"
    note_range: MidiNoteRange = field(default_factory=MidiNoteRange)
    log_floor: float = 0.0
    log_ceiling: float = 3.0
    velocity_min: int = 40
    velocity_max: int = 115
    sustain: bool = True
    # --- musical pacing (read live by the strategy) ---
    min_interval_seconds: float = 1.5
    min_duration_seconds: float = 0.6
    change_threshold_norm: float = 0.18
    # Fixed note length (gate) in seconds. When > 0 every note_on carries a
    # duration and the engine sends the note_off after exactly that long
    # (15 ms default: short trigger that lets DAW envelopes modulate).
    # 0 = hold the note until the voice changes pitch (legato, old behaviour).
    note_length_seconds: float = 0.015
    # Liveness failsafe (seconds, 0 = off): if the voice has emitted nothing
    # for this long while the signal is audible, retrigger anyway (bypassing
    # the hysteresis and the same-pitch dedup). A stable band power can
    # otherwise mute a voice indefinitely — heard as "one of the MIDI
    # channels stopped transmitting".
    max_silence_seconds: float = 10.0
    # --- normalisation mode ---
    adaptive: bool = True
    enabled: bool = True


@dataclass
class PerChannelBandStrategy:
    """One independent band->note voice per EEG channel."""

    voices: list[PerChannelVoice] = field(default_factory=list)
    channel: int = 0   # ignored; per-voice channel used instead
    name: str = "PerChannelBand"
    enabled: bool = True

    _last_notes: dict[int, int] = field(
        default_factory=dict, init=False, repr=False,
    )
    # True if the voice's last note_on was gated (carried a duration): the
    # engine owns its release, so the strategy must not emit a second
    # note_off for it.
    _gated: dict[int, bool] = field(
        default_factory=dict, init=False, repr=False,
    )
    _gates: dict[int, NoteGate] = field(
        default_factory=dict, init=False, repr=False,
    )
    _norms: dict[int, AdaptiveNorm] = field(
        default_factory=dict, init=False, repr=False,
    )

    @classmethod
    def default_for_n_channels(
        cls,
        n_channels: int = 8,
        midi_channel_start: int = 0,
        band: str = "alpha",
        scale: str = "minor_pentatonic",
        root: str = "A",
    ) -> "PerChannelBandStrategy":
        """Build N voices on consecutive MIDI channels, all on the same scale."""
        voices = [
            PerChannelVoice(
                eeg_channel=i,
                midi_channel=(midi_channel_start + i) % 16,
                band=band,
                scale=scale,
                root=root,
            )
            for i in range(n_channels)
        ]
        return cls(voices=voices)

    # ------------------------------------------------------------------ #

    def _gate(self, key: int) -> NoteGate:
        g = self._gates.get(key)
        if g is None:
            g = NoteGate()
            self._gates[key] = g
        return g

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
        self, voice: PerChannelVoice, bands: BandPowers, ctx: WindowContext
    ) -> list[MappingEvent]:
        try:
            band_power = bands.by_band[voice.band]
        except KeyError:
            return []
        if not 0 <= voice.eeg_channel < band_power.shape[0]:
            return []
        power = float(band_power[voice.eeg_channel])
        log_p = float(np.log10(max(power, 1e-6)))

        if voice.adaptive:
            norm = self._norm(voice.eeg_channel).normalize(log_p)
        else:
            span = max(voice.log_ceiling - voice.log_floor, 1e-9)
            norm = float(
                np.clip((log_p - voice.log_floor) / span, 0.0, 1.0)
            )

        scale_notes = build_scale(voice.scale, voice.root, voice.note_range)
        if not scale_notes:
            return []

        events: list[MappingEvent] = []
        gate = self._gate(voice.eeg_channel)
        prev_note = self._last_notes.get(voice.eeg_channel)
        t = ctx.t_seconds

        # Near-silence: release the held note (subject to min hold).
        if norm <= 0.02:
            if prev_note is not None and gate.may_release(
                t, min_hold=voice.min_duration_seconds
            ):
                if not self._gated.get(voice.eeg_channel):
                    events.append(MappingEvent(
                        kind="note_off",
                        channel=voice.midi_channel,
                        note=prev_note,
                    ))
                self._last_notes.pop(voice.eeg_channel, None)
                self._gated.pop(voice.eeg_channel, None)
                gate.commit_off()
            return events

        new_note = quantize_to_scale(norm, scale_notes)
        velocity = int(round(
            voice.velocity_min
            + norm * (voice.velocity_max - voice.velocity_min)
        ))
        velocity = max(1, min(127, velocity))

        if voice.sustain:
            starved = gate.is_starved(
                t, max_silence=voice.max_silence_seconds
            )
            # Same pitch normally means "nothing to do" — unless the voice
            # has been starved quiet, in which case we retrigger it so the
            # channel stays audibly alive.
            if new_note == prev_note and not starved:
                return events
            allowed = gate.may_change(
                t, norm,
                min_interval=voice.min_interval_seconds,
                change_threshold=voice.change_threshold_norm,
                max_silence=voice.max_silence_seconds,
            )
            if prev_note is not None and not gate.may_release(
                t, min_hold=voice.min_duration_seconds
            ):
                allowed = False
            if not allowed:
                return events
            gate_len = max(0.0, voice.note_length_seconds)
            if prev_note is not None and not self._gated.get(
                voice.eeg_channel
            ):
                events.append(MappingEvent(
                    kind="note_off",
                    channel=voice.midi_channel,
                    note=prev_note,
                ))
            events.append(MappingEvent(
                kind="note_on",
                channel=voice.midi_channel,
                note=new_note,
                velocity=velocity,
                duration=gate_len,
            ))
            self._last_notes[voice.eeg_channel] = new_note
            self._gated[voice.eeg_channel] = gate_len > 0.0
            gate.commit_on(t, norm)
        else:
            # Retrigger mode, but still rate-limited.
            if not gate.may_change(
                t, norm,
                min_interval=voice.min_interval_seconds,
                change_threshold=0.0,
            ):
                return events
            gate_len = max(0.0, voice.note_length_seconds)
            if prev_note is not None and not self._gated.get(
                voice.eeg_channel
            ):
                events.append(MappingEvent(
                    kind="note_off",
                    channel=voice.midi_channel,
                    note=prev_note,
                ))
            events.append(MappingEvent(
                kind="note_on",
                channel=voice.midi_channel,
                note=new_note,
                velocity=velocity,
                duration=gate_len,
            ))
            self._last_notes[voice.eeg_channel] = new_note
            self._gated[voice.eeg_channel] = gate_len > 0.0
            gate.commit_on(t, norm)
        return events

    def reset_runtime(self) -> None:
        """Forget per-run state (remembered notes, gate timers).

        Called when a fresh engine starts: its simulation clock restarts at
        zero, so timestamps and note memory from the previous run must not
        leak in (they would mute voices for the previous run's duration).
        The AGC calibration survives on purpose — the signal scale did not
        change across a stop/start.
        """
        self._last_notes.clear()
        self._gated.clear()
        for g in self._gates.values():
            g.reset()

    def shutdown(self) -> list[MappingEvent]:
        events: list[MappingEvent] = []
        for voice in self.voices:
            note = self._last_notes.pop(voice.eeg_channel, None)
            gated = self._gated.pop(voice.eeg_channel, False)
            if note is not None and not gated:
                events.append(MappingEvent(
                    kind="note_off",
                    channel=voice.midi_channel,
                    note=note,
                ))
            g = self._gates.get(voice.eeg_channel)
            if g is not None:
                g.commit_off()
        return events
