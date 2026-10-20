"""Per-strategy mapping configuration panel.

Built defensively: every attribute read uses ``getattr`` with a sensible
default, and each tab is built inside a try/except so a single broken
strategy never prevents the whole window from opening. This lets users
apply our fixes in any order (or partially) without bricking the UI.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sogno_cane.midi.presets import PresetBundle
from sogno_cane.midi.scales import NOTE_NAME_TO_PC, SCALES


_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def _note_length_spinbox(current_seconds: float) -> QDoubleSpinBox:
    """Build the shared "note length" control (milliseconds).

    0 shows as "HOLD (legato)" — the note is held until the strategy itself
    releases it. Any other value is a fixed gate: the engine sends the
    note_off after exactly that long (15 ms default = short trigger that
    lets DAW envelopes attack + release freely).
    """
    sb = QDoubleSpinBox()
    sb.setRange(0.0, 5000.0)
    sb.setDecimals(0)
    sb.setSingleStep(5.0)
    sb.setSuffix(" ms")
    sb.setSpecialValueText("HOLD (legato)")
    sb.setValue(round(max(0.0, float(current_seconds)) * 1000.0))
    sb.setToolTip(
        "Durata fissa della nota MIDI, in millisecondi.\n"
        "Valori brevi (es. 15 ms) sparano note-trigger, così gli inviluppi\n"
        "del synth possono lavorare liberamente. Con 0 = HOLD la nota resta\n"
        "tenuta finché la voce non cambia altezza (legato)."
    )
    return sb


def _note_length_seconds(sb: QDoubleSpinBox) -> float:
    return max(0.0, float(sb.value())) / 1000.0


def _adaptive_controls(
    strategy, set_safe, default_threshold: float,
) -> tuple[QCheckBox, QDoubleSpinBox]:
    """Shared 'adattiva' switch + relative threshold for event strategies.

    In adaptive mode an event fires when the band rises the given number of
    typical deviations above its own recent baseline, so the setting means
    the same thing on the simulator and on any real headset. The absolute
    uV^2 threshold only applies with the switch off.
    """
    cb = QCheckBox("adattiva (consigliata con gli EEG veri)")
    cb.setChecked(bool(getattr(strategy, "adaptive", True)))
    cb.setToolTip(
        "Adattiva: scatta quando la banda sale di N volte la sua variabilità\n"
        "tipica sopra la propria media recente — funziona uguale su\n"
        "simulatore, cuffie a secco o con gel, umano o cane.\n"
        "Spenta: usa la soglia fissa in uV^2 (tarata sul simulatore: con un\n"
        "EEG vero quasi non scatta mai)."
    )
    cb.toggled.connect(lambda v: set_safe(strategy, "adaptive", bool(v)))

    sb = QDoubleSpinBox()
    sb.setRange(0.5, 8.0)
    sb.setSingleStep(0.25)
    sb.setDecimals(2)
    sb.setSuffix(" σ")
    sb.setValue(float(getattr(strategy, "relative_threshold",
                              default_threshold)))
    sb.setToolTip(
        "Quante deviazioni tipiche sopra la norma deve salire la banda per\n"
        "far scattare l'evento. Più alto = più raro."
    )
    sb.valueChanged.connect(
        lambda v: set_safe(strategy, "relative_threshold", float(v))
    )
    return cb, sb


def _safe_tab(builder, label: str) -> QWidget:
    """Run a tab builder, return a placeholder QWidget if it raises."""
    try:
        w = builder()
        if w is None:
            raise RuntimeError("builder returned None")
        return w
    except Exception as exc:
        w = QWidget()
        lay = QVBoxLayout(w)
        msg = QLabel(
            f"Scheda '{label}' non disponibile: {type(exc).__name__}: "
            f"{exc}\nDi solito significa che un modulo su disco non "
            f"combacia; il resto dell'app funziona lo stesso."
        )
        msg.setWordWrap(True)
        msg.setStyleSheet("color: #FFE372; padding: 12px;")
        lay.addWidget(msg)
        lay.addStretch(1)
        return w


class MappingPanel(QWidget):
    """Editor that mutates a :py:class:`PresetBundle` in-place."""

    def __init__(
        self, bundle: PresetBundle, title: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._bundle = bundle

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        tabs = QTabWidget()
        tabs.addTab(_safe_tab(self._build_per_channel_band_tab, "NOTE / CANALE"), "NOTE / CANALE")
        tabs.addTab(_safe_tab(self._build_per_channel_cc_tab, "CC / CANALE"),     "CC / CANALE")
        tabs.addTab(_safe_tab(self._build_threshold_tab, "EVENTI"),            "EVENTI")
        tabs.addTab(_safe_tab(self._build_coherence_tab, "COERENZA"),         "COERENZA")
        tabs.addTab(_safe_tab(self._build_markov_tab, "MARKOV"),               "MARKOV")
        tabs.addTab(_safe_tab(self._build_clip_tab, "CLIP"),                  "CLIP")
        outer.addWidget(tabs)

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _set_safe(obj, attr: str, value):
        """setattr that never raises - safe for dataclasses with __slots__."""
        try:
            setattr(obj, attr, value)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Tab builders                                                       #
    # ------------------------------------------------------------------ #

    def _build_per_channel_band_tab(self) -> QWidget:
        strategy = getattr(self._bundle, "per_channel_band", None)
        wrap = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(wrap)

        layout = QVBoxLayout(wrap)
        if strategy is None:
            layout.addWidget(QLabel("per_channel_band non presente in questa configurazione."))
            return scroll

        enable = QCheckBox("ATTIVA NOTE PER CANALE")
        enable.setChecked(getattr(strategy, "enabled", True))
        enable.toggled.connect(
            lambda v: self._set_safe(strategy, "enabled", v)
        )
        layout.addWidget(enable)

        voices_list = getattr(strategy, "voices", []) or []

        # --- PACING (live, applies to all voices) ---------------------
        pacing = QGroupBox("ANDAMENTO (vale per tutte le voci, in tempo reale)")
        pgrid = QGridLayout(pacing)

        first_interval = getattr(
            voices_list[0], "min_interval_seconds", 1.5
        ) if voices_list else 1.5
        first_duration = getattr(
            voices_list[0], "min_duration_seconds", 0.6
        ) if voices_list else 0.6
        first_threshold = getattr(
            voices_list[0], "change_threshold_norm", 0.18
        ) if voices_list else 0.18
        first_note_len = getattr(
            voices_list[0], "note_length_seconds", 0.015
        ) if voices_list else 0.015

        sb_interval = QDoubleSpinBox()
        sb_interval.setRange(0.0, 60.0)
        sb_interval.setSingleStep(0.5)
        sb_interval.setSuffix(" s")
        sb_interval.setDecimals(1)
        sb_interval.setValue(float(first_interval))
        sb_interval.setToolTip(
            "Secondi minimi fra due note della stessa voce."
        )
        def _apply_interval(v):
            for vc in voices_list:
                self._set_safe(vc, "min_interval_seconds", float(v))
        sb_interval.valueChanged.connect(_apply_interval)
        pgrid.addWidget(QLabel("intervallo minimo"), 0, 0)
        pgrid.addWidget(sb_interval, 0, 1)

        sb_duration = QDoubleSpinBox()
        sb_duration.setRange(0.0, 30.0)
        sb_duration.setSingleStep(0.2)
        sb_duration.setSuffix(" s")
        sb_duration.setDecimals(1)
        sb_duration.setValue(float(first_duration))
        sb_duration.setToolTip(
            "Secondi minimi che una nota deve durare prima di cambiare."
        )
        def _apply_duration(v):
            for vc in voices_list:
                self._set_safe(vc, "min_duration_seconds", float(v))
        sb_duration.valueChanged.connect(_apply_duration)
        pgrid.addWidget(QLabel("durata minima"), 0, 2)
        pgrid.addWidget(sb_duration, 0, 3)

        sb_thresh = QDoubleSpinBox()
        sb_thresh.setRange(0.0, 1.0)
        sb_thresh.setSingleStep(0.02)
        sb_thresh.setDecimals(2)
        sb_thresh.setValue(float(first_threshold))
        sb_thresh.setToolTip(
            "Quanto deve variare la potenza di banda (0..1) prima che la "
            "voce possa riparlare. Più alto = più rado."
        )
        def _apply_thresh(v):
            for vc in voices_list:
                self._set_safe(vc, "change_threshold_norm", float(v))
        sb_thresh.valueChanged.connect(_apply_thresh)
        pgrid.addWidget(QLabel("soglia cambio"), 1, 0)
        pgrid.addWidget(sb_thresh, 1, 1)

        sb_note_len = _note_length_spinbox(first_note_len)
        def _apply_note_len(_v):
            secs = _note_length_seconds(sb_note_len)
            for vc in voices_list:
                self._set_safe(vc, "note_length_seconds", secs)
        sb_note_len.valueChanged.connect(_apply_note_len)
        pgrid.addWidget(QLabel("durata nota"), 1, 2)
        pgrid.addWidget(sb_note_len, 1, 3)

        first_max_silence = getattr(
            voices_list[0], "max_silence_seconds", 10.0
        ) if voices_list else 10.0
        sb_max_sil = QDoubleSpinBox()
        sb_max_sil.setRange(0.0, 120.0)
        sb_max_sil.setSingleStep(1.0)
        sb_max_sil.setSuffix(" s")
        sb_max_sil.setDecimals(0)
        sb_max_sil.setSpecialValueText("OFF")
        sb_max_sil.setValue(float(first_max_silence))
        sb_max_sil.setToolTip(
            "Rete di sicurezza: se una voce tace per questo tempo mentre il\n"
            "segnale c'è, riparte comunque (anche sulla stessa nota). Tiene\n"
            "vivi tutti i canali MIDI quando la banda è troppo stabile per\n"
            "superare la soglia di cambiamento. 0 = SPENTA."
        )
        def _apply_max_sil(v):
            for vc in voices_list:
                self._set_safe(vc, "max_silence_seconds", float(v))
        sb_max_sil.valueChanged.connect(_apply_max_sil)
        pgrid.addWidget(QLabel("silenzio max"), 2, 0)
        pgrid.addWidget(sb_max_sil, 2, 1)

        layout.addWidget(pacing)

        # --- per-voice config -----------------------------------------
        for i, voice in enumerate(voices_list):
            box = QGroupBox(f"CANALE EEG {i}")
            grid = QGridLayout(box)

            cb_on = QCheckBox("attivo")
            cb_on.setChecked(getattr(voice, "enabled", True))
            cb_on.toggled.connect(
                lambda v, vc=voice: self._set_safe(vc, "enabled", v)
            )
            grid.addWidget(cb_on, 0, 0)

            band_cb = QComboBox()
            band_cb.addItems(_BANDS)
            band_cb.setCurrentText(getattr(voice, "band", "alpha"))
            band_cb.currentTextChanged.connect(
                lambda t, vc=voice: self._set_safe(vc, "band", t)
            )
            grid.addWidget(QLabel("banda"), 0, 1)
            grid.addWidget(band_cb, 0, 2)

            scale_cb = QComboBox()
            scale_cb.addItems(sorted(SCALES.keys()))
            scale_cb.setCurrentText(getattr(voice, "scale", "minor_pentatonic"))
            scale_cb.currentTextChanged.connect(
                lambda t, vc=voice: self._set_safe(vc, "scale", t)
            )
            grid.addWidget(QLabel("scala"), 0, 3)
            grid.addWidget(scale_cb, 0, 4)

            root_cb = QComboBox()
            root_cb.addItems(sorted(NOTE_NAME_TO_PC.keys()))
            root_cb.setCurrentText(getattr(voice, "root", "A"))
            root_cb.currentTextChanged.connect(
                lambda t, vc=voice: self._set_safe(vc, "root", t)
            )
            grid.addWidget(QLabel("tonica"), 1, 1)
            grid.addWidget(root_cb, 1, 2)

            midi_sb = QSpinBox()
            midi_sb.setRange(1, 16)   # 1-16 like Ableton (stored 0-15)
            midi_sb.setValue(int(getattr(voice, "midi_channel", 0)) + 1)
            midi_sb.valueChanged.connect(
                lambda v, vc=voice: self._set_safe(vc, "midi_channel", v - 1)
            )
            grid.addWidget(QLabel("canale midi"), 1, 3)
            grid.addWidget(midi_sb, 1, 4)

            layout.addWidget(box)
        layout.addStretch(1)
        return scroll

    def _build_per_channel_cc_tab(self) -> QWidget:
        strategy = getattr(self._bundle, "per_channel_cc", None)
        wrap = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(wrap)

        layout = QVBoxLayout(wrap)
        if strategy is None:
            layout.addWidget(QLabel("per_channel_cc non presente in questa configurazione."))
            return scroll

        enable = QCheckBox("ATTIVA CC PER CANALE")
        enable.setChecked(getattr(strategy, "enabled", False))
        enable.toggled.connect(
            lambda v: self._set_safe(strategy, "enabled", v)
        )
        layout.addWidget(enable)

        for i, v in enumerate(getattr(strategy, "voices", []) or []):
            box = QGroupBox(f"CANALE EEG {i}")
            grid = QGridLayout(box)

            cb_on = QCheckBox("attivo")
            cb_on.setChecked(getattr(v, "enabled", True))
            cb_on.toggled.connect(
                lambda val, vc=v: self._set_safe(vc, "enabled", val)
            )
            grid.addWidget(cb_on, 0, 0)

            band_cb = QComboBox()
            band_cb.addItems(_BANDS + ["broadband"])
            band_cb.setCurrentText(getattr(v, "band", "alpha"))
            band_cb.currentTextChanged.connect(
                lambda t, vc=v: self._set_safe(vc, "band", t)
            )
            grid.addWidget(QLabel("banda"), 0, 1)
            grid.addWidget(band_cb, 0, 2)

            cc_sb = QSpinBox()
            cc_sb.setRange(0, 127)
            cc_sb.setValue(int(getattr(v, "cc_number", 20 + i)))
            cc_sb.valueChanged.connect(
                lambda val, vc=v: self._set_safe(vc, "cc_number", val)
            )
            grid.addWidget(QLabel("cc#"), 0, 3)
            grid.addWidget(cc_sb, 0, 4)

            midi_sb = QSpinBox()
            midi_sb.setRange(1, 16)   # 1-16 like Ableton (stored 0-15)
            midi_sb.setValue(int(getattr(v, "midi_channel", 12)) + 1)
            midi_sb.valueChanged.connect(
                lambda val, vc=v: self._set_safe(vc, "midi_channel", val - 1)
            )
            grid.addWidget(QLabel("canale midi"), 1, 1)
            grid.addWidget(midi_sb, 1, 2)

            inv_cb = QCheckBox("inverti")
            inv_cb.setChecked(getattr(v, "invert", False))
            inv_cb.toggled.connect(
                lambda val, vc=v: self._set_safe(vc, "invert", val)
            )
            grid.addWidget(inv_cb, 1, 3)

            sm = QDoubleSpinBox()
            sm.setRange(0.0, 0.99)
            sm.setSingleStep(0.05)
            sm.setValue(float(getattr(v, "smoothing", 0.7)))
            sm.valueChanged.connect(
                lambda val, vc=v: self._set_safe(vc, "smoothing", val)
            )
            grid.addWidget(QLabel("morbidezza"), 1, 4)
            grid.addWidget(sm, 1, 5)

            layout.addWidget(box)
        layout.addStretch(1)
        return scroll

    def _build_threshold_tab(self) -> QWidget:
        s = getattr(self._bundle, "threshold", None)
        w = QWidget()
        layout = QFormLayout(w)
        if s is None:
            layout.addRow(QLabel("strategia soglia non presente in questa configurazione."))
            return w

        enable = QCheckBox()
        enable.setChecked(getattr(s, "enabled", False))
        enable.toggled.connect(lambda v: self._set_safe(s, "enabled", v))
        layout.addRow("ATTIVA", enable)

        band_cb = QComboBox()
        band_cb.addItems(_BANDS)
        band_cb.setCurrentText(getattr(s, "band", "beta"))
        band_cb.currentTextChanged.connect(
            lambda t: self._set_safe(s, "band", t)
        )
        layout.addRow("banda", band_cb)

        adaptive_cb, rel_sb = _adaptive_controls(s, self._set_safe, 2.0)
        layout.addRow("", adaptive_cb)
        layout.addRow("soglia relativa", rel_sb)

        thr = QDoubleSpinBox()
        thr.setRange(0.0, 10_000.0)
        thr.setValue(float(getattr(s, "threshold_uv2", 80.0)))
        thr.setSuffix(" uV^2")
        thr.setToolTip("Soglia assoluta: usata solo se 'adattiva' è spenta.")
        thr.valueChanged.connect(
            lambda v: self._set_safe(s, "threshold_uv2", v)
        )
        layout.addRow("soglia fissa", thr)
        thr.setEnabled(not adaptive_cb.isChecked())
        rel_sb.setEnabled(adaptive_cb.isChecked())
        adaptive_cb.toggled.connect(lambda v: thr.setEnabled(not v))
        adaptive_cb.toggled.connect(rel_sb.setEnabled)

        ch = QSpinBox()
        ch.setRange(1, 16)   # 1-16 like Ableton (stored 0-15)
        ch.setValue(int(getattr(s, "channel", 9)) + 1)
        ch.valueChanged.connect(lambda v: self._set_safe(s, "channel", v - 1))
        layout.addRow("canale midi", ch)

        nl = _note_length_spinbox(getattr(s, "note_length_seconds", 0.015))
        nl.valueChanged.connect(
            lambda _v, sb=nl: self._set_safe(
                s, "note_length_seconds", _note_length_seconds(sb)
            )
        )
        layout.addRow("durata nota", nl)
        return w

    def _build_coherence_tab(self) -> QWidget:
        s = getattr(self._bundle, "coherence", None)
        w = QWidget()
        layout = QFormLayout(w)
        if s is None:
            layout.addRow(QLabel("strategia coerenza non presente in questa configurazione."))
            return w

        enable = QCheckBox()
        enable.setChecked(getattr(s, "enabled", False))
        enable.toggled.connect(lambda v: self._set_safe(s, "enabled", v))
        layout.addRow("ATTIVA", enable)

        band_cb = QComboBox()
        band_cb.addItems(_BANDS)
        band_cb.setCurrentText(getattr(s, "band", "alpha"))
        band_cb.currentTextChanged.connect(
            lambda t: self._set_safe(s, "band", t)
        )
        layout.addRow("banda", band_cb)

        cc = QSpinBox()
        cc.setRange(0, 127)
        cc.setValue(int(getattr(s, "cc_number", 1)))
        cc.valueChanged.connect(lambda v: self._set_safe(s, "cc_number", v))
        layout.addRow("cc #", cc)

        ch = QSpinBox()
        ch.setRange(1, 16)   # 1-16 like Ableton (stored 0-15)
        ch.setValue(int(getattr(s, "channel", 10)) + 1)
        ch.valueChanged.connect(lambda v: self._set_safe(s, "channel", v - 1))
        layout.addRow("canale midi", ch)

        sm = QDoubleSpinBox()
        sm.setRange(0.0, 0.99)
        sm.setSingleStep(0.05)
        sm.setValue(float(getattr(s, "smoothing", 0.6)))
        sm.valueChanged.connect(lambda v: self._set_safe(s, "smoothing", v))
        layout.addRow("morbidezza", sm)
        return w

    def _build_markov_tab(self) -> QWidget:
        s = getattr(self._bundle, "markov", None)
        w = QWidget()
        layout = QFormLayout(w)
        if s is None:
            layout.addRow(QLabel("strategia markov non presente in questa configurazione."))
            return w

        enable = QCheckBox()
        enable.setChecked(getattr(s, "enabled", True))
        enable.toggled.connect(lambda v: self._set_safe(s, "enabled", v))
        layout.addRow("ATTIVA", enable)

        scale_cb = QComboBox()
        scale_cb.addItems(sorted(SCALES.keys()))
        scale_cb.setCurrentText(getattr(s, "scale", "dorian"))
        scale_cb.currentTextChanged.connect(
            lambda t: self._set_safe(s, "scale", t)
        )
        layout.addRow("scala", scale_cb)

        root_cb = QComboBox()
        root_cb.addItems(sorted(NOTE_NAME_TO_PC.keys()))
        root_cb.setCurrentText(getattr(s, "root", "D"))
        root_cb.currentTextChanged.connect(
            lambda t: self._set_safe(s, "root", t)
        )
        layout.addRow("tonica", root_cb)

        ch = QSpinBox()
        ch.setRange(1, 16)   # 1-16 like Ableton (stored 0-15)
        ch.setValue(int(getattr(s, "channel", 8)) + 1)
        ch.valueChanged.connect(lambda v: self._set_safe(s, "channel", v - 1))
        layout.addRow("canale midi", ch)

        density = QSpinBox()
        density.setRange(1, 16)
        density.setValue(int(getattr(s, "max_notes_per_window", 1)))
        density.valueChanged.connect(
            lambda v: self._set_safe(s, "max_notes_per_window", v)
        )
        layout.addRow("note max/finestra", density)

        mk_interval = QDoubleSpinBox()
        mk_interval.setRange(0.0, 60.0)
        mk_interval.setSingleStep(0.5)
        mk_interval.setDecimals(1)
        mk_interval.setSuffix(" s")
        mk_interval.setValue(float(getattr(s, "min_interval_seconds", 2.5)))
        mk_interval.setToolTip(
            "Secondi minimi fra due frasi Markov. Più alto = più rado."
        )
        mk_interval.valueChanged.connect(
            lambda v: self._set_safe(s, "min_interval_seconds", float(v))
        )
        layout.addRow("intervallo minimo", mk_interval)

        nl = _note_length_spinbox(getattr(s, "note_length_seconds", 0.015))
        nl.valueChanged.connect(
            lambda _v, sb=nl: self._set_safe(
                s, "note_length_seconds", _note_length_seconds(sb)
            )
        )
        layout.addRow("durata nota", nl)
        return w

    def _build_clip_tab(self) -> QWidget:
        s = getattr(self._bundle, "clips", None)
        wrap = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(wrap)

        layout = QVBoxLayout(wrap)
        if s is None:
            layout.addWidget(QLabel("strategia clip non presente in questa configurazione."))
            return scroll

        enable = QCheckBox("ATTIVA LANCIA-CLIP")
        enable.setChecked(getattr(s, "enabled", False))
        enable.toggled.connect(lambda v: self._set_safe(s, "enabled", v))
        layout.addWidget(enable)

        adaptive_box = QGroupBox("SCATTO (vale per tutte le regole)")
        agrid = QGridLayout(adaptive_box)
        adaptive_cb, rel_sb = _adaptive_controls(s, self._set_safe, 3.0)
        agrid.addWidget(adaptive_cb, 0, 0, 1, 2)
        agrid.addWidget(QLabel("soglia relativa"), 1, 0)
        agrid.addWidget(rel_sb, 1, 1)
        rel_sb.setEnabled(adaptive_cb.isChecked())
        adaptive_cb.toggled.connect(rel_sb.setEnabled)
        layout.addWidget(adaptive_box)

        for i, rule in enumerate(getattr(s, "rules", []) or []):
            box = QGroupBox(f"REGOLA {i}")
            grid = QGridLayout(box)

            on_cb = QCheckBox("attivo")
            on_cb.setChecked(getattr(rule, "enabled", True))
            on_cb.toggled.connect(
                lambda v, r=rule: self._set_safe(r, "enabled", v)
            )
            grid.addWidget(on_cb, 0, 0)

            ch_sb = QSpinBox()
            ch_sb.setRange(0, 7)
            ch_sb.setValue(int(getattr(rule, "eeg_channel", 0)))
            ch_sb.valueChanged.connect(
                lambda v, r=rule: self._set_safe(r, "eeg_channel", v)
            )
            grid.addWidget(QLabel("canale eeg"), 0, 1)
            grid.addWidget(ch_sb, 0, 2)

            band_cb = QComboBox()
            band_cb.addItems(_BANDS)
            band_cb.setCurrentText(getattr(rule, "band", "theta"))
            band_cb.currentTextChanged.connect(
                lambda t, r=rule: self._set_safe(r, "band", t)
            )
            grid.addWidget(QLabel("banda"), 0, 3)
            grid.addWidget(band_cb, 0, 4)

            note_sb = QSpinBox()
            note_sb.setRange(0, 127)
            note_sb.setValue(int(getattr(rule, "midi_note", 36 + i)))
            note_sb.valueChanged.connect(
                lambda v, r=rule: self._set_safe(r, "midi_note", v)
            )
            grid.addWidget(QLabel("nota"), 1, 1)
            grid.addWidget(note_sb, 1, 2)

            mc = QSpinBox()
            mc.setRange(1, 16)   # 1-16 like Ableton (stored 0-15)
            mc.setValue(int(getattr(rule, "midi_channel", 11)) + 1)
            mc.valueChanged.connect(
                lambda v, r=rule: self._set_safe(r, "midi_channel", v - 1)
            )
            grid.addWidget(QLabel("canale midi"), 1, 3)
            grid.addWidget(mc, 1, 4)

            thr = QDoubleSpinBox()
            thr.setRange(0.0, 10_000.0)
            thr.setValue(float(getattr(rule, "threshold_uv2", 60.0)))
            thr.valueChanged.connect(
                lambda v, r=rule: self._set_safe(r, "threshold_uv2", v)
            )
            thr.setSuffix(" uV^2")
            thr.setToolTip(
                "Soglia assoluta: usata solo se 'adattiva' è spenta."
            )
            thr.setEnabled(not adaptive_cb.isChecked())
            adaptive_cb.toggled.connect(
                lambda v, w=thr: w.setEnabled(not v)
            )
            grid.addWidget(QLabel("soglia fissa"), 2, 1)
            grid.addWidget(thr, 2, 2)

            layout.addWidget(box)
        layout.addStretch(1)
        return scroll
