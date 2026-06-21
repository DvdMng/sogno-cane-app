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
            f"Tab '{label}' unavailable: {type(exc).__name__}: {exc}\n"
            f"This usually means a mismatched module on disk; the rest of "
            f"the app still works."
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
        tabs.addTab(_safe_tab(self._build_per_channel_band_tab, "NOTES / CH"), "NOTES / CH")
        tabs.addTab(_safe_tab(self._build_per_channel_cc_tab, "CC / CH"),     "CC / CH")
        tabs.addTab(_safe_tab(self._build_threshold_tab, "EVENTS"),            "EVENTS")
        tabs.addTab(_safe_tab(self._build_coherence_tab, "COHERENCE"),         "COHERENCE")
        tabs.addTab(_safe_tab(self._build_markov_tab, "MARKOV"),               "MARKOV")
        tabs.addTab(_safe_tab(self._build_clip_tab, "CLIPS"),                  "CLIPS")
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
            layout.addWidget(QLabel("per_channel_band not in this bundle."))
            return scroll

        enable = QCheckBox("ENABLE PER-CHANNEL NOTES")
        enable.setChecked(getattr(strategy, "enabled", True))
        enable.toggled.connect(
            lambda v: self._set_safe(strategy, "enabled", v)
        )
        layout.addWidget(enable)

        voices_list = getattr(strategy, "voices", []) or []

        # --- PACING (live, applies to all voices) ---------------------
        pacing = QGroupBox("PACING (applies to all voices in real time)")
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

        sb_interval = QDoubleSpinBox()
        sb_interval.setRange(0.0, 60.0)
        sb_interval.setSingleStep(0.5)
        sb_interval.setSuffix(" s")
        sb_interval.setDecimals(1)
        sb_interval.setValue(float(first_interval))
        sb_interval.setToolTip(
            "Min seconds between two consecutive notes from the same voice."
        )
        def _apply_interval(v):
            for vc in voices_list:
                self._set_safe(vc, "min_interval_seconds", float(v))
        sb_interval.valueChanged.connect(_apply_interval)
        pgrid.addWidget(QLabel("min interval"), 0, 0)
        pgrid.addWidget(sb_interval, 0, 1)

        sb_duration = QDoubleSpinBox()
        sb_duration.setRange(0.0, 30.0)
        sb_duration.setSingleStep(0.2)
        sb_duration.setSuffix(" s")
        sb_duration.setDecimals(1)
        sb_duration.setValue(float(first_duration))
        sb_duration.setToolTip(
            "Min seconds a note must hold before it can be changed/released."
        )
        def _apply_duration(v):
            for vc in voices_list:
                self._set_safe(vc, "min_duration_seconds", float(v))
        sb_duration.valueChanged.connect(_apply_duration)
        pgrid.addWidget(QLabel("min hold"), 0, 2)
        pgrid.addWidget(sb_duration, 0, 3)

        sb_thresh = QDoubleSpinBox()
        sb_thresh.setRange(0.0, 1.0)
        sb_thresh.setSingleStep(0.02)
        sb_thresh.setDecimals(2)
        sb_thresh.setValue(float(first_threshold))
        sb_thresh.setToolTip(
            "How much the band power must shift (0..1 normalised) before "
            "the voice may speak again. Higher = more sparse."
        )
        def _apply_thresh(v):
            for vc in voices_list:
                self._set_safe(vc, "change_threshold_norm", float(v))
        sb_thresh.valueChanged.connect(_apply_thresh)
        pgrid.addWidget(QLabel("change threshold"), 1, 0)
        pgrid.addWidget(sb_thresh, 1, 1)

        layout.addWidget(pacing)

        # --- per-voice config -----------------------------------------
        for i, voice in enumerate(voices_list):
            box = QGroupBox(f"EEG CH {i}")
            grid = QGridLayout(box)

            cb_on = QCheckBox("on")
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
            grid.addWidget(QLabel("band"), 0, 1)
            grid.addWidget(band_cb, 0, 2)

            scale_cb = QComboBox()
            scale_cb.addItems(sorted(SCALES.keys()))
            scale_cb.setCurrentText(getattr(voice, "scale", "minor_pentatonic"))
            scale_cb.currentTextChanged.connect(
                lambda t, vc=voice: self._set_safe(vc, "scale", t)
            )
            grid.addWidget(QLabel("scale"), 0, 3)
            grid.addWidget(scale_cb, 0, 4)

            root_cb = QComboBox()
            root_cb.addItems(sorted(NOTE_NAME_TO_PC.keys()))
            root_cb.setCurrentText(getattr(voice, "root", "A"))
            root_cb.currentTextChanged.connect(
                lambda t, vc=voice: self._set_safe(vc, "root", t)
            )
            grid.addWidget(QLabel("root"), 1, 1)
            grid.addWidget(root_cb, 1, 2)

            midi_sb = QSpinBox()
            midi_sb.setRange(0, 15)
            midi_sb.setValue(int(getattr(voice, "midi_channel", 0)))
            midi_sb.valueChanged.connect(
                lambda v, vc=voice: self._set_safe(vc, "midi_channel", v)
            )
            grid.addWidget(QLabel("midi ch"), 1, 3)
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
            layout.addWidget(QLabel("per_channel_cc not in this bundle."))
            return scroll

        enable = QCheckBox("ENABLE PER-CHANNEL CCs")
        enable.setChecked(getattr(strategy, "enabled", False))
        enable.toggled.connect(
            lambda v: self._set_safe(strategy, "enabled", v)
        )
        layout.addWidget(enable)

        for i, v in enumerate(getattr(strategy, "voices", []) or []):
            box = QGroupBox(f"EEG CH {i}")
            grid = QGridLayout(box)

            cb_on = QCheckBox("on")
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
            grid.addWidget(QLabel("band"), 0, 1)
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
            midi_sb.setRange(0, 15)
            midi_sb.setValue(int(getattr(v, "midi_channel", 12)))
            midi_sb.valueChanged.connect(
                lambda val, vc=v: self._set_safe(vc, "midi_channel", val)
            )
            grid.addWidget(QLabel("midi ch"), 1, 1)
            grid.addWidget(midi_sb, 1, 2)

            inv_cb = QCheckBox("invert")
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
            grid.addWidget(QLabel("smooth"), 1, 4)
            grid.addWidget(sm, 1, 5)

            layout.addWidget(box)
        layout.addStretch(1)
        return scroll

    def _build_threshold_tab(self) -> QWidget:
        s = getattr(self._bundle, "threshold", None)
        w = QWidget()
        layout = QFormLayout(w)
        if s is None:
            layout.addRow(QLabel("threshold strategy not in this bundle."))
            return w

        enable = QCheckBox()
        enable.setChecked(getattr(s, "enabled", False))
        enable.toggled.connect(lambda v: self._set_safe(s, "enabled", v))
        layout.addRow("ENABLED", enable)

        band_cb = QComboBox()
        band_cb.addItems(_BANDS)
        band_cb.setCurrentText(getattr(s, "band", "beta"))
        band_cb.currentTextChanged.connect(
            lambda t: self._set_safe(s, "band", t)
        )
        layout.addRow("band", band_cb)

        thr = QDoubleSpinBox()
        thr.setRange(0.0, 10_000.0)
        thr.setValue(float(getattr(s, "threshold_uv2", 80.0)))
        thr.setSuffix(" uV^2")
        thr.valueChanged.connect(
            lambda v: self._set_safe(s, "threshold_uv2", v)
        )
        layout.addRow("threshold", thr)

        ch = QSpinBox()
        ch.setRange(0, 15)
        ch.setValue(int(getattr(s, "channel", 9)))
        ch.valueChanged.connect(lambda v: self._set_safe(s, "channel", v))
        layout.addRow("midi channel", ch)
        return w

    def _build_coherence_tab(self) -> QWidget:
        s = getattr(self._bundle, "coherence", None)
        w = QWidget()
        layout = QFormLayout(w)
        if s is None:
            layout.addRow(QLabel("coherence strategy not in this bundle."))
            return w

        enable = QCheckBox()
        enable.setChecked(getattr(s, "enabled", False))
        enable.toggled.connect(lambda v: self._set_safe(s, "enabled", v))
        layout.addRow("ENABLED", enable)

        band_cb = QComboBox()
        band_cb.addItems(_BANDS)
        band_cb.setCurrentText(getattr(s, "band", "alpha"))
        band_cb.currentTextChanged.connect(
            lambda t: self._set_safe(s, "band", t)
        )
        layout.addRow("band", band_cb)

        cc = QSpinBox()
        cc.setRange(0, 127)
        cc.setValue(int(getattr(s, "cc_number", 1)))
        cc.valueChanged.connect(lambda v: self._set_safe(s, "cc_number", v))
        layout.addRow("cc #", cc)

        ch = QSpinBox()
        ch.setRange(0, 15)
        ch.setValue(int(getattr(s, "channel", 10)))
        ch.valueChanged.connect(lambda v: self._set_safe(s, "channel", v))
        layout.addRow("midi channel", ch)

        sm = QDoubleSpinBox()
        sm.setRange(0.0, 0.99)
        sm.setSingleStep(0.05)
        sm.setValue(float(getattr(s, "smoothing", 0.6)))
        sm.valueChanged.connect(lambda v: self._set_safe(s, "smoothing", v))
        layout.addRow("smoothing", sm)
        return w

    def _build_markov_tab(self) -> QWidget:
        s = getattr(self._bundle, "markov", None)
        w = QWidget()
        layout = QFormLayout(w)
        if s is None:
            layout.addRow(QLabel("markov strategy not in this bundle."))
            return w

        enable = QCheckBox()
        enable.setChecked(getattr(s, "enabled", True))
        enable.toggled.connect(lambda v: self._set_safe(s, "enabled", v))
        layout.addRow("ENABLED", enable)

        scale_cb = QComboBox()
        scale_cb.addItems(sorted(SCALES.keys()))
        scale_cb.setCurrentText(getattr(s, "scale", "dorian"))
        scale_cb.currentTextChanged.connect(
            lambda t: self._set_safe(s, "scale", t)
        )
        layout.addRow("scale", scale_cb)

        root_cb = QComboBox()
        root_cb.addItems(sorted(NOTE_NAME_TO_PC.keys()))
        root_cb.setCurrentText(getattr(s, "root", "D"))
        root_cb.currentTextChanged.connect(
            lambda t: self._set_safe(s, "root", t)
        )
        layout.addRow("root", root_cb)

        ch = QSpinBox()
        ch.setRange(0, 15)
        ch.setValue(int(getattr(s, "channel", 8)))
        ch.valueChanged.connect(lambda v: self._set_safe(s, "channel", v))
        layout.addRow("midi channel", ch)

        density = QSpinBox()
        density.setRange(1, 16)
        density.setValue(int(getattr(s, "max_notes_per_window", 1)))
        density.valueChanged.connect(
            lambda v: self._set_safe(s, "max_notes_per_window", v)
        )
        layout.addRow("max notes/win", density)

        mk_interval = QDoubleSpinBox()
        mk_interval.setRange(0.0, 60.0)
        mk_interval.setSingleStep(0.5)
        mk_interval.setDecimals(1)
        mk_interval.setSuffix(" s")
        mk_interval.setValue(float(getattr(s, "min_interval_seconds", 2.5)))
        mk_interval.setToolTip(
            "Min seconds between Markov phrases. Higher = sparser."
        )
        mk_interval.valueChanged.connect(
            lambda v: self._set_safe(s, "min_interval_seconds", float(v))
        )
        layout.addRow("min interval", mk_interval)
        return w

    def _build_clip_tab(self) -> QWidget:
        s = getattr(self._bundle, "clips", None)
        wrap = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(wrap)

        layout = QVBoxLayout(wrap)
        if s is None:
            layout.addWidget(QLabel("clips strategy not in this bundle."))
            return scroll

        enable = QCheckBox("ENABLE CLIP LAUNCHER")
        enable.setChecked(getattr(s, "enabled", False))
        enable.toggled.connect(lambda v: self._set_safe(s, "enabled", v))
        layout.addWidget(enable)

        for i, rule in enumerate(getattr(s, "rules", []) or []):
            box = QGroupBox(f"RULE {i}")
            grid = QGridLayout(box)

            on_cb = QCheckBox("on")
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
            grid.addWidget(QLabel("eeg ch"), 0, 1)
            grid.addWidget(ch_sb, 0, 2)

            band_cb = QComboBox()
            band_cb.addItems(_BANDS)
            band_cb.setCurrentText(getattr(rule, "band", "theta"))
            band_cb.currentTextChanged.connect(
                lambda t, r=rule: self._set_safe(r, "band", t)
            )
            grid.addWidget(QLabel("band"), 0, 3)
            grid.addWidget(band_cb, 0, 4)

            note_sb = QSpinBox()
            note_sb.setRange(0, 127)
            note_sb.setValue(int(getattr(rule, "midi_note", 36 + i)))
            note_sb.valueChanged.connect(
                lambda v, r=rule: self._set_safe(r, "midi_note", v)
            )
            grid.addWidget(QLabel("note"), 1, 1)
            grid.addWidget(note_sb, 1, 2)

            mc = QSpinBox()
            mc.setRange(0, 15)
            mc.setValue(int(getattr(rule, "midi_channel", 11)))
            mc.valueChanged.connect(
                lambda v, r=rule: self._set_safe(r, "midi_channel", v)
            )
            grid.addWidget(QLabel("midi ch"), 1, 3)
            grid.addWidget(mc, 1, 4)

            thr = QDoubleSpinBox()
            thr.setRange(0.0, 10_000.0)
            thr.setValue(float(getattr(rule, "threshold_uv2", 60.0)))
            thr.valueChanged.connect(
                lambda v, r=rule: self._set_safe(r, "threshold_uv2", v)
            )
            grid.addWidget(QLabel("threshold"), 2, 1)
            grid.addWidget(thr, 2, 2)

            layout.addWidget(box)
        layout.addStretch(1)
        return scroll
