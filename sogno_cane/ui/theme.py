"""Clean old-web theme: minimal, calm, readable.

A single radial glow on a deep midnight background. Thin 1-px hot-pink
accent borders. Three colors: hot pink, electric cyan, off-white. The only
ornament is the gradient SOGNO_CANE title.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QFontMetricsF,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QLabel,
    QSizePolicy,
    QWidget,
)

# ---------------------------------------------------------------------- #
# Palette                                                                #
# ---------------------------------------------------------------------- #
BG_DEEP      = "#0A0418"
BG_PANEL_HEX = "#140828"
PANEL_FILL   = "#140828"
ACCENT_PINK  = "#FF3D8A"
ACCENT_CYAN  = "#5BE9FF"
TEXT         = "#F1E9F4"
TEXT_DIM     = "#7F7298"
BORDER_DIM   = "#3B2554"

# Back-compat aliases.
HOT_PINK     = ACCENT_PINK
ELECTRIC     = ACCENT_CYAN
NEON_LIME    = "#7DFF5F"
NEON_YELLOW  = "#FFE372"
NEON_PURPLE  = "#A06AD9"
NEON_ORANGE  = "#FF9D45"
FG_LIME      = NEON_LIME
FG_CYAN      = ACCENT_CYAN
FG_MAGENTA   = ACCENT_PINK
FG_YELLOW    = NEON_YELLOW
FG_WHITE     = TEXT
FG_DIM       = TEXT_DIM
BG_BLACK     = BG_DEEP
BG_PANEL     = PANEL_FILL
BG_PANEL_2   = "#1A0E33"


QSS = f"""
* {{
    color: {TEXT};
    font-family: "Inter", "Trebuchet MS", "Segoe UI",
                 "DejaVu Sans", sans-serif;
    font-size: 10pt;
}}

QMainWindow, #BackgroundWidget {{ background: transparent; }}
QWidget {{ background: transparent; }}

QLabel {{ color: {TEXT}; background: transparent; }}
QLabel[role="value"] {{ color: {ACCENT_CYAN}; }}
QLabel[role="bad"]   {{ color: {ACCENT_PINK}; }}
QLabel[role="good"]  {{ color: {NEON_LIME}; }}
QLabel[role="warn"]  {{ color: {NEON_YELLOW}; }}

QLabel#SubHeader {{
    color: {ACCENT_PINK};
    font-size: 15pt;
    font-weight: 700;
    letter-spacing: 3px;
    padding: 0;
}}

QGroupBox {{
    border: 1px solid {BORDER_DIM};
    border-radius: 2px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    background-color: rgba(20, 8, 40, 200);
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: {ACCENT_PINK};
    background-color: {BG_DEEP};
    font-size: 9pt;
    font-weight: 700;
    letter-spacing: 1px;
}}

QPushButton {{
    background-color: transparent;
    color: {ACCENT_PINK};
    border: 1px solid {ACCENT_PINK};
    border-radius: 2px;
    padding: 5px 11px;
    font-weight: 700;
    letter-spacing: 1px;
    min-height: 18px;
}}
QPushButton:hover {{
    color: {BG_DEEP};
    background-color: {ACCENT_PINK};
}}
QPushButton:pressed {{
    background-color: #C92B6E;
    color: {BG_DEEP};
}}
QPushButton:disabled {{
    color: {BORDER_DIM};
    border-color: {BORDER_DIM};
    background-color: transparent;
}}
QPushButton#PrimaryButton {{
    color: {BG_DEEP};
    background-color: {ACCENT_PINK};
    border: 1px solid {ACCENT_PINK};
    font-size: 11pt;
    letter-spacing: 2px;
    min-height: 24px;
}}
QPushButton#PrimaryButton:hover {{
    background-color: #FF5FA0;
    color: {BG_DEEP};
}}
QPushButton#DangerButton {{
    color: {BG_DEEP};
    background-color: {ACCENT_CYAN};
    border: 1px solid {ACCENT_CYAN};
    font-size: 11pt;
    letter-spacing: 2px;
    min-height: 24px;
}}
QPushButton#DangerButton:hover {{ background-color: #82F1FF; }}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background-color: {BG_PANEL_HEX};
    color: {ACCENT_CYAN};
    border: 1px solid {BORDER_DIM};
    border-radius: 2px;
    padding: 3px 6px;
    selection-background-color: {ACCENT_PINK};
    selection-color: {BG_DEEP};
    min-height: 18px;
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border-color: {ACCENT_PINK};
}}
QComboBox QAbstractItemView {{
    background-color: {BG_PANEL_HEX};
    color: {ACCENT_CYAN};
    selection-background-color: {ACCENT_PINK};
    selection-color: {BG_DEEP};
    border: 1px solid {BORDER_DIM};
}}
QComboBox::drop-down {{ border: none; width: 18px; background: transparent; }}
QComboBox::down-arrow {{
    image: none;
    border-top: 5px solid {ACCENT_PINK};
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    width: 0; height: 0;
}}

QCheckBox {{ spacing: 10px; color: {TEXT}; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {ACCENT_PINK};
    border-radius: 2px;
    background: transparent;
}}
QCheckBox::indicator:checked {{ background: {ACCENT_PINK}; }}

QStatusBar {{
    background-color: transparent;
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER_DIM};
    letter-spacing: 1px;
}}

QSlider::groove:horizontal {{
    background: {BG_PANEL_HEX};
    height: 4px;
    border: none;
}}
QSlider::handle:horizontal {{
    background: {ACCENT_PINK};
    width: 12px;
    margin: -6px 0;
    border-radius: 2px;
}}

QScrollArea, QListWidget, QTextEdit {{
    background-color: rgba(20, 8, 40, 230);
    border: 1px solid {BORDER_DIM};
}}
QListWidget {{
    color: {ACCENT_CYAN};
    font-family: "JetBrains Mono", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 9pt;
    letter-spacing: 0;
}}

QTabWidget::pane {{
    border: 1px solid {BORDER_DIM};
    background: rgba(20, 8, 40, 220);
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_DIM};
    padding: 6px 14px;
    border: 1px solid {BORDER_DIM};
    border-bottom: none;
    font-weight: 700;
    letter-spacing: 1px;
}}
QTabBar::tab:selected {{
    color: {ACCENT_PINK};
    border-color: {ACCENT_PINK};
    background: rgba(255, 61, 138, 30);
}}
QTabBar::tab:hover:!selected {{ color: {ACCENT_CYAN}; }}

QToolTip {{
    background-color: {BG_PANEL_HEX};
    color: {ACCENT_CYAN};
    border: 1px solid {ACCENT_PINK};
    padding: 4px;
}}

QScrollBar:vertical {{
    background: {BG_PANEL_HEX};
    width: 10px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {ACCENT_PINK};
    min-height: 24px;
    border-radius: 2px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    background: transparent; height: 0;
}}
"""


# ---------------------------------------------------------------------- #
# Background: solid color + a single soft glow at the top center        #
# ---------------------------------------------------------------------- #
class BackgroundWidget(QWidget):
    """Static dark background with one soft radial glow under the title."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("BackgroundWidget")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

    def paintEvent(self, ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(BG_DEEP))
        # Soft pink/cyan glow at top center.
        grad = QRadialGradient(QPointF(w / 2, h * 0.08), max(w, h) * 0.55)
        grad.setColorAt(0.00, QColor(255, 61, 138, 90))
        grad.setColorAt(0.45, QColor(255, 61, 138, 30))
        grad.setColorAt(0.75, QColor(91, 233, 255, 18))
        grad.setColorAt(1.00, QColor(0, 0, 0, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawRect(self.rect())

        # Thin pink rule under the title (~ y=180).
        rule_y = min(int(h * 0.18), 220)
        p.setPen(QPen(QColor(255, 61, 138, 180), 1))
        margin_x = int(w * 0.06)
        p.drawLine(margin_x, rule_y, w - margin_x, rule_y)


# Backwards-compat alias for any code still importing StarCanvas.
StarCanvas = BackgroundWidget


# ---------------------------------------------------------------------- #
# Title: chrome gradient with one drop shadow                            #
# ---------------------------------------------------------------------- #
class ChromeTitle(QLabel):
    """Large pink+cyan gradient title, single soft drop shadow."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
        )
        self._family = _first_available_family([
            "Inter", "Trebuchet MS", "Segoe UI",
            "DejaVu Sans", "sans-serif",
        ]) or "sans-serif"
        self.setFixedHeight(64)

        glow = QGraphicsDropShadowEffect(self)
        glow.setBlurRadius(24)
        glow.setColor(QColor(255, 61, 138, 180))
        glow.setOffset(0, 0)
        self.setGraphicsEffect(glow)

    def paintEvent(self, ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        rect = self.rect()
        text = self.text()
        font = QFont(self._family)
        font.setBold(True)
        font.setPixelSize(max(34, int(rect.height() * 0.62)))
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 6.0)
        p.setFont(font)
        fm = QFontMetricsF(font)
        text_w = fm.horizontalAdvance(text)
        text_h = fm.height()
        tx = (rect.width() - text_w) / 2.0
        ty = (rect.height() + text_h) / 2.0 - fm.descent()

        path = QPainterPath()
        path.addText(QPointF(tx, ty), font, text)

        # Single subtle cyan shadow below.
        p.translate(0, 3)
        p.fillPath(path, QColor(91, 233, 255, 110))
        p.translate(0, -3)

        # Pink->white->cyan vertical gradient fill.
        grad = QLinearGradient(0, ty - text_h, 0, ty)
        grad.setColorAt(0.00, QColor("#FFE6F0"))
        grad.setColorAt(0.45, QColor("#FF3D8A"))
        grad.setColorAt(1.00, QColor("#5BE9FF"))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(path)


# ---------------------------------------------------------------------- #
# Thin divider                                                            #
# ---------------------------------------------------------------------- #
class SparkleDivider(QFrame):
    """A thin horizontal accent line. Replaces the previous starry row."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFixedHeight(2)
        self.setStyleSheet(
            f"QFrame {{ background-color: {BORDER_DIM}; border: none; }}"
        )


class CRTOverlay(QWidget):
    """No-op overlay kept for back-compat; renders nothing."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

    def paintEvent(self, ev) -> None:  # noqa: N802
        return


class ChromeFrame(QFrame):
    """Simple thin-bordered container."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame {{ "
            f"  background: rgba(20, 8, 40, 200); "
            f"  border: 1px solid {BORDER_DIM}; "
            f"}}"
        )


# ---------------------------------------------------------------------- #
# Helpers / apply                                                         #
# ---------------------------------------------------------------------- #
def _first_available_family(candidates: list[str]) -> str | None:
    families = QFontDatabase.families()
    for c in candidates:
        if c in families:
            return c
    return None


def icon_path() -> str:
    """Absolute path to the app icon shipped in the package assets."""
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "assets", "icon.ico"
    )


def app_icon() -> QIcon:
    p = icon_path()
    return QIcon(p) if os.path.exists(p) else QIcon()


def apply_theme(app: QApplication) -> None:
    app.setStyleSheet(QSS)
    try:
        ic = app_icon()
        if not ic.isNull():
            app.setWindowIcon(ic)
    except Exception:
        pass

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(BG_DEEP))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Base, QColor(PANEL_FILL))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#1A0E33"))
    pal.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Button, QColor(BG_PANEL_HEX))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(ACCENT_PINK))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT_PINK))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(BG_DEEP))
    app.setPalette(pal)

    family = _first_available_family([
        "Inter", "Trebuchet MS", "Segoe UI",
        "DejaVu Sans", "sans-serif",
    ])
    if family:
        app.setFont(QFont(family, 11))
