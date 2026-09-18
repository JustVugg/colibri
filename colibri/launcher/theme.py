"""Consistent launcher colors, including Qt popups and child dialogs."""

from pathlib import Path
from string import Template

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


# Each theme supplies the same semantic colors to Qt's palette and stylesheet.
# Keeping both in sync prevents system colors from leaking into custom surfaces.
_COLORS = {
    "light": {
        "window": "#F7F4EC",
        "text": "#1D2C25",
        "sidebar": "#E8EEE6",
        "divider": "#CCD7CE",
        "brand": "#18553B",
        "heading": "#173C2D",
        "muted": "#65746C",
        "alternate": "#EEF3EC",
        "empty_border": "#A9B9AD",
        "empty_text": "#4B6256",
        "surface": "#FFFEFA",
        "card_border": "#D9DED7",
        "title": "#244C38",
        "selection": "#CFE2D5",
        "selected_text": "#153D2B",
        "border": "#BBC7BE",
        "control_hover": "#DFEADF",
        "disabled_surface": "#ECEFEA",
        "hover_border": "#3A7557",
        "button_hover": "#F3F7F2",
        "disabled_text": "#607066",
        "primary": "#1D6646",
        "primary_hover": "#17583C",
        "primary_disabled": "#A7B7AC",
        "primary_disabled_text": "#294537",
        "tool_text": "#315B46",
        "log_surface": "#14221B",
        "log_text": "#D9E9DE",
        "primary_text": "#FFFFFF",
        "light_edge": "#FFFFFF",
    },
    "dark": {
        "window": "#151C19",
        "text": "#E5EEE8",
        "sidebar": "#1A241F",
        "divider": "#3B4E42",
        "brand": "#9BDFB3",
        "heading": "#E5F5E9",
        "muted": "#ADBCB2",
        "alternate": "#29372F",
        "empty_border": "#6C8876",
        "empty_text": "#BACDC1",
        "surface": "#202A24",
        "card_border": "#43584A",
        "title": "#B9DFC5",
        "selection": "#355D47",
        "selected_text": "#F0FFF4",
        "border": "#53665A",
        "control_hover": "#3B5143",
        "disabled_surface": "#26322B",
        "hover_border": "#8BCBA0",
        "button_hover": "#2B3D32",
        "disabled_text": "#A6B7AC",
        "primary": "#96DBAB",
        "primary_hover": "#AFEDC0",
        "primary_disabled": "#3C5546",
        "primary_disabled_text": "#C2D1C6",
        "tool_text": "#C9E8D3",
        "log_surface": "#101713",
        "log_text": "#D9E9DE",
        "primary_text": "#12241A",
        "light_edge": "#43584A",
    },
}


def apply_theme(app: QApplication, theme: str = "light") -> None:
    """Apply a complete palette so OS light/dark settings cannot obscure text."""
    colors = _COLORS[theme]
    app.styleHints().setColorScheme(Qt.ColorScheme.Dark if theme == "dark" else Qt.ColorScheme.Light)
    if app.style().objectName().lower() != "fusion":
        app.setStyle("Fusion")
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: "window",
        QPalette.ColorRole.WindowText: "text",
        QPalette.ColorRole.Base: "surface",
        QPalette.ColorRole.AlternateBase: "alternate",
        QPalette.ColorRole.Text: "text",
        QPalette.ColorRole.Button: "surface",
        QPalette.ColorRole.ButtonText: "text",
        QPalette.ColorRole.PlaceholderText: "muted",
        QPalette.ColorRole.Highlight: "selection",
        QPalette.ColorRole.HighlightedText: "selected_text",
        QPalette.ColorRole.ToolTipBase: "surface",
        QPalette.ColorRole.ToolTipText: "text",
        QPalette.ColorRole.Light: "light_edge",
        QPalette.ColorRole.Midlight: "sidebar",
        QPalette.ColorRole.Mid: "border",
        QPalette.ColorRole.Dark: "muted",
        QPalette.ColorRole.Shadow: "text",
        QPalette.ColorRole.BrightText: "primary_text",
        QPalette.ColorRole.Link: "primary",
        QPalette.ColorRole.LinkVisited: "title",
        QPalette.ColorRole.Accent: "primary",
    }
    for role, token in roles.items():
        palette.setColor(role, QColor(colors[token]))  # Active, inactive, disabled.
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(colors["disabled_text"]))
    app.setPalette(palette)


def stylesheet(theme: str = "light") -> str:
    """Return the matching widget styles and packaged arrow assets."""
    icons = Path(__file__).with_name("icons").as_posix().replace('"', r'\"')
    return Template(_STYLE).substitute(
        _COLORS[theme], icons=icons, icon_suffix="-dark" if theme == "dark" else "",
    )


_STYLE = """
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget { background: ${window}; color: ${text}; }
#sidebar { background: ${sidebar}; border-right: 1px solid ${divider}; }
#brand { color: ${brand}; font-size: 22px; font-weight: 800; letter-spacing: 3px; }
#heading { color: ${heading}; font-size: 27px; font-weight: 700; }
#muted, #pathLabel, #helpText { color: ${muted}; }
#pathLabel { font-size: 12px; }
#emptyState { background: ${alternate}; border: 1px dashed ${empty_border}; border-radius: 10px; color: ${empty_text}; padding: 20px; }
#card { background: ${surface}; border: 1px solid ${card_border}; border-radius: 10px; }
#cardTitle, #statusLabel { color: ${title}; font-size: 16px; font-weight: 700; }
QListWidget { background: transparent; border: none; outline: none; }
QListWidget::item { padding: 10px 9px; margin: 2px 0; border-radius: 7px; }
QListWidget::item:selected { background: ${selection}; color: ${selected_text}; }
QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { min-height: 30px; border: 1px solid ${border}; border-radius: 6px; background: ${surface}; padding: 2px 9px; }
QComboBox QAbstractItemView { background: ${surface}; color: ${text}; selection-background-color: ${selection}; selection-color: ${selected_text}; border: 1px solid ${border}; outline: none; }
QComboBox, QSpinBox, QDoubleSpinBox { padding-right: 36px; }
QSpinBox QLineEdit, QDoubleSpinBox QLineEdit { min-height: 0; border: none; padding: 0; background: transparent; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 28px; border: none; border-left: 1px solid ${border}; border-top-right-radius: 5px; border-bottom-right-radius: 5px; background: ${alternate}; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: padding; subcontrol-position: top right; width: 28px; border: none; border-left: 1px solid ${border}; border-top-right-radius: 5px; background: ${alternate}; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: padding; subcontrol-position: bottom right; width: 28px; border: none; border-left: 1px solid ${border}; border-top: 1px solid ${border}; border-bottom-right-radius: 5px; background: ${alternate}; }
QComboBox::drop-down:hover, QSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background: ${control_hover}; }
QComboBox::drop-down:pressed, QSpinBox::up-button:pressed, QSpinBox::down-button:pressed, QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed { background: ${selection}; }
QComboBox::drop-down:disabled, QSpinBox::up-button:disabled, QSpinBox::down-button:disabled, QDoubleSpinBox::up-button:disabled, QDoubleSpinBox::down-button:disabled, QSpinBox::up-button:off, QSpinBox::down-button:off, QDoubleSpinBox::up-button:off, QDoubleSpinBox::down-button:off { background: ${disabled_surface}; }
QComboBox::down-arrow, QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url("${icons}/chevron-down${icon_suffix}.svg"); width: 12px; height: 12px; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url("${icons}/chevron-up${icon_suffix}.svg"); width: 12px; height: 12px; }
QComboBox::down-arrow:disabled, QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled, QSpinBox::down-arrow:off, QDoubleSpinBox::down-arrow:off { image: url("${icons}/chevron-down-disabled${icon_suffix}.svg"); }
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled, QSpinBox::up-arrow:off, QDoubleSpinBox::up-arrow:off { image: url("${icons}/chevron-up-disabled${icon_suffix}.svg"); }
QPushButton:hover { border-color: ${hover_border}; background: ${button_hover}; }
QPushButton:disabled { color: ${disabled_text}; background: ${disabled_surface}; }
#primaryButton { min-width: 112px; min-height: 38px; background: ${primary}; color: ${primary_text}; border: none; font-weight: 700; }
#primaryButton:hover { background: ${primary_hover}; }
#primaryButton:disabled { background: ${primary_disabled}; color: ${primary_disabled_text}; }
#stopButton { min-width: 90px; min-height: 38px; }
QToolButton { color: ${tool_text}; font-weight: 600; border: none; padding: 4px; }
QPlainTextEdit { background: ${log_surface}; color: ${log_text}; border-radius: 6px; padding: 7px; font-family: Consolas, monospace; }
"""
