"""Render and exercise the launcher's styled dropdown and stepper controls."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QSpinBox,
    QStyle, QStyleOptionComboBox, QStyleOptionSpinBox, QWidget,
)

from colibri.launcher.theme import apply_theme, stylesheet


APP = QApplication.instance() or QApplication([])


def control_rect(widget, part):
    if isinstance(widget, QComboBox):
        option, kind = QStyleOptionComboBox(), QStyle.ComplexControl.CC_ComboBox
    else:
        option, kind = QStyleOptionSpinBox(), QStyle.ComplexControl.CC_SpinBox
    widget.initStyleOption(option)
    return widget.style().subControlRect(kind, option, part, widget)


class StyledControlTests(unittest.TestCase):
    theme = "light"

    def setUp(self):
        palette = APP.palette()
        self.addCleanup(APP.setPalette, palette)
        apply_theme(APP, self.theme)
        self.panel = QWidget()
        self.panel.setStyleSheet(stylesheet(self.theme))
        layout = QFormLayout(self.panel)
        self.combo = QComboBox()
        self.combo.addItems(["Web Chat", "API Server"])
        self.spin = QSpinBox()
        self.double = QDoubleSpinBox()
        for spin in (self.spin, self.double):
            spin.setRange(0, 100)
            spin.setValue(16)
        for name, control in (("App", self.combo), ("RAM", self.spin), ("GPU memory", self.double)):
            layout.addRow(name, control)
        self.panel.resize(500, 200)
        self.panel.show()
        APP.processEvents()
        self.addCleanup(self.panel.close)

    def test_surface_and_popup_follow_the_selected_theme(self):
        dark = self.theme == "dark"
        for widget in (self.panel, self.combo.view()):
            palette = widget.palette()
            background = palette.color(widget.backgroundRole())
            self.assertEqual(background.lightnessF() < 0.5, dark)
            foreground = palette.color(widget.foregroundRole())
            self.assertGreaterEqual(contrast(foreground, background), 4.5)
        for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive,
                      QPalette.ColorGroup.Disabled):
            palette = APP.palette()
            for foreground, background in ((QPalette.ColorRole.Text, QPalette.ColorRole.Base),
                                          (QPalette.ColorRole.WindowText, QPalette.ColorRole.Window),
                                          (QPalette.ColorRole.HighlightedText, QPalette.ColorRole.Highlight)):
                self.assertGreaterEqual(contrast(palette.color(group, foreground),
                                                palette.color(group, background)), 4.5)

    def assert_visible_arrow(self, widget, part):
        rect = control_rect(widget, part)
        # Ignore the surrounding frame: borders alone must not pass as arrows.
        center = rect.adjusted(4, 3, -4, -3)
        image = widget.grab().toImage()
        scale = image.devicePixelRatio()
        background = image.pixelColor(round(center.left() * scale), round(center.top() * scale))
        ink = sum(
            contrast(image.pixelColor(x, y), background) >= 3
            for y in range(round(center.top() * scale), round((center.bottom() + 1) * scale))
            for x in range(round(center.left() * scale), round((center.right() + 1) * scale))
        )
        self.assertGreaterEqual(ink, 4 * scale * scale, "The arrow is missing from its button")
        self.assertGreaterEqual(rect.width(), 24, "The arrow button is too narrow to click comfortably")
        self.assertTrue(widget.rect().contains(rect), "The arrow button is clipped by the field")

    def test_dropdown_arrow_is_visible_and_opens_its_popup(self):
        arrow = QStyle.SubControl.SC_ComboBoxArrow
        self.assert_visible_arrow(self.combo, arrow)
        edit = control_rect(self.combo, QStyle.SubControl.SC_ComboBoxEditField)
        self.assertFalse(edit.intersects(control_rect(self.combo, arrow)))
        QTest.mouseClick(self.combo, Qt.MouseButton.LeftButton, pos=control_rect(self.combo, arrow).center())
        self.assertTrue(self.combo.view().isVisible())
        QTest.keyClick(self.combo, Qt.Key.Key_Down)
        QTest.keyClick(self.combo, Qt.Key.Key_Return)
        self.assertEqual(self.combo.currentText(), "API Server")
        self.combo.setEnabled(False)
        self.assert_visible_arrow(self.combo, arrow)

    def test_stepper_arrows_render_and_click_without_overlapping_values(self):
        up, down = QStyle.SubControl.SC_SpinBoxUp, QStyle.SubControl.SC_SpinBoxDown
        for widget in (self.spin, self.double):
            with self.subTest(control=type(widget).__name__):
                self.assert_visible_arrow(widget, up)
                self.assert_visible_arrow(widget, down)
                edit = control_rect(widget, QStyle.SubControl.SC_SpinBoxEditField)
                self.assertFalse(edit.intersects(control_rect(widget, up)))
                self.assertFalse(edit.intersects(control_rect(widget, down)))
                QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=control_rect(widget, up).center())
                self.assertEqual(widget.value(), 17)
                QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=control_rect(widget, down).center())
                self.assertEqual(widget.value(), 16)
                widget.setValue(widget.minimum())
                self.assert_visible_arrow(widget, down)
                QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=control_rect(widget, down).center())
                self.assertEqual(widget.value(), 0)
                widget.setValue(widget.maximum())
                self.assert_visible_arrow(widget, up)
                QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=control_rect(widget, up).center())
                self.assertEqual(widget.value(), 100)
                widget.setEnabled(False)
                self.assert_visible_arrow(widget, up)
                self.assert_visible_arrow(widget, down)


class DarkStyledControlTests(StyledControlTests):
    theme = "dark"


def contrast(first, second):
    def luminance(color):
        values = (channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
                  for channel in (color.redF(), color.greenF(), color.blueF()))
        return sum(channel * weight for channel, weight in zip(values, (0.2126, 0.7152, 0.0722)))

    light, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


if __name__ == "__main__":
    unittest.main()
