from PyQt5.QtWidgets import QLabel, QPushButton, QLineEdit, QWidget
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


class MyLabel(QLabel):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        self.setText(kwargs.get("text", ""))
        self.setFont(QFont(kwargs.get("font", "Arial"), kwargs.get("size", 12)))
        self.setAlignment(kwargs.get("alignment", Qt.AlignLeft))
        # Additional styles can be added here


class MyButton(QPushButton):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        self.setText(kwargs.get("text", ""))
        self.setFont(QFont(kwargs.get("font", "Arial"), kwargs.get("size", 12)))
        # Additional styles can be added here


class MyEntry(QLineEdit):
    def __init__(self, parent=None, placeholder="", **kwargs):
        super().__init__(parent)
        self.placeholder = placeholder
        self.setPlaceholderText(self.placeholder)
        self.setFont(QFont(kwargs.get("font", "Arial"), kwargs.get("size", 12)))
        # Placeholder behavior is handled by setPlaceholderText in QLineEdit

        # Connect the resizing event
        parent.resizeEvent = self._resize_event

    def _resize_event(self, event):
        # Adjust the width of the entry widget based on the parent's width
        new_width = event.size().width()
        self.setFixedWidth(new_width - 100)
        super().resizeEvent(event)
