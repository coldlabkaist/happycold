from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox


class NoWheelComboBox(QComboBox):
    """Combo box that leaves mouse-wheel scrolling to its parent."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """Integer spin box that leaves mouse-wheel scrolling to its parent."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Floating-point spin box that leaves mouse-wheel scrolling to its parent."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()
