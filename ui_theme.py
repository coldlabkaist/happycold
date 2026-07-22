from __future__ import annotations


THEME_COLORS: dict[str, str] = {
    "text": "#2f2f2f",
    "text_strong": "#1f1f1f",
    "text_muted": "#747b86",
    "app_bg": "#f2f3f5",
    "surface": "#ffffff",
    "surface_soft": "#f6f7f9",
    "surface_pressed": "#eceff3",
    "surface_tab": "#e9ebef",
    "border": "#b8bcc3",
    "border_soft": "#d4d7dc",
    "accent": "#4677df",
    "accent_hover": "#315fbd",
    "accent_soft": "#eaf0ff",
}


def build_app_stylesheet() -> str:
    c = THEME_COLORS
    return f"""
    QWidget {{
        color: {c['text']};
        font-size: 12px;
    }}
    QMainWindow, QDialog, QMessageBox {{
        background: {c['app_bg']};
    }}
    QGroupBox {{
        background: {c['surface']};
        border: 1px solid {c['border_soft']};
        border-radius: 10px;
        margin-top: 10px;
        padding: 12px 8px 8px 8px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: {c['text_strong']};
    }}
    QFrame#fileSidebar, QFrame#viewerHeader, QFrame#viewerNodeControls, QFrame#frameNavigation,
    QFrame#batchDialogHeader, QFrame#batchDialogFooter {{
        background: {c['surface']};
        border: 1px solid {c['border_soft']};
        border-radius: 10px;
    }}
    QFrame#pipelineStageCard {{
        background: {c['surface_soft']};
        border: 1px solid {c['border_soft']};
        border-radius: 8px;
    }}
    QFrame#batchSelectionHeader {{
        background: {c['surface']};
        border: 1px solid {c['border_soft']};
        border-radius: 10px;
    }}
    QLabel#batchSelectionEyebrow {{
        color: {c['accent']};
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.8px;
    }}
    QLabel#batchSelectionTitle {{
        color: {c['text_strong']};
        font-size: 17px;
        font-weight: 700;
    }}
    QFrame#batchSelectionNotice {{
        background: #fff8e8;
        border: 1px solid #e6c987;
        border-radius: 9px;
    }}
    QLabel#batchSelectionNoticeTitle {{
        color: #805b16;
        font-weight: 700;
    }}
    QLabel#batchSelectionWarning {{
        color: #6f5728;
    }}
    QLabel#batchSelectionStatus {{
        background: {c['surface_soft']};
        color: {c['text_muted']};
        border: 1px solid {c['border_soft']};
        border-radius: 7px;
        padding: 7px 9px;
        font-weight: 600;
    }}
    QFrame#batchSelectionFooter {{
        background: {c['surface']};
        border: 1px solid {c['border_soft']};
        border-radius: 9px;
    }}
    QListWidget#batchVideoList::item {{
        padding: 0;
        border-bottom: 1px solid {c['border_soft']};
    }}
    QLabel#batchVideoName {{
        color: {c['text_strong']};
        background: transparent;
    }}
    QLabel[csvState="ready"] {{
        color: #15803d;
        background: transparent;
        font-size: 11px;
        font-weight: 700;
    }}
    QLabel[csvState="missing"] {{
        color: {c['text_muted']};
        background: transparent;
        font-size: 11px;
        font-weight: 600;
    }}
    QFrame#outputWorkspace {{
        background: {c['surface']};
    }}
    QLabel#outputWorkspaceTitle {{
        color: {c['text_muted']};
        font-size: 11px;
        font-weight: 700;
        padding: 0 2px;
    }}
    QLabel[sectionTitle="true"] {{
        color: {c['text_strong']};
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}
    QLabel[muted="true"] {{
        color: {c['text_muted']};
    }}
    QLabel#annotateHelpTitle {{
        color: {c['text_strong']};
        font-size: 18px;
        font-weight: 700;
        padding: 2px 2px 4px 2px;
    }}
    QLabel[shortcutKey="true"] {{
        background: {c['accent_soft']};
        color: {c['accent_hover']};
        border: 1px solid {c['border_soft']};
        border-radius: 6px;
        padding: 5px 8px;
        font-weight: 700;
    }}
    QLabel[shortcutDescription="true"] {{
        color: {c['text_strong']};
    }}
    QLabel#folderPathLabel {{
        background: {c['surface_soft']};
        color: {c['text_muted']};
        border: 1px solid {c['border_soft']};
        border-radius: 7px;
        padding: 7px;
    }}
    QWidget#frameViewer {{
        background: {c['surface']};
        border: 1px solid {c['border_soft']};
        border-radius: 10px;
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget,
    QTreeWidget, QPlainTextEdit, QTextEdit {{
        background: {c['surface']};
        color: {c['text_strong']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        padding: 1px 6px;
        selection-background-color: {c['accent']};
        selection-color: white;
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        min-height: 20px;
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QListWidget:focus, QTreeWidget:focus {{
        border-color: {c['accent']};
    }}
    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,
    QDoubleSpinBox:disabled {{
        background: {c['surface_soft']};
        color: {c['text_muted']};
        border-color: {c['border_soft']};
    }}
    QLabel:disabled, QGroupBox:disabled, QCheckBox:disabled,
    QRadioButton:disabled {{
        color: {c['text_muted']};
    }}
    QComboBox {{
        padding-right: 24px;
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 22px;
        border-left: 1px solid {c['border_soft']};
        background: {c['surface_soft']};
        border-top-right-radius: 8px;
        border-bottom-right-radius: 8px;
    }}
    QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
        width: 18px;
        border-left: 1px solid {c['border_soft']};
        background: {c['surface_soft']};
    }}
    QCheckBox, QRadioButton {{
        spacing: 6px;
    }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 14px;
        height: 14px;
    }}
    QSlider::groove:horizontal {{
        height: 6px;
        border-radius: 3px;
        background: {c['border']};
    }}
    QSlider::sub-page:horizontal {{
        background: {c['accent']};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        width: 14px;
        margin: -5px 0;
        border: 1px solid {c['accent']};
        border-radius: 7px;
        background: {c['surface']};
    }}
    QListWidget {{
        outline: none;
        padding: 4px;
    }}
    QListWidget::item {{
        border-radius: 6px;
        margin: 1px 0;
        padding: 4px 7px;
    }}
    QListWidget::item:hover {{
        background: {c['accent_soft']};
        color: {c['text_strong']};
    }}
    QListWidget::item:selected, QListWidget::item:selected:!active {{
        background: {c['accent']};
        color: white;
    }}
    QTabWidget::pane {{
        background: {c['surface']};
        border: 1px solid {c['border_soft']};
        border-radius: 10px;
        top: -1px;
    }}
    QTabBar::tab {{
        background: {c['surface_tab']};
        color: {c['text']};
        border: 1px solid {c['border']};
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        padding: 5px 10px;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background: {c['surface']};
        color: {c['text_strong']};
        font-weight: 600;
    }}
    QTabBar::tab:hover:!selected {{
        background: {c['accent_soft']};
    }}
    QLabel#workflowHeading {{
        color: {c['text_muted']};
        font-size: 11px;
        font-weight: 700;
        padding: 0 2px;
    }}
    QTabBar#workflowCategoryBar {{
        background: {c['surface_soft']};
        border: 1px solid {c['border_soft']};
        border-radius: 9px;
        padding: 2px;
    }}
    QTabBar#workflowCategoryBar::tab {{
        background: transparent;
        color: {c['text']};
        border: none;
        border-radius: 7px;
        padding: 7px 12px;
        margin: 1px;
        font-weight: 600;
    }}
    QTabBar#workflowCategoryBar::tab:selected {{
        background: {c['accent']};
        color: white;
    }}
    QTabBar#workflowCategoryBar::tab:hover:!selected {{
        background: {c['accent_soft']};
        color: {c['text_strong']};
    }}
    QLabel#workflowToolsLabel {{
        color: {c['text_muted']};
        font-size: 11px;
        font-weight: 700;
        padding: 2px 2px 0 2px;
    }}
    QTabWidget#modeTabs::pane {{
        background: {c['surface']};
        border: 1px solid {c['border_soft']};
        border-radius: 8px;
        top: 0;
    }}
    QTabBar#modeToolTabBar::tab {{
        background: transparent;
        color: {c['text_muted']};
        border: none;
        border-bottom: 2px solid transparent;
        border-radius: 0;
        padding: 6px 8px;
        margin: 0 4px 0 0;
        font-weight: 500;
    }}
    QTabBar#modeToolTabBar::tab:selected {{
        background: transparent;
        color: {c['accent']};
        border-bottom: 2px solid {c['accent']};
        font-weight: 700;
    }}
    QTabBar#modeToolTabBar::tab:hover:!selected {{
        background: {c['accent_soft']};
        color: {c['text_strong']};
    }}
    QPushButton {{
        background: {c['surface']};
        color: {c['text_strong']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        padding: 1px 8px;
        min-height: 20px;
        font-weight: 500;
    }}
    QPushButton#annotateHelpButton {{
        background: {c['surface']};
        color: {c['text_muted']};
        border: 1px solid {c['border']};
        border-radius: 7px;
        padding: 0;
        min-width: 22px;
        min-height: 20px;
        max-width: 24px;
        max-height: 22px;
        font-size: 14px;
        font-weight: 700;
    }}
    QPushButton#annotateHelpButton:hover {{
        background: {c['accent_soft']};
        color: {c['accent_hover']};
        border-color: {c['accent']};
    }}
    QPushButton#annotateHelpButton:pressed {{
        background: {c['surface_pressed']};
        color: {c['accent_hover']};
        border-color: {c['accent']};
    }}
    QPushButton:hover {{
        background: {c['accent_soft']};
        border-color: {c['accent']};
    }}
    QPushButton:pressed {{
        background: {c['surface_pressed']};
    }}
    QPushButton:disabled {{
        background: {c['surface_soft']};
        color: {c['text_muted']};
        border-color: {c['border_soft']};
    }}
    QPushButton[primary="true"] {{
        background: {c['accent']};
        color: white;
        border-color: {c['accent']};
        font-weight: 600;
    }}
    QPushButton[primary="true"]:hover {{
        background: {c['accent_hover']};
        border-color: {c['accent_hover']};
    }}
    QScrollArea {{
        background: transparent;
        border: none;
    }}
    QScrollArea > QWidget > QWidget {{
        background: transparent;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {c['border']};
        border-radius: 4px;
        min-height: 28px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QSplitter::handle {{
        background: transparent;
    }}
    QSplitter::handle:horizontal {{
        width: 8px;
    }}
    QSplitter#controlsSplitter::handle:vertical {{
        height: 8px;
        background: transparent;
        border-top: 1px solid {c['border_soft']};
        margin: 3px 0 0 0;
    }}
    QStatusBar {{
        background: {c['surface']};
        color: {c['text_muted']};
        border-top: 1px solid {c['border_soft']};
    }}
    QToolTip {{
        background: {c['text_strong']};
        color: white;
        border: none;
        padding: 5px;
    }}
    QMenuBar {{
        background: {c['surface']};
        color: {c['text']};
        border-bottom: 1px solid {c['border_soft']};
    }}
    QMenuBar::item {{
        background: transparent;
        border-radius: 5px;
        padding: 4px 8px;
        margin: 2px;
    }}
    QMenuBar::item:selected {{
        background: {c['accent_soft']};
        color: {c['text_strong']};
    }}
    QMenu {{
        background: {c['surface']};
        color: {c['text_strong']};
        border: 1px solid {c['border']};
        padding: 4px;
    }}
    QMenu::item {{
        border-radius: 5px;
        padding: 5px 26px 5px 22px;
    }}
    QMenu::item:selected {{
        background: {c['accent_soft']};
        color: {c['text_strong']};
    }}
    QMenu::item:checked {{
        font-weight: 600;
    }}
    QMenu::separator {{
        height: 1px;
        background: {c['border_soft']};
        margin: 4px 6px;
    }}
    """
