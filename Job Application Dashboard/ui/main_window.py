"""
Main Window v3.0 – Sidebar navigation with stacked views. PyQt6.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QStackedWidget, QSizePolicy, QFrame,
    QSpacerItem, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor

from assets.theme import (
    GLOBAL_STYLE, SIDEBAR_STYLE, COLORS, FONT, RADIUS,
    sidebar_button_style,
)
from database.db_manager_v2 import DatabaseManager
from ui.dashboard_view import DashboardView
from ui.applications_view import ApplicationsView
from ui.quick_answers_view import QuickAnswersView
from ui.analytics_view import AnalyticsView


class SidebarButton(QPushButton):
    """Navigation button for the sidebar."""

    def __init__(self, icon_text, label, parent=None):
        super().__init__(parent)
        self.icon_text = icon_text
        self.label = label
        self.setText(f"  {icon_text}   {label}")
        self.setFixedHeight(46)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(sidebar_button_style(False))

    def set_active(self, active: bool):
        self.setStyleSheet(sidebar_button_style(active))


class MainWindow(QMainWindow):
    """Application main window with sidebar navigation."""

    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self.db = db_manager
        self.setWindowTitle("Job Application Tracker")
        self.setMinimumSize(1280, 780)
        self.resize(1400, 860)
        self.setStyleSheet(GLOBAL_STYLE)
        self._build_ui()

    # ── Build UI ──────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        sidebar = self._build_sidebar()
        root.addWidget(sidebar)

        # Content stack
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background-color: {COLORS['bg']};")

        self.dashboard_view = DashboardView(self.db, self)
        self.applications_view = ApplicationsView(self.db, self)
        self.quick_answers_view = QuickAnswersView(self.db, self)
        self.analytics_view = AnalyticsView(self.db, self)

        self.stack.addWidget(self.dashboard_view)       # 0
        self.stack.addWidget(self.applications_view)    # 1
        self.stack.addWidget(self.quick_answers_view)   # 2
        self.stack.addWidget(self.analytics_view)       # 3

        root.addWidget(self.stack, 1)

        # Select dashboard by default
        self._navigate(0)

    def _build_sidebar(self):
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setStyleSheet(SIDEBAR_STYLE)
        sidebar.setFixedWidth(220)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 20, 12, 20)
        layout.setSpacing(6)

        # Logo / title area
        logo_label = QLabel("  💼  Tracker")
        logo_label.setStyleSheet(f"""
            color: #FFFFFF;
            font-size: {FONT['xl']}px;
            font-weight: 700;
            padding: 8px 4px 20px 4px;
            background: transparent;
        """)
        layout.addWidget(logo_label)

        # Divider
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background-color: {COLORS['sidebar_hover']};")
        layout.addWidget(div)
        layout.addSpacing(12)

        # Nav buttons
        self.nav_buttons: list[SidebarButton] = []
        nav_items = [
            ("📊", "Dashboard"),
            ("📋", "Applications"),
            ("📝", "Quick Answers"),
            ("📈", "Analytics"),
        ]
        for idx, (icon, label) in enumerate(nav_items):
            btn = SidebarButton(icon, label)
            btn.clicked.connect(lambda checked, i=idx: self._navigate(i))
            layout.addWidget(btn)
            self.nav_buttons.append(btn)

        layout.addStretch()

        # Version label
        ver = QLabel("v3.0")
        ver.setStyleSheet(f"""
            color: {COLORS['sidebar_text']};
            font-size: {FONT['xs']}px;
            padding: 4px;
            background: transparent;
        """)
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ver)

        return sidebar

    # ── Navigation ────────────────────────────────────────────────────
    def _navigate(self, index: int):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.set_active(i == index)

        # Refresh data when switching views
        widget = self.stack.currentWidget()
        if hasattr(widget, "refresh"):
            widget.refresh()

    def navigate_to_applications(self):
        """Public method for child views to switch to Applications tab."""
        self._navigate(1)

    def navigate_to_quick_answers(self):
        """Public method for child views to switch to Quick Answers tab."""
        self._navigate(2)
