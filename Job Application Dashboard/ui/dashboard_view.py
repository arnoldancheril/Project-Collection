"""
Dashboard View – Overview with stats cards, recent applications, and quick actions.
Upgraded to PyQt6.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QSizePolicy, QGridLayout, QSpacerItem,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from assets.theme import (
    COLORS, FONT, RADIUS, CARD_STYLE, PRIMARY_BTN, SECONDARY_BTN,
    status_card_style, STATUS_COLORS,
)
from database.db_manager_v2 import DatabaseManager
from datetime import datetime


class StatCard(QFrame):
    """Colored stat card showing a count and label."""

    def __init__(self, label, count, color, parent=None):
        super().__init__(parent)
        self.label_text = label
        self.count = count
        self.color = color
        self._build()

    def _build(self):
        self.setStyleSheet(f"""
            StatCard {{
                background-color: {self.color};
                border-radius: {RADIUS['lg']}px;
                min-height: 100px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._count_label = QLabel(str(self.count))
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._count_label.setStyleSheet(f"""
            color: white; background: transparent;
            font-size: {FONT['hero']}px; font-weight: 700;
        """)

        title = QLabel(self.label_text)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"""
            color: rgba(255,255,255,0.9); background: transparent;
            font-size: {FONT['md']}px; font-weight: 600;
        """)

        layout.addWidget(self._count_label)
        layout.addWidget(title)

    def set_count(self, n):
        self.count = n
        self._count_label.setText(str(n))


class RecentAppRow(QFrame):
    """Single row for a recent application."""

    def __init__(self, app_data, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            RecentAppRow {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border_light']};
                border-radius: {RADIUS['md']}px;
                padding: 4px;
            }}
            RecentAppRow:hover {{
                border-color: {COLORS['primary']};
                background-color: {COLORS['primary_light']};
            }}
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # Company & Role
        info = QVBoxLayout()
        info.setSpacing(2)
        company = QLabel(app_data.get("company", ""))
        company.setStyleSheet(f"font-weight: 600; font-size: {FONT['md']}px; color: {COLORS['text']}; background: transparent;")
        role = QLabel(app_data.get("role", ""))
        role.setStyleSheet(f"font-size: {FONT['sm']}px; color: {COLORS['text_secondary']}; background: transparent;")
        info.addWidget(company)
        info.addWidget(role)
        layout.addLayout(info, 1)

        # Location
        loc = app_data.get("location", "")
        if loc:
            loc_label = QLabel(f"📍 {loc}")
            loc_label.setStyleSheet(f"font-size: {FONT['xs']}px; color: {COLORS['text_muted']}; background: transparent;")
            layout.addWidget(loc_label)

        # Date
        raw_date = app_data.get("date_applied", "")
        try:
            dt = datetime.strptime(raw_date, "%Y-%m-%d")
            display_date = dt.strftime("%b %d, %Y")
        except (ValueError, TypeError):
            display_date = raw_date
        date_label = QLabel(display_date)
        date_label.setStyleSheet(f"font-size: {FONT['xs']}px; color: {COLORS['text_muted']}; background: transparent;")
        layout.addWidget(date_label)

        # Status badge
        status = app_data.get("status", "Applied")
        color = STATUS_COLORS.get(status, COLORS["primary"])
        badge = QLabel(status)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedWidth(100)
        badge.setStyleSheet(f"""
            background-color: {color}; color: white;
            border-radius: {RADIUS['pill']}px;
            padding: 4px 12px;
            font-size: {FONT['xs']}px; font-weight: 600;
        """)
        layout.addWidget(badge)


class DashboardView(QWidget):
    """Dashboard overview page."""

    def __init__(self, db: DatabaseManager, main_window=None, parent=None):
        super().__init__(parent)
        self.db = db
        self.main_window = main_window
        self.setStyleSheet(f"background-color: {COLORS['bg']};")
        self._build()

    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        self.layout_main = QVBoxLayout(container)
        self.layout_main.setContentsMargins(32, 28, 32, 28)
        self.layout_main.setSpacing(24)

        # Header
        header = QHBoxLayout()
        greeting = QLabel("Dashboard")
        greeting.setStyleSheet(f"""
            font-size: {FONT['xxl']}px; font-weight: 700;
            color: {COLORS['text']}; background: transparent;
        """)
        header.addWidget(greeting)
        header.addStretch()

        add_btn = QPushButton("＋  New Application")
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.setStyleSheet(PRIMARY_BTN)
        add_btn.clicked.connect(self._add_app)
        header.addWidget(add_btn)

        self.layout_main.addLayout(header)

        # Stat cards
        self.cards_layout = QHBoxLayout()
        self.cards_layout.setSpacing(16)
        self._build_stat_cards()
        self.layout_main.addLayout(self.cards_layout)

        # Quick actions row
        actions = QHBoxLayout()
        actions.setSpacing(12)

        qa_btn = QPushButton("📝  Quick Answers")
        qa_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        qa_btn.setStyleSheet(SECONDARY_BTN)
        qa_btn.clicked.connect(self._go_quick_answers)
        actions.addWidget(qa_btn)

        apps_btn = QPushButton("📋  View All Applications")
        apps_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        apps_btn.setStyleSheet(SECONDARY_BTN)
        apps_btn.clicked.connect(self._go_applications)
        actions.addWidget(apps_btn)

        analytics_btn = QPushButton("📈  Analytics")
        analytics_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        analytics_btn.setStyleSheet(SECONDARY_BTN)
        analytics_btn.clicked.connect(self._go_analytics)
        actions.addWidget(analytics_btn)

        actions.addStretch()
        self.layout_main.addLayout(actions)

        # Recent applications section
        section_header = QLabel("Recent Applications")
        section_header.setStyleSheet(f"""
            font-size: {FONT['xl']}px; font-weight: 600;
            color: {COLORS['text']}; background: transparent;
            margin-top: 8px;
        """)
        self.layout_main.addWidget(section_header)

        self.recent_container = QVBoxLayout()
        self.recent_container.setSpacing(8)
        self._build_recent()
        self.layout_main.addLayout(self.recent_container)

        self.layout_main.addStretch()

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── Stat cards ────────────────────────────────────────────────────
    def _build_stat_cards(self):
        counts = self.db.get_status_counts()
        total = self.db.get_total_count()

        self.stat_total = StatCard("Total", total, COLORS["primary"])
        self.stat_applied = StatCard("Applied", counts.get("Applied", 0), COLORS["applied"])
        self.stat_interview = StatCard("Interviewing", counts.get("Interviewing", 0), COLORS["interviewing"])
        self.stat_offer = StatCard("Offers", counts.get("Offer", 0), COLORS["offer"])
        self.stat_rejected = StatCard("Rejected", counts.get("Rejected", 0), COLORS["rejected"])

        for card in [self.stat_total, self.stat_applied, self.stat_interview,
                     self.stat_offer, self.stat_rejected]:
            self.cards_layout.addWidget(card)

    def _refresh_stat_cards(self):
        counts = self.db.get_status_counts()
        total = self.db.get_total_count()
        self.stat_total.set_count(total)
        self.stat_applied.set_count(counts.get("Applied", 0))
        self.stat_interview.set_count(counts.get("Interviewing", 0))
        self.stat_offer.set_count(counts.get("Offer", 0))
        self.stat_rejected.set_count(counts.get("Rejected", 0))

    # ── Recent applications ──────────────────────────────────────────
    def _build_recent(self):
        recent = self.db.get_recent_applications(8)
        if not recent:
            empty = QLabel("No applications yet. Click '+ New Application' to get started!")
            empty.setStyleSheet(f"""
                color: {COLORS['text_muted']}; font-size: {FONT['md']}px;
                background: transparent; padding: 30px;
            """)
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.recent_container.addWidget(empty)
            return

        for app in recent:
            self.recent_container.addWidget(RecentAppRow(app))

    def _clear_recent(self):
        while self.recent_container.count():
            item = self.recent_container.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    # ── Actions ──────────────────────────────────────────────────────
    def _add_app(self):
        if self.main_window:
            self.main_window.navigate_to_applications()
            self.main_window.applications_view.open_add_dialog()

    def _go_applications(self):
        if self.main_window:
            self.main_window.navigate_to_applications()

    def _go_quick_answers(self):
        if self.main_window:
            self.main_window.navigate_to_quick_answers()

    def _go_analytics(self):
        if self.main_window:
            self.main_window._navigate(3)

    # ── Refresh ──────────────────────────────────────────────────────
    def refresh(self):
        self._refresh_stat_cards()
        self._clear_recent()
        self._build_recent()
