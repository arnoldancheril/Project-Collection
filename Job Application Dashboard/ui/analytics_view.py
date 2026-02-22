"""
Analytics View – Clean charts and statistics using matplotlib.
Upgraded to PyQt6 with fixed date display and proper chart rendering.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QSizePolicy, QGridLayout, QPushButton,
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from collections import defaultdict
from datetime import datetime

from assets.theme import COLORS, FONT, RADIUS, STATUS_COLORS, CARD_STYLE, PRIMARY_BTN
from database.db_manager_v2 import DatabaseManager


class MetricCard(QFrame):
    """Small card showing a metric value and label."""

    def __init__(self, label, value, color=None, parent=None):
        super().__init__(parent)
        color = color or COLORS["primary"]
        self.setStyleSheet(f"""
            MetricCard {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: {RADIUS['lg']}px;
                padding: 10px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        val = QLabel(str(value))
        val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val.setStyleSheet(f"""
            font-size: {FONT['hero']}px; font-weight: 700;
            color: {color}; background: transparent;
        """)
        layout.addWidget(val)

        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"""
            font-size: {FONT['sm']}px; font-weight: 500;
            color: {COLORS['text_secondary']}; background: transparent;
        """)
        layout.addWidget(lbl)


class AnalyticsView(QWidget):
    """Analytics dashboard with charts."""

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
        self.main_layout = QVBoxLayout(container)
        self.main_layout.setContentsMargins(32, 28, 32, 28)
        self.main_layout.setSpacing(24)

        # Header
        header = QHBoxLayout()
        title = QLabel("Analytics")
        title.setStyleSheet(f"""
            font-size: {FONT['xxl']}px; font-weight: 700;
            color: {COLORS['text']}; background: transparent;
        """)
        header.addWidget(title)
        header.addStretch()

        add_btn = QPushButton("＋  Add Application")
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.setStyleSheet(PRIMARY_BTN)
        add_btn.clicked.connect(self._add_app)
        header.addWidget(add_btn)

        self.main_layout.addLayout(header)

        # Metric cards row
        self.metrics_layout = QHBoxLayout()
        self.metrics_layout.setSpacing(16)
        self._build_metrics()
        self.main_layout.addLayout(self.metrics_layout)

        # Charts – each in its own frame
        self.charts_container = QVBoxLayout()
        self.charts_container.setSpacing(20)
        self._build_all_charts()
        self.main_layout.addLayout(self.charts_container)

        self.main_layout.addStretch()

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _wrap_chart(self, canvas, title_text):
        """Wrap a matplotlib canvas in a styled frame with a title."""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: {RADIUS['lg']}px;
            }}
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        title = QLabel(title_text)
        title.setStyleSheet(f"""
            font-size: {FONT['lg']}px; font-weight: 600;
            color: {COLORS['text']}; background: transparent;
        """)
        layout.addWidget(title)
        layout.addWidget(canvas)
        return frame

    # ── Metrics ──────────────────────────────────────────────────────
    def _build_metrics(self):
        apps = self.db.get_applications()
        total = len(apps)
        counts = self.db.get_status_counts()

        interviewing = counts.get("Interviewing", 0)
        offers = counts.get("Offer", 0)
        rejected = counts.get("Rejected", 0)

        response_rate = ((interviewing + offers) / total * 100) if total else 0
        offer_rate = (offers / total * 100) if total else 0

        cards_data = [
            ("Total Applications", str(total), COLORS["primary"]),
            ("Response Rate", f"{response_rate:.1f}%", COLORS["info"]),
            ("Offer Rate", f"{offer_rate:.1f}%", COLORS["offer"]),
            ("Interviewing", str(interviewing), COLORS["interviewing"]),
            ("Rejected", str(rejected), COLORS["rejected"]),
        ]
        for label, value, color in cards_data:
            self.metrics_layout.addWidget(MetricCard(label, value, color))

    def _clear_metrics(self):
        while self.metrics_layout.count():
            item = self.metrics_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    # ── Build all charts ─────────────────────────────────────────────
    def _build_all_charts(self):
        # Row 1: pie + timeline
        row1 = QHBoxLayout()
        row1.setSpacing(20)

        pie_canvas = self._create_pie_chart()
        pie_frame = self._wrap_chart(pie_canvas, "Status Distribution")
        row1.addWidget(pie_frame, 1)

        bar_canvas = self._create_timeline_chart()
        bar_frame = self._wrap_chart(bar_canvas, "Applications Over Time")
        row1.addWidget(bar_frame, 1)

        self.charts_container.addLayout(row1)

        # Row 2: top companies + pipeline
        row2 = QHBoxLayout()
        row2.setSpacing(20)

        company_canvas = self._create_top_companies_chart()
        company_frame = self._wrap_chart(company_canvas, "Top Companies Applied To")
        row2.addWidget(company_frame, 1)

        rate_canvas = self._create_response_rate_chart()
        rate_frame = self._wrap_chart(rate_canvas, "Pipeline Conversion")
        row2.addWidget(rate_frame, 1)

        self.charts_container.addLayout(row2)

    def _clear_charts(self):
        """Remove all chart layouts and their widgets."""
        while self.charts_container.count():
            item = self.charts_container.takeAt(0)
            if item.layout():
                layout = item.layout()
                while layout.count():
                    child = layout.takeAt(0)
                    w = child.widget()
                    if w:
                        w.deleteLater()
            elif item.widget():
                item.widget().deleteLater()

    # ── Charts ───────────────────────────────────────────────────────
    def _create_pie_chart(self):
        counts = self.db.get_status_counts()
        fig = Figure(figsize=(5, 4), dpi=100, facecolor="white")
        canvas = FigureCanvas(fig)

        if counts and any(v > 0 for v in counts.values()):
            # Filter out zero counts
            labels = [k for k, v in counts.items() if v > 0]
            sizes = [v for v in counts.values() if v > 0]
            colors = [STATUS_COLORS.get(s, COLORS["primary"]) for s in labels]

            ax = fig.add_subplot(111)
            wedges, texts, autotexts = ax.pie(
                sizes, labels=None, colors=colors,
                autopct="%1.0f%%", startangle=90, pctdistance=0.75,
                textprops={"fontsize": 11, "weight": "bold"},
                wedgeprops={"edgecolor": "white", "linewidth": 2},
            )
            for at in autotexts:
                at.set_color("white")
                at.set_weight("bold")
                at.set_fontsize(10)

            # Add legend instead of labels on the pie (cleaner)
            ax.legend(
                labels, loc="lower center", ncol=len(labels),
                fontsize=10, frameon=False,
                bbox_to_anchor=(0.5, -0.05),
            )
            ax.axis("equal")
        else:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    fontsize=14, color=COLORS["text_muted"])
            ax.axis("off")

        fig.tight_layout()
        return canvas

    def _create_timeline_chart(self):
        monthly = self.db.get_applications_by_month()
        fig = Figure(figsize=(5, 4), dpi=100, facecolor="white")
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        if monthly:
            months = [m for m, _ in monthly]
            counts_vals = [c for _, c in monthly]

            # Better date formatting
            display = []
            for m in months:
                try:
                    display.append(datetime.strptime(m, "%Y-%m").strftime("%b\n%Y"))
                except ValueError:
                    display.append(m)

            x_pos = range(len(display))
            bars = ax.bar(x_pos, counts_vals, color=COLORS["primary"], alpha=0.85,
                          width=0.6, edgecolor="white", linewidth=0.5)

            # Value labels on bars
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                            str(int(h)), ha="center", va="bottom",
                            fontsize=10, fontweight="bold", color=COLORS["text"])

            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(display, fontsize=9, ha="center")
            ax.set_ylabel("Applications", fontsize=11, color=COLORS["text_secondary"])

            # Only show integer ticks on y-axis
            max_count = max(counts_vals) if counts_vals else 1
            ax.set_ylim(0, max_count + max(1, max_count * 0.2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        else:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    fontsize=14, color=COLORS["text_muted"])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLORS["border"])
        ax.spines["bottom"].set_color(COLORS["border"])
        ax.tick_params(colors=COLORS["text_secondary"])
        ax.grid(axis="y", alpha=0.3, color=COLORS["border"])
        fig.tight_layout(pad=2.0)
        return canvas

    def _create_top_companies_chart(self):
        apps = self.db.get_applications()
        fig = Figure(figsize=(5, 4), dpi=100, facecolor="white")
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        company_counts = defaultdict(int)
        for a in apps:
            company_counts[a.get("company", "Unknown")] += 1

        if company_counts:
            sorted_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            names = [c for c, _ in sorted_companies]
            counts_vals = [c for _, c in sorted_companies]

            y_pos = range(len(names))
            bars = ax.barh(y_pos, counts_vals, color=COLORS["primary"], alpha=0.85,
                           height=0.6, edgecolor="white")
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(names, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel("Applications", fontsize=11, color=COLORS["text_secondary"])

            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            for bar in bars:
                w = bar.get_width()
                if w > 0:
                    ax.text(w + 0.1, bar.get_y() + bar.get_height() / 2,
                            str(int(w)), va="center", fontsize=10,
                            fontweight="bold", color=COLORS["text"])
        else:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    fontsize=14, color=COLORS["text_muted"])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLORS["border"])
        ax.spines["bottom"].set_color(COLORS["border"])
        ax.tick_params(colors=COLORS["text_secondary"])
        fig.tight_layout(pad=2.0)
        return canvas

    def _create_response_rate_chart(self):
        counts = self.db.get_status_counts()
        fig = Figure(figsize=(5, 4), dpi=100, facecolor="white")
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        total = sum(counts.values()) if counts else 0

        if total > 0:
            # Pipeline: Applied → Interviewing → Offer
            applied = counts.get("Applied", 0) + counts.get("Interviewing", 0) + counts.get("Offer", 0) + counts.get("Rejected", 0)
            interviewing = counts.get("Interviewing", 0) + counts.get("Offer", 0)
            offers = counts.get("Offer", 0)

            stages = ["Applied\n(All)", "Interviewing\n+ Offers", "Offers"]
            values = [applied, interviewing, offers]
            stage_colors = [COLORS["applied"], COLORS["interviewing"], COLORS["offer"]]

            bars = ax.bar(stages, values, color=stage_colors, width=0.55,
                          edgecolor="white", linewidth=1.5)

            for bar, val in zip(bars, values):
                h = bar.get_height()
                pct = (val / applied * 100) if applied > 0 else 0
                label = f"{int(val)}\n({pct:.0f}%)"
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                        label, ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color=COLORS["text"])

            ax.set_ylabel("Count", fontsize=11, color=COLORS["text_secondary"])
            max_val = max(values) if values else 1
            ax.set_ylim(0, max_val + max(1, max_val * 0.3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        else:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    fontsize=14, color=COLORS["text_muted"])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLORS["border"])
        ax.spines["bottom"].set_color(COLORS["border"])
        ax.tick_params(colors=COLORS["text_secondary"])
        ax.grid(axis="y", alpha=0.3, color=COLORS["border"])
        fig.tight_layout(pad=2.0)
        return canvas

    # ── Actions ──────────────────────────────────────────────────────
    def _add_app(self):
        if self.main_window:
            self.main_window.navigate_to_applications()
            self.main_window.applications_view.open_add_dialog()

    # ── Refresh ──────────────────────────────────────────────────────
    def refresh(self):
        """Rebuild all charts and metrics with fresh data."""
        self._clear_metrics()
        self._build_metrics()
        self._clear_charts()
        self._build_all_charts()
