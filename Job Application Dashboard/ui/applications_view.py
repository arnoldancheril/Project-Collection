"""
Applications View – Full application list with inline editing, search, filters,
sortable columns, and modern status display.
Upgraded to PyQt6 with consistent status colors.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QLineEdit, QFrame, QDialog, QFormLayout, QDateEdit,
    QTextEdit, QSizePolicy, QMessageBox, QCompleter, QScrollArea,
    QMenu,
)
from PyQt6.QtCore import Qt, QDate, QStringListModel
from PyQt6.QtGui import QFont, QColor, QBrush, QPixmap, QPainter, QIcon, QAction

from assets.theme import (
    COLORS, FONT, RADIUS, STATUS_COLORS,
    PRIMARY_BTN, SECONDARY_BTN, DANGER_BTN, SUCCESS_BTN,
    TABLE_STYLE, INPUT_STYLE, COMBO_STYLE, DATE_STYLE, DIALOG_STYLE,
    SEARCH_STYLE, CARD_STYLE,
)
from database.db_manager_v2 import DatabaseManager
from datetime import datetime

VALID_STATUSES = ["Applied", "Interviewing", "Offer", "Rejected"]

PREDEFINED_ROLES = [
    "Software Engineer", "Software Developer", "Full Stack Developer",
    "Backend Developer", "Frontend Developer", "DevOps Engineer",
    "Data Engineer", "Machine Learning Engineer", "Mobile Developer",
    "Cloud Engineer", "Site Reliability Engineer", "Security Engineer",
    "Systems Engineer", "QA Engineer", "Platform Engineer",
]

# Major US cities for location autocomplete
PREDEFINED_LOCATIONS = [
    "Remote", "Hybrid",
    "San Francisco, CA", "San Jose, CA", "San Diego, CA", "Santa Clara, CA",
    "Sunnyvale, CA", "Mountain View, CA", "Palo Alto, CA", "Cupertino, CA",
    "Los Angeles, CA", "Irvine, CA", "Sacramento, CA", "Oakland, CA",
    "New York, NY", "Brooklyn, NY", "Manhattan, NY",
    "Seattle, WA", "Bellevue, WA", "Redmond, WA",
    "Austin, TX", "Dallas, TX", "Houston, TX", "San Antonio, TX", "Plano, TX",
    "Boston, MA", "Cambridge, MA",
    "Chicago, IL",
    "Denver, CO", "Boulder, CO",
    "Atlanta, GA",
    "Portland, OR",
    "Phoenix, AZ", "Scottsdale, AZ",
    "Detroit, MI", "Ann Arbor, MI",
    "Washington, DC", "Arlington, VA", "Reston, VA", "McLean, VA",
    "Philadelphia, PA", "Pittsburgh, PA",
    "Minneapolis, MN",
    "Nashville, TN",
    "Charlotte, NC", "Raleigh, NC", "Durham, NC",
    "Miami, FL", "Tampa, FL", "Orlando, FL",
    "Salt Lake City, UT",
    "Columbus, OH", "Cincinnati, OH",
    "Indianapolis, IN",
    "Kansas City, MO", "St. Louis, MO",
    "San Bruno, CA", "Menlo Park, CA",
]


class ApplicationsView(QWidget):
    """Full applications management view."""

    def __init__(self, db: DatabaseManager, main_window=None, parent=None):
        super().__init__(parent)
        self.db = db
        self.main_window = main_window
        self.applications = []
        self.current_sort_col = 3  # Default sort by Date Applied
        self.current_sort_order = Qt.SortOrder.DescendingOrder
        self.setStyleSheet(f"background-color: {COLORS['bg']};")
        self._build()

    # ── Build UI ──────────────────────────────────────────────────────
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 28, 32, 28)
        outer.setSpacing(20)

        # Header row
        header = QHBoxLayout()
        title = QLabel("Applications")
        title.setStyleSheet(f"""
            font-size: {FONT['xxl']}px; font-weight: 700;
            color: {COLORS['text']}; background: transparent;
        """)
        header.addWidget(title)
        header.addStretch()

        add_btn = QPushButton("＋  Add Application")
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.setStyleSheet(PRIMARY_BTN)
        add_btn.clicked.connect(self.open_add_dialog)
        header.addWidget(add_btn)
        outer.addLayout(header)

        # Filter bar
        filter_bar = QHBoxLayout()
        filter_bar.setSpacing(12)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍  Search by company or role…")
        self.search_input.setStyleSheet(SEARCH_STYLE)
        self.search_input.setMinimumWidth(280)
        self.search_input.textChanged.connect(self._apply_filters)
        filter_bar.addWidget(self.search_input, 1)

        self.status_filter = QComboBox()
        self.status_filter.setStyleSheet(COMBO_STYLE)
        self.status_filter.addItems(["All Statuses"] + VALID_STATUSES)
        self.status_filter.currentTextChanged.connect(self._apply_filters)
        self.status_filter.setFixedWidth(160)
        filter_bar.addWidget(self.status_filter)

        clear_btn = QPushButton("Clear")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.setStyleSheet(SECONDARY_BTN)
        clear_btn.clicked.connect(self._clear_filters)
        filter_bar.addWidget(clear_btn)

        outer.addLayout(filter_bar)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Company ↕", "Role ↕", "Location ↕", "Date Applied ↕", "Status", "Actions"
        ])
        self.table.setStyleSheet(TABLE_STYLE)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(52)
        self.table.setShowGrid(False)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Enable sorting by clicking column headers
        h = self.table.horizontalHeader()
        h.setSectionsClickable(True)
        h.sectionClicked.connect(self._on_header_clicked)

        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        h.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 180)
        self.table.setColumnWidth(2, 200)  # Wider for full city names
        self.table.setColumnWidth(4, 160)  # Wider for status pills
        self.table.setColumnWidth(5, 160)

        outer.addWidget(self.table, 1)

        # Status bar
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"""
            font-size: {FONT['sm']}px; color: {COLORS['text_muted']};
            background: transparent; padding: 4px 0;
        """)
        outer.addWidget(self.status_label)

    # ── Sorting ──────────────────────────────────────────────────────
    def _on_header_clicked(self, logical_index):
        """Handle column header click for sorting."""
        if logical_index in (4, 5):  # Don't sort by Status widget or Actions
            return
        if self.current_sort_col == logical_index:
            # Toggle sort order
            if self.current_sort_order == Qt.SortOrder.AscendingOrder:
                self.current_sort_order = Qt.SortOrder.DescendingOrder
            else:
                self.current_sort_order = Qt.SortOrder.AscendingOrder
        else:
            self.current_sort_col = logical_index
            self.current_sort_order = Qt.SortOrder.AscendingOrder

        self._apply_filters()

    # ── Data ──────────────────────────────────────────────────────────
    def refresh(self):
        self.applications = self.db.get_applications()
        self._populate_table(self.applications)

    def _apply_filters(self):
        text = self.search_input.text().lower().strip()
        status = self.status_filter.currentText()

        filtered = self.applications
        if text:
            filtered = [a for a in filtered
                        if text in a.get("company", "").lower()
                        or text in a.get("role", "").lower()
                        or text in a.get("location", "").lower()]
        if status != "All Statuses":
            filtered = [a for a in filtered if a.get("status") == status]

        self._populate_table(filtered)

    def _clear_filters(self):
        self.search_input.clear()
        self.status_filter.setCurrentIndex(0)
        self._populate_table(self.applications)

    def _populate_table(self, apps):
        self.table.setRowCount(0)

        # Sort based on current sort column and order
        col_map = {0: "company", 1: "role", 2: "location", 3: "date_applied"}
        sort_key_name = col_map.get(self.current_sort_col, "date_applied")
        reverse = self.current_sort_order == Qt.SortOrder.DescendingOrder

        def sort_key(a):
            val = a.get(sort_key_name, "")
            if sort_key_name == "date_applied":
                try:
                    return datetime.strptime(val or "", "%Y-%m-%d").timestamp()
                except (ValueError, TypeError):
                    return 0
            return (val or "").lower()

        apps = sorted(apps, key=sort_key, reverse=reverse)

        for row, app in enumerate(apps):
            self.table.insertRow(row)

            # Company
            item = QTableWidgetItem(app.get("company", ""))
            item.setData(Qt.ItemDataRole.UserRole, app.get("id"))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, item)

            # Role
            role_item = QTableWidgetItem(app.get("role", ""))
            role_item.setFlags(role_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, role_item)

            # Location – show at least the first full city
            raw_loc = app.get("location", "") or "—"
            if ";" in raw_loc:
                parts = [p.strip() for p in raw_loc.split(";") if p.strip()]
                if len(parts) > 1:
                    display_loc = f"{parts[0]}  (+{len(parts)-1})"
                else:
                    display_loc = parts[0] if parts else "—"
            else:
                display_loc = raw_loc

            loc_item = QTableWidgetItem(display_loc)
            loc_item.setToolTip(raw_loc.replace(";", "\n"))  # Full location on hover
            loc_item.setForeground(QBrush(QColor(COLORS["text_secondary"])))
            loc_item.setFlags(loc_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 2, loc_item)

            # Date
            raw = app.get("date_applied", "")
            try:
                display_date = datetime.strptime(raw, "%Y-%m-%d").strftime("%b %d, %Y")
            except (ValueError, TypeError):
                display_date = raw
            date_item = QTableWidgetItem(display_date)
            date_item.setFlags(date_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 3, date_item)

            # Status – solid color pill matching dashboard colors
            status_widget = self._create_status_button(app)
            self.table.setCellWidget(row, 4, status_widget)

            # Actions
            actions = self._create_actions(app)
            self.table.setCellWidget(row, 5, actions)

        self.status_label.setText(f"Showing {len(apps)} application{'s' if len(apps) != 1 else ''}")

    def _create_status_button(self, app):
        """Create a clickable status pill with SOLID background matching dashboard."""
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        current_status = app.get("status", "Applied")
        color = STATUS_COLORS.get(current_status, COLORS["primary"])

        btn = QPushButton(f"  {current_status}  ▾")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedHeight(32)
        btn.setMinimumWidth(140)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 16px;
                padding: 4px 14px;
                font-weight: 600;
                font-size: {FONT['xs']}px;
            }}
            QPushButton:hover {{
                background-color: {color};
                filter: brightness(1.1);
            }}
            QPushButton::menu-indicator {{ image: none; width: 0; }}
        """)

        menu = QMenu(btn)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 6px;
            }}
            QMenu::item {{
                padding: 8px 20px;
                border-radius: 6px;
                margin: 2px 4px;
                color: {COLORS['text']};
                font-size: {FONT['md']}px;
            }}
            QMenu::item:selected {{
                background-color: {COLORS['primary_light']};
                color: {COLORS['primary']};
            }}
        """)

        for status in VALID_STATUSES:
            scolor = STATUS_COLORS.get(status, COLORS["primary"])
            action = menu.addAction(f"● {status}")
            action.triggered.connect(
                lambda checked, s=status, b=btn, a=app: self._change_status(a, s, b)
            )

        btn.setMenu(menu)
        layout.addWidget(btn)
        return container

    def _change_status(self, app, new_status, btn):
        """Handle status change from the menu."""
        self.db.update_application(app["id"], status=new_status)
        self.applications = self.db.get_applications()

        # Update button appearance with solid color
        color = STATUS_COLORS.get(new_status, COLORS["primary"])
        btn.setText(f"  {new_status}  ▾")
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 16px;
                padding: 4px 14px;
                font-weight: 600;
                font-size: {FONT['xs']}px;
            }}
            QPushButton:hover {{
                background-color: {color};
                filter: brightness(1.1);
            }}
            QPushButton::menu-indicator {{ image: none; width: 0; }}
        """)

        if self.main_window and hasattr(self.main_window, "dashboard_view"):
            self.main_window.dashboard_view._refresh_stat_cards()

    def _create_actions(self, app):
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(6)

        edit_btn = QPushButton("Edit")
        edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        edit_btn.setFixedSize(60, 30)
        edit_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']}; color: white;
                border: none; border-radius: 6px;
                font-size: {FONT['xs']}px; font-weight: 600;
            }}
            QPushButton:hover {{ background-color: {COLORS['primary_hover']}; }}
        """)
        edit_btn.clicked.connect(lambda: self._edit_app(app["id"]))

        del_btn = QPushButton("Delete")
        del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        del_btn.setFixedSize(60, 30)
        del_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent; color: {COLORS['danger']};
                border: 1px solid {COLORS['danger']}; border-radius: 6px;
                font-size: {FONT['xs']}px; font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {COLORS['danger']}; color: white;
            }}
        """)
        del_btn.clicked.connect(lambda: self._delete_app(app["id"]))

        layout.addWidget(edit_btn)
        layout.addWidget(del_btn)
        layout.addStretch()
        return container

    # ── Dialogs ──────────────────────────────────────────────────────
    def open_add_dialog(self):
        dlg = ApplicationDialog(self.db, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.refresh()
            if self.main_window:
                self.main_window.dashboard_view.refresh()

    def _edit_app(self, app_id):
        app_data = self.db.get_application(app_id)
        if not app_data:
            return
        dlg = ApplicationDialog(self.db, app_data=app_data, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.refresh()
            if self.main_window:
                self.main_window.dashboard_view.refresh()

    def _delete_app(self, app_id):
        reply = QMessageBox.question(
            self, "Delete Application",
            "Are you sure you want to delete this application?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.db.delete_application(app_id)
            self.refresh()
            if self.main_window:
                self.main_window.dashboard_view.refresh()


# ======================================================================
#  Add / Edit Dialog – Larger window, improved autocomplete, multi-location
# ======================================================================
class ApplicationDialog(QDialog):
    """Modal dialog for adding or editing an application."""

    def __init__(self, db: DatabaseManager, app_data=None, parent=None):
        super().__init__(parent)
        self.db = db
        self.app_data = app_data
        self.is_edit = app_data is not None
        self.setWindowTitle("Edit Application" if self.is_edit else "New Application")
        self.setFixedWidth(620)
        self.setMinimumHeight(720)  # Larger to reduce scrolling
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['surface']};
                border-radius: {RADIUS['lg']}px;
            }}
            QLabel {{
                background: transparent;
                color: {COLORS['text']};
            }}
        """)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Modern gradient header ──
        header = QWidget()
        header.setFixedHeight(64)
        header.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {COLORS['primary']}, stop:1 #818CF8);
            border-top-left-radius: {RADIUS['lg']}px;
            border-top-right-radius: {RADIUS['lg']}px;
        """)
        h_layout = QVBoxLayout(header)
        h_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_lbl = QLabel("✏️  Edit Application" if self.is_edit else "✨  New Application")
        title_lbl.setStyleSheet("""
            color: white; font-size: 20px; font-weight: 700;
            background: transparent;
        """)
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h_layout.addWidget(title_lbl)
        layout.addWidget(header)

        # ── Scrollable form content ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")

        form_container = QWidget()
        form_container.setStyleSheet("background: transparent;")
        form_layout = QVBoxLayout(form_container)
        form_layout.setContentsMargins(32, 20, 32, 16)
        form_layout.setSpacing(14)

        label_style = f"""
            font-weight: 600; font-size: {FONT['sm']}px;
            color: {COLORS['text_secondary']}; background: transparent;
            text-transform: uppercase; letter-spacing: 0.5px;
        """
        field_style = f"""
            QLineEdit, QComboBox, QDateEdit {{
                background-color: {COLORS['surface']};
                color: {COLORS['text']};
                border: 1.5px solid {COLORS['border']};
                border-radius: {RADIUS['sm']}px;
                padding: 10px 14px;
                font-size: {FONT['md']}px;
            }}
            QLineEdit:focus, QComboBox:focus, QDateEdit:focus {{
                border-color: {COLORS['primary']};
                background-color: #FAFAFF;
            }}
            QLineEdit::placeholder {{
                color: {COLORS['text_muted']};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 30px;
                border: none;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {COLORS['text_secondary']};
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS['surface']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: {RADIUS['sm']}px;
                selection-background-color: {COLORS['primary_light']};
                selection-color: {COLORS['primary']};
                outline: none;
                padding: 4px;
            }}
            QComboBox QLineEdit {{
                border: none;
                padding: 0px;
                background: transparent;
                color: {COLORS['text']};
            }}
            QDateEdit::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 30px;
                border: none;
            }}
        """

        # -- Company (autocomplete sorted by frequency) --
        lbl = QLabel("COMPANY")
        lbl.setStyleSheet(label_style)
        form_layout.addWidget(lbl)
        self.company_input = QLineEdit()
        self.company_input.setPlaceholderText("e.g. Google, Apple, Microsoft")
        self.company_input.setStyleSheet(field_style)

        # Get companies sorted by frequency for better autocomplete
        freq_companies = self.db.get_companies_by_frequency()
        all_companies = self.db.get_unique_companies()
        seen = set()
        company_suggestions = []
        for c in freq_companies:
            if c.lower() not in seen:
                company_suggestions.append(c)
                seen.add(c.lower())
        for c in all_companies:
            if c.lower() not in seen:
                company_suggestions.append(c)
                seen.add(c.lower())

        comp = QCompleter(company_suggestions)
        comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        comp.setFilterMode(Qt.MatchFlag.MatchContains)
        comp.setMaxVisibleItems(12)
        self.company_input.setCompleter(comp)
        form_layout.addWidget(self.company_input)

        # -- Role (autocomplete sorted by frequency, most-applied first) --
        lbl = QLabel("ROLE")
        lbl.setStyleSheet(label_style)
        form_layout.addWidget(lbl)
        self.role_input = QComboBox()
        self.role_input.setEditable(True)
        self.role_input.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)

        # Build role list: DB frequency first, then predefined
        freq_roles = self.db.get_roles_by_frequency()
        seen_roles = set()
        role_list = []
        for r in freq_roles:
            if r.lower() not in seen_roles:
                role_list.append(r)
                seen_roles.add(r.lower())
        for r in PREDEFINED_ROLES:
            if r.lower() not in seen_roles:
                role_list.append(r)
                seen_roles.add(r.lower())

        self.role_input.addItems(role_list)
        self.role_input.setCurrentText("")
        self.role_input.lineEdit().setPlaceholderText("Select or type role")
        self.role_input.setStyleSheet(field_style)

        role_comp = QCompleter(role_list)
        role_comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        role_comp.setFilterMode(Qt.MatchFlag.MatchContains)
        role_comp.setMaxVisibleItems(12)
        self.role_input.setCompleter(role_comp)
        form_layout.addWidget(self.role_input)

        # -- Locations (multi-location support) --
        lbl = QLabel("LOCATION(S)")
        lbl.setStyleSheet(label_style)
        form_layout.addWidget(lbl)

        self.locations_container = QVBoxLayout()
        self.locations_container.setSpacing(6)
        self.location_inputs = []
        form_layout.addLayout(self.locations_container)

        # Build location suggestions: DB frequency first, then predefined
        freq_locs = self.db.get_locations_by_frequency()
        seen_locs = set()
        self.location_suggestions = []
        for loc in freq_locs:
            if ";" not in loc and loc.lower() not in seen_locs:
                self.location_suggestions.append(loc)
                seen_locs.add(loc.lower())
        for loc in PREDEFINED_LOCATIONS:
            if loc.lower() not in seen_locs:
                self.location_suggestions.append(loc)
                seen_locs.add(loc.lower())

        # Add first location input
        self._add_location_input()

        # Add location button
        add_loc_btn = QPushButton("＋  Add another location")
        add_loc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_loc_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {COLORS['primary']};
                border: 1px dashed {COLORS['primary']};
                border-radius: {RADIUS['sm']}px;
                padding: 6px 14px;
                font-size: {FONT['sm']}px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_light']};
            }}
        """)
        add_loc_btn.clicked.connect(self._add_location_input)
        form_layout.addWidget(add_loc_btn)

        # -- Date & Status on same row --
        row = QHBoxLayout()
        row.setSpacing(16)

        date_col = QVBoxLayout()
        lbl = QLabel("DATE APPLIED")
        lbl.setStyleSheet(label_style)
        date_col.addWidget(lbl)
        self.date_input = QDateEdit()
        self.date_input.setCalendarPopup(True)
        self.date_input.setDate(QDate.currentDate())
        self.date_input.setStyleSheet(field_style)
        date_col.addWidget(self.date_input)
        row.addLayout(date_col, 1)

        status_col = QVBoxLayout()
        lbl = QLabel("STATUS")
        lbl.setStyleSheet(label_style)
        status_col.addWidget(lbl)
        self.status_input = QComboBox()
        self.status_input.addItems(VALID_STATUSES)
        self.status_input.setStyleSheet(field_style)
        status_col.addWidget(self.status_input)
        row.addLayout(status_col, 1)

        form_layout.addLayout(row)

        # -- Job URL --
        lbl = QLabel("JOB URL")
        lbl.setStyleSheet(label_style)
        form_layout.addWidget(lbl)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://... (optional)")
        self.url_input.setStyleSheet(field_style)
        form_layout.addWidget(self.url_input)

        # -- Notes --
        lbl = QLabel("NOTES")
        lbl.setStyleSheet(label_style)
        form_layout.addWidget(lbl)
        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText("Any additional notes…")
        self.notes_input.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['surface']};
                color: {COLORS['text']};
                border: 1.5px solid {COLORS['border']};
                border-radius: {RADIUS['sm']}px;
                padding: 10px 14px;
                font-size: {FONT['md']}px;
            }}
            QTextEdit:focus {{
                border-color: {COLORS['primary']};
                background-color: #FAFAFF;
            }}
        """)
        self.notes_input.setMaximumHeight(90)
        form_layout.addWidget(self.notes_input)

        scroll.setWidget(form_container)
        layout.addWidget(scroll, 1)

        # ── Bottom button bar ──
        btn_bar = QWidget()
        btn_bar.setStyleSheet(f"""
            background-color: {COLORS['surface_alt']};
            border-top: 1px solid {COLORS['border']};
        """)
        btn_layout = QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(24, 14, 24, 14)
        btn_layout.addStretch()

        cancel = QPushButton("Cancel")
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel.setStyleSheet(SECONDARY_BTN)
        cancel.setFixedHeight(40)
        cancel.clicked.connect(self.reject)
        btn_layout.addWidget(cancel)

        save = QPushButton("💾  Save" if self.is_edit else "✚  Add Application")
        save.setCursor(Qt.CursorShape.PointingHandCursor)
        save.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: {RADIUS['md']}px;
                padding: 10px 28px;
                font-size: {FONT['md']}px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_hover']};
            }}
        """)
        save.setFixedHeight(40)
        save.clicked.connect(self._save)
        btn_layout.addWidget(save)

        layout.addWidget(btn_bar)

        # Pre-fill if editing
        if self.is_edit:
            self.company_input.setText(self.app_data.get("company", ""))
            self.role_input.setCurrentText(self.app_data.get("role", ""))

            # Handle multiple locations
            existing_loc = self.app_data.get("location", "")
            if existing_loc and ";" in existing_loc:
                parts = [p.strip() for p in existing_loc.split(";") if p.strip()]
                if parts:
                    self.location_inputs[0].setText(parts[0])
                    for p in parts[1:]:
                        self._add_location_input(p)
            elif existing_loc:
                self.location_inputs[0].setText(existing_loc)

            self.url_input.setText(self.app_data.get("job_url", ""))
            self.notes_input.setPlainText(self.app_data.get("notes", ""))
            idx = self.status_input.findText(self.app_data.get("status", "Applied"))
            if idx >= 0:
                self.status_input.setCurrentIndex(idx)
            try:
                d = datetime.strptime(self.app_data.get("date_applied", ""), "%Y-%m-%d").date()
                self.date_input.setDate(QDate(d.year, d.month, d.day))
            except (ValueError, TypeError):
                pass

    def _add_location_input(self, text=""):
        """Add a new location input field with autocomplete."""
        row_layout = QHBoxLayout()
        row_layout.setSpacing(6)

        loc_input = QLineEdit()
        loc_input.setPlaceholderText("e.g. Remote, San Francisco, CA")
        loc_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLORS['surface']};
                color: {COLORS['text']};
                border: 1.5px solid {COLORS['border']};
                border-radius: {RADIUS['sm']}px;
                padding: 10px 14px;
                font-size: {FONT['md']}px;
            }}
            QLineEdit:focus {{
                border-color: {COLORS['primary']};
                background-color: #FAFAFF;
            }}
            QLineEdit::placeholder {{
                color: {COLORS['text_muted']};
            }}
        """)

        loc_comp = QCompleter(self.location_suggestions)
        loc_comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        loc_comp.setFilterMode(Qt.MatchFlag.MatchContains)
        loc_comp.setMaxVisibleItems(10)
        loc_input.setCompleter(loc_comp)

        if text:
            loc_input.setText(text)

        row_layout.addWidget(loc_input, 1)

        # Remove button (not for the first location)
        if len(self.location_inputs) > 0:
            remove_btn = QPushButton("✕")
            remove_btn.setFixedSize(32, 32)
            remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            remove_btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    color: {COLORS['danger']};
                    border: 1px solid {COLORS['danger']};
                    border-radius: 6px;
                    font-size: 14px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['danger']};
                    color: white;
                }}
            """)
            remove_btn.clicked.connect(lambda _, li=loc_input, rl=row_layout: self._remove_location(li, rl))
            row_layout.addWidget(remove_btn)

        self.location_inputs.append(loc_input)
        self.locations_container.addLayout(row_layout)

    def _remove_location(self, loc_input, row_layout):
        """Remove a location input field."""
        if loc_input in self.location_inputs:
            self.location_inputs.remove(loc_input)
        while row_layout.count():
            item = row_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _save(self):
        company = self.company_input.text().strip()
        role = self.role_input.currentText().strip()
        if not company or not role:
            QMessageBox.warning(self, "Missing Fields", "Company and Role are required.")
            return

        # Collect all locations, join with semicolons
        locations = []
        for loc_input in self.location_inputs:
            loc = loc_input.text().strip()
            if loc:
                locations.append(loc)
        location = "; ".join(locations)

        date_applied = self.date_input.date().toString("yyyy-MM-dd")
        status = self.status_input.currentText()
        url = self.url_input.text().strip()
        notes = self.notes_input.toPlainText().strip()

        if self.is_edit:
            self.db.update_application(
                self.app_data["id"],
                company=company, role=role, location=location,
                date_applied=date_applied, status=status,
                job_url=url, notes=notes,
            )
        else:
            self.db.add_application(
                company=company, role=role, date_applied=date_applied,
                status=status, location=location,
                job_url=url, notes=notes,
            )

        self.accept()
