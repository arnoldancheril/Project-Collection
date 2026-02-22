"""
Quick Answers View – Organized Q&A clipboard for common application questions.
Supports categories (Education, Work Experience, etc.), adding/editing/deleting
answers, one-click copy, search, and category management.
Upgraded to PyQt6.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QLineEdit, QTextEdit, QComboBox,
    QDialog, QFormLayout, QSizePolicy, QMessageBox, QApplication,
    QSplitter, QListWidget, QListWidgetItem, QToolButton, QInputDialog,
    QMenu,
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QFont, QColor, QIcon, QAction

from assets.theme import (
    COLORS, FONT, RADIUS, PRIMARY_BTN, SECONDARY_BTN, DANGER_BTN,
    SUCCESS_BTN, COPY_BTN, INPUT_STYLE, COMBO_STYLE, SEARCH_STYLE,
    CARD_STYLE,
)
from database.db_manager_v2 import DatabaseManager


# ======================================================================
#  Answer Card Widget
# ======================================================================
class AnswerCard(QFrame):
    """A single Q&A card with copy button."""

    def __init__(self, qa_data: dict, db: DatabaseManager,
                 on_edit=None, on_delete=None, on_change_category=None,
                 parent=None):
        super().__init__(parent)
        self.qa = qa_data
        self.db = db
        self.on_edit = on_edit
        self.on_delete = on_delete
        self.on_change_category = on_change_category
        self._build()

    def _build(self):
        self.setStyleSheet(f"""
            AnswerCard {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: {RADIUS['md']}px;
            }}
            AnswerCard:hover {{
                border-color: {COLORS['primary']};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        # Top: question + category badge + actions
        top = QHBoxLayout()
        top.setSpacing(10)

        q_label = QLabel(self.qa.get("question", ""))
        q_label.setWordWrap(True)
        q_label.setStyleSheet(f"""
            font-size: {FONT['md']}px; font-weight: 600;
            color: {COLORS['text']}; background: transparent;
        """)
        top.addWidget(q_label, 1)

        # Category badge (clickable to change)
        cat_icon = self.qa.get("category_icon", "📁")
        cat_name = self.qa.get("category_name", "")
        badge = QPushButton(f" {cat_icon} {cat_name} ")
        badge.setCursor(Qt.CursorShape.PointingHandCursor)
        badge.setToolTip("Click to change category")
        badge.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary_light']};
                color: {COLORS['primary']};
                border-radius: {RADIUS['pill']}px;
                padding: 3px 10px;
                font-size: {FONT['xs']}px; font-weight: 600;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary']};
                color: white;
            }}
        """)
        badge.clicked.connect(
            lambda: self.on_change_category(self.qa) if self.on_change_category else None
        )
        top.addWidget(badge)

        # Edit button
        edit_btn = QPushButton("✏️")
        edit_btn.setFixedSize(32, 32)
        edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        edit_btn.setToolTip("Edit")
        edit_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none;
                font-size: 14px; border-radius: 6px;
            }}
            QPushButton:hover {{ background-color: {COLORS['surface_alt']}; }}
        """)
        edit_btn.clicked.connect(lambda: self.on_edit(self.qa) if self.on_edit else None)
        top.addWidget(edit_btn)

        # Delete button
        del_btn = QPushButton("🗑️")
        del_btn.setFixedSize(32, 32)
        del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        del_btn.setToolTip("Delete")
        del_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none;
                font-size: 14px; border-radius: 6px;
            }}
            QPushButton:hover {{ background-color: #FEE2E2; }}
        """)
        del_btn.clicked.connect(lambda: self.on_delete(self.qa) if self.on_delete else None)
        top.addWidget(del_btn)

        layout.addLayout(top)

        # Separator
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {COLORS['border_light']};")
        layout.addWidget(sep)

        # Answer text
        answer_text = self.qa.get("answer", "")
        a_label = QLabel(answer_text)
        a_label.setWordWrap(True)
        a_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        a_label.setStyleSheet(f"""
            font-size: {FONT['md']}px; color: {COLORS['text_secondary']};
            background: transparent; line-height: 1.5;
            padding: 4px 0;
        """)
        layout.addWidget(a_label)

        # Bottom: copy button + times copied
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        copy_btn = QPushButton("📋  Copy Answer")
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.setStyleSheet(COPY_BTN)
        copy_btn.clicked.connect(self._copy)
        bottom.addWidget(copy_btn)

        times = self.qa.get("times_copied", 0)
        if times > 0:
            count_label = QLabel(f"Copied {times} time{'s' if times != 1 else ''}")
            count_label.setStyleSheet(f"""
                font-size: {FONT['xs']}px; color: {COLORS['text_muted']};
                background: transparent;
            """)
            bottom.addWidget(count_label)

        bottom.addStretch()
        layout.addLayout(bottom)

        # Feedback label (hidden by default)
        self.feedback = QLabel("✓ Copied!")
        self.feedback.setStyleSheet(f"""
            color: {COLORS['success']}; font-weight: 600;
            font-size: {FONT['sm']}px; background: transparent;
        """)
        self.feedback.hide()
        layout.addWidget(self.feedback)

    def _copy(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.qa.get("answer", ""))
        self.db.increment_copy_count(self.qa["id"])

        # Update the local count
        self.qa["times_copied"] = self.qa.get("times_copied", 0) + 1

        # Show feedback
        self.feedback.show()
        QTimer.singleShot(1500, self.feedback.hide)


# ======================================================================
#  Quick Answers View
# ======================================================================
class QuickAnswersView(QWidget):
    """Main Quick Answers page with category sidebar and answer cards."""

    def __init__(self, db: DatabaseManager, main_window=None, parent=None):
        super().__init__(parent)
        self.db = db
        self.main_window = main_window
        self.current_category_id = None  # None = show all
        self.setStyleSheet(f"background-color: {COLORS['bg']};")
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 28, 32, 28)
        outer.setSpacing(20)

        # Header
        header = QHBoxLayout()
        title = QLabel("Quick Answers")
        title.setStyleSheet(f"""
            font-size: {FONT['xxl']}px; font-weight: 700;
            color: {COLORS['text']}; background: transparent;
        """)
        header.addWidget(title)

        subtitle = QLabel("Store common application answers for quick copy-paste")
        subtitle.setStyleSheet(f"""
            font-size: {FONT['sm']}px; color: {COLORS['text_muted']};
            background: transparent; padding-top: 8px;
        """)
        header.addWidget(subtitle)
        header.addStretch()

        # Add Application button
        add_app_btn = QPushButton("＋  Add Application")
        add_app_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_app_btn.setStyleSheet(PRIMARY_BTN)
        add_app_btn.clicked.connect(self._add_application)
        header.addWidget(add_app_btn)

        add_btn = QPushButton("＋  Add Answer")
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
                border-radius: {RADIUS['md']}px;
                padding: 10px 22px;
                font-size: {FONT['md']}px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #16A34A;
            }}
        """)
        add_btn.clicked.connect(self._add_answer)
        header.addWidget(add_btn)

        outer.addLayout(header)

        # Search bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍  Search questions or answers…")
        self.search_input.setStyleSheet(SEARCH_STYLE)
        self.search_input.setMaximumWidth(500)
        self.search_input.textChanged.connect(self._on_search)
        outer.addWidget(self.search_input)

        # Splitter: category sidebar | answer cards
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle { background: transparent; width: 8px; }
        """)

        # Category list
        cat_panel = QWidget()
        cat_panel.setStyleSheet("background: transparent;")
        cat_panel.setFixedWidth(220)
        cat_layout = QVBoxLayout(cat_panel)
        cat_layout.setContentsMargins(0, 0, 8, 0)
        cat_layout.setSpacing(4)

        cat_header = QHBoxLayout()
        cat_header.setSpacing(6)
        cat_title = QLabel("Categories")
        cat_title.setStyleSheet(f"""
            font-size: {FONT['md']}px; font-weight: 600;
            color: {COLORS['text']}; background: transparent;
            padding-bottom: 6px;
        """)
        cat_header.addWidget(cat_title)
        cat_header.addStretch()

        add_cat_btn = QPushButton("＋")
        add_cat_btn.setFixedSize(26, 26)
        add_cat_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_cat_btn.setToolTip("Add new category")
        add_cat_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['surface_alt']};
                color: {COLORS['primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 13px;
                font-size: 14px;
                font-weight: 700;
                padding: 0;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_light']};
                border-color: {COLORS['primary']};
            }}
        """)
        add_cat_btn.clicked.connect(self._add_category)
        cat_header.addWidget(add_cat_btn)
        cat_layout.addLayout(cat_header)

        self.cat_list = QListWidget()
        self.cat_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: {RADIUS['md']}px;
                outline: none;
                padding: 6px;
            }}
            QListWidget::item {{
                padding: 10px 14px;
                border-radius: {RADIUS['sm']}px;
                margin: 2px 0;
                color: {COLORS['text']};
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['primary_light']};
                color: {COLORS['primary']};
                font-weight: 600;
            }}
            QListWidget::item:hover:!selected {{
                background-color: {COLORS['surface_alt']};
            }}
        """)
        self.cat_list.currentRowChanged.connect(self._on_category_changed)
        self.cat_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.cat_list.customContextMenuRequested.connect(self._cat_context_menu)
        cat_layout.addWidget(self.cat_list, 1)

        splitter.addWidget(cat_panel)

        # Answers scroll area
        ans_panel = QWidget()
        ans_panel.setStyleSheet("background: transparent;")
        ans_layout = QVBoxLayout(ans_panel)
        ans_layout.setContentsMargins(8, 0, 0, 0)
        ans_layout.setSpacing(0)

        self.answers_scroll = QScrollArea()
        self.answers_scroll.setWidgetResizable(True)
        self.answers_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.answers_scroll.setStyleSheet("background: transparent; border: none;")

        self.answers_container = QWidget()
        self.answers_container.setStyleSheet("background: transparent;")
        self.answers_layout = QVBoxLayout(self.answers_container)
        self.answers_layout.setContentsMargins(0, 0, 0, 0)
        self.answers_layout.setSpacing(12)
        self.answers_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.answers_scroll.setWidget(self.answers_container)
        ans_layout.addWidget(self.answers_scroll)

        splitter.addWidget(ans_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        outer.addWidget(splitter, 1)

    # ── Data loading ─────────────────────────────────────────────────
    def refresh(self):
        self._load_categories()
        self._load_answers()

    def _load_categories(self):
        self.cat_list.blockSignals(True)
        self.cat_list.clear()

        # "All" item
        all_item = QListWidgetItem("📋  All Answers")
        all_item.setData(Qt.ItemDataRole.UserRole, None)
        self.cat_list.addItem(all_item)

        categories = self.db.get_categories()
        for cat in categories:
            item = QListWidgetItem(f"{cat['icon']}  {cat['name']}")
            item.setData(Qt.ItemDataRole.UserRole, cat["id"])
            self.cat_list.addItem(item)

        # Restore selection
        self.cat_list.blockSignals(False)
        if self.current_category_id is None:
            self.cat_list.setCurrentRow(0)
        else:
            for i in range(self.cat_list.count()):
                if self.cat_list.item(i).data(Qt.ItemDataRole.UserRole) == self.current_category_id:
                    self.cat_list.setCurrentRow(i)
                    break

    def _load_answers(self, search_query=None):
        # Clear existing cards
        while self.answers_layout.count():
            item = self.answers_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if search_query:
            answers = self.db.search_quick_answers(search_query)
        else:
            answers = self.db.get_quick_answers(self.current_category_id)

        if not answers:
            empty = QLabel(
                "No answers yet.\nClick '+ Add Answer' to store your first response!"
                if not search_query else "No results found."
            )
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet(f"""
                color: {COLORS['text_muted']}; font-size: {FONT['md']}px;
                background: transparent; padding: 60px 20px;
            """)
            self.answers_layout.addWidget(empty)
            return

        # Sort: most-copied first when showing "All Answers"
        if self.current_category_id is None and not search_query:
            answers = sorted(answers, key=lambda a: a.get("times_copied", 0), reverse=True)

        for qa in answers:
            card = AnswerCard(qa, self.db,
                              on_edit=self._edit_answer,
                              on_delete=self._delete_answer,
                              on_change_category=self._change_answer_category)
            self.answers_layout.addWidget(card)

    # ── Signals ──────────────────────────────────────────────────────
    def _on_category_changed(self, row):
        if row < 0:
            return
        item = self.cat_list.item(row)
        self.current_category_id = item.data(Qt.ItemDataRole.UserRole)
        self.search_input.clear()
        self._load_answers()

    def _on_search(self, text):
        text = text.strip()
        if text:
            self._load_answers(search_query=text)
        else:
            self._load_answers()

    # ── Category context menu ────────────────────────────────────────
    def _cat_context_menu(self, pos):
        item = self.cat_list.itemAt(pos)
        if not item:
            return
        cat_id = item.data(Qt.ItemDataRole.UserRole)
        if cat_id is None:
            return  # "All" can't be edited

        menu = QMenu(self)
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
            }}
            QMenu::item:selected {{
                background-color: {COLORS['primary_light']};
                color: {COLORS['primary']};
            }}
        """)
        rename_action = menu.addAction("✏️  Rename Category")
        icon_action = menu.addAction("🎨  Change Icon")
        del_action = menu.addAction("🗑️  Delete Category")

        action = menu.exec(self.cat_list.mapToGlobal(pos))
        if action == rename_action:
            self._rename_category(cat_id)
        elif action == icon_action:
            self._change_category_icon(cat_id)
        elif action == del_action:
            self._delete_category(cat_id)

    def _rename_category(self, cat_id):
        name, ok = QInputDialog.getText(self, "Rename Category", "New name:")
        if ok and name.strip():
            self.db.update_category(cat_id, name=name.strip())
            self.refresh()

    def _change_category_icon(self, cat_id):
        icon, ok = QInputDialog.getText(
            self, "Change Icon", "Emoji icon (e.g. 🎓, 💼):", text="📁"
        )
        if ok and icon.strip():
            self.db.update_category(cat_id, icon=icon.strip())
            self.refresh()

    def _delete_category(self, cat_id):
        reply = QMessageBox.question(
            self, "Delete Category",
            "Delete this category and all its answers?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.db.delete_category(cat_id)
            self.current_category_id = None
            self.refresh()

    # ── Change answer category ───────────────────────────────────────
    def _change_answer_category(self, qa_data):
        """Let user change the category of an existing answer."""
        categories = self.db.get_categories()
        cat_names = [f"{c['icon']}  {c['name']}" for c in categories]
        cat_ids = [c['id'] for c in categories]

        current_idx = 0
        for i, cid in enumerate(cat_ids):
            if cid == qa_data.get("category_id"):
                current_idx = i
                break

        name, ok = QInputDialog.getItem(
            self, "Change Category",
            "Select new category:", cat_names, current_idx, False
        )
        if ok and name:
            idx = cat_names.index(name)
            new_cat_id = cat_ids[idx]
            self.db.update_quick_answer(qa_data["id"], category_id=new_cat_id)
            self.refresh()

    # ── Add / Edit / Delete ──────────────────────────────────────────
    def _add_answer(self):
        dlg = AnswerDialog(self.db, category_id=self.current_category_id, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.refresh()

    def _add_application(self):
        """Open add application dialog (same as in other tabs)."""
        if self.main_window:
            self.main_window.navigate_to_applications()
            self.main_window.applications_view.open_add_dialog()

    def _edit_answer(self, qa_data):
        dlg = AnswerDialog(self.db, qa_data=qa_data, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.refresh()

    def _delete_answer(self, qa_data):
        reply = QMessageBox.question(
            self, "Delete Answer",
            f"Delete the answer for:\n\n\"{qa_data.get('question', '')}\"?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.db.delete_quick_answer(qa_data["id"])
            self.refresh()

    def _add_category(self):
        dlg = CategoryDialog(self.db, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.refresh()


# ======================================================================
#  Category Dialog – Modern dialog for creating categories
# ======================================================================
class CategoryDialog(QDialog):
    """Dialog for adding a new category with improved styling."""

    def __init__(self, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("New Category")
        self.setFixedWidth(440)
        self.setFixedHeight(310)
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

        # Header
        header = QWidget()
        header.setFixedHeight(56)
        header.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {COLORS['primary']}, stop:1 #818CF8);
            border-top-left-radius: {RADIUS['lg']}px;
            border-top-right-radius: {RADIUS['lg']}px;
        """)
        h_layout = QVBoxLayout(header)
        h_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_lbl = QLabel("📁  New Category")
        title_lbl.setStyleSheet("color: white; font-size: 18px; font-weight: 700; background: transparent;")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h_layout.addWidget(title_lbl)
        layout.addWidget(header)

        # Form
        form = QWidget()
        form.setStyleSheet(f"background-color: {COLORS['surface']};")
        form_layout = QVBoxLayout(form)
        form_layout.setContentsMargins(28, 24, 28, 12)
        form_layout.setSpacing(16)

        label_style = f"""
            font-weight: 600; font-size: {FONT['sm']}px;
            color: {COLORS['text_secondary']}; text-transform: uppercase;
            letter-spacing: 0.5px; background: transparent;
        """
        input_style = f"""
            QLineEdit {{
                background-color: {COLORS['surface']};
                color: {COLORS['text']};
                border: 1.5px solid {COLORS['border']};
                border-radius: {RADIUS['sm']}px;
                padding: 10px 14px;
                font-size: {FONT['md']}px;
                min-height: 20px;
            }}
            QLineEdit:focus {{
                border-color: {COLORS['primary']};
                background-color: #FAFAFF;
            }}
            QLineEdit::placeholder {{
                color: {COLORS['text_muted']};
            }}
        """

        lbl = QLabel("CATEGORY NAME")
        lbl.setStyleSheet(label_style)
        form_layout.addWidget(lbl)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. Technical Skills")
        self.name_input.setStyleSheet(input_style)
        form_layout.addWidget(self.name_input)

        lbl = QLabel("ICON (EMOJI)")
        lbl.setStyleSheet(label_style)
        form_layout.addWidget(lbl)
        self.icon_input = QLineEdit()
        self.icon_input.setPlaceholderText("e.g. 🎓 💼 🛠️")
        self.icon_input.setText("📁")
        self.icon_input.setStyleSheet(input_style)
        form_layout.addWidget(self.icon_input)

        form_layout.addStretch()
        layout.addWidget(form, 1)

        # Buttons
        btn_bar = QWidget()
        btn_bar.setStyleSheet(f"background-color: {COLORS['surface_alt']}; border-top: 1px solid {COLORS['border']};")
        btn_layout = QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(24, 12, 24, 12)
        btn_layout.addStretch()

        cancel = QPushButton("Cancel")
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel.setStyleSheet(SECONDARY_BTN)
        cancel.setFixedHeight(38)
        cancel.clicked.connect(self.reject)
        btn_layout.addWidget(cancel)

        save = QPushButton("✚  Create")
        save.setCursor(Qt.CursorShape.PointingHandCursor)
        save.setFixedHeight(38)
        save.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white; border: none;
                border-radius: {RADIUS['md']}px;
                padding: 10px 24px;
                font-size: {FONT['md']}px; font-weight: 600;
            }}
            QPushButton:hover {{ background-color: {COLORS['primary_hover']}; }}
        """)
        save.clicked.connect(self._save)
        btn_layout.addWidget(save)

        layout.addWidget(btn_bar)

    def _save(self):
        name = self.name_input.text().strip()
        icon = self.icon_input.text().strip() or "📁"
        if not name:
            QMessageBox.warning(self, "Missing", "Category name is required.")
            return
        self.db.add_category(name, icon)
        self.accept()


# ======================================================================
#  Answer Dialog – Redesigned with better question input
# ======================================================================
class AnswerDialog(QDialog):
    """Dialog for adding or editing a quick answer."""

    def __init__(self, db: DatabaseManager, qa_data=None, category_id=None, parent=None):
        super().__init__(parent)
        self.db = db
        self.qa_data = qa_data
        self.is_edit = qa_data is not None
        self.default_category_id = category_id
        self.setWindowTitle("Edit Answer" if self.is_edit else "New Answer")
        self.setFixedWidth(620)
        self.setMinimumHeight(560)
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

        # ── Header ──
        header = QWidget()
        header.setFixedHeight(66)
        header.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {COLORS['primary']}, stop:1 #818CF8);
        """)
        h_layout = QVBoxLayout(header)
        h_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_lbl = QLabel("✏️  Edit Answer" if self.is_edit else "📝  New Answer")
        title_lbl.setStyleSheet("color: white; font-size: 20px; font-weight: 700; background: transparent;")
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
        form_layout.setContentsMargins(32, 24, 32, 24)
        form_layout.setSpacing(18)

        label_style = f"""
            font-weight: 600; font-size: {FONT['sm']}px;
            color: {COLORS['text_secondary']}; text-transform: uppercase;
            letter-spacing: 0.5px;
        """
        field_style = f"""
            QComboBox {{
                background-color: {COLORS['surface']};
                color: {COLORS['text']};
                border: 1.5px solid {COLORS['border']};
                border-radius: {RADIUS['sm']}px;
                padding: 10px 14px;
                font-size: {FONT['md']}px;
            }}
            QComboBox:focus {{
                border-color: {COLORS['primary']};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 30px; border: none;
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
                selection-background-color: {COLORS['primary_light']};
                selection-color: {COLORS['primary']};
                outline: none; padding: 4px;
            }}
        """

        # Category (with option to create new)
        lbl = QLabel("CATEGORY")
        lbl.setStyleSheet(label_style)
        form_layout.addWidget(lbl)

        cat_row = QHBoxLayout()
        self.cat_combo = QComboBox()
        self.cat_combo.setStyleSheet(field_style)
        categories = self.db.get_categories()
        for cat in categories:
            self.cat_combo.addItem(f"{cat['icon']}  {cat['name']}", cat["id"])
        # Pre-select
        if self.is_edit:
            for i in range(self.cat_combo.count()):
                if self.cat_combo.itemData(i) == self.qa_data.get("category_id"):
                    self.cat_combo.setCurrentIndex(i)
                    break
        elif self.default_category_id:
            for i in range(self.cat_combo.count()):
                if self.cat_combo.itemData(i) == self.default_category_id:
                    self.cat_combo.setCurrentIndex(i)
                    break
        cat_row.addWidget(self.cat_combo, 1)

        new_cat_btn = QPushButton("＋ New")
        new_cat_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        new_cat_btn.setFixedHeight(38)
        new_cat_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['surface_alt']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: {RADIUS['sm']}px;
                padding: 6px 14px;
                font-size: {FONT['sm']}px; font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_light']};
                color: {COLORS['primary']};
                border-color: {COLORS['primary']};
            }}
        """)
        new_cat_btn.clicked.connect(self._create_category_inline)
        cat_row.addWidget(new_cat_btn)
        form_layout.addLayout(cat_row)

        # Question – use QTextEdit so longer questions are visible
        lbl = QLabel("QUESTION / LABEL")
        lbl.setStyleSheet(label_style)
        form_layout.addWidget(lbl)
        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText("e.g. What university did you attend?")
        self.question_input.setStyleSheet(f"""
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
        self.question_input.setFixedHeight(80)
        form_layout.addWidget(self.question_input)

        # Answer
        lbl = QLabel("ANSWER")
        lbl.setStyleSheet(label_style)
        form_layout.addWidget(lbl)
        self.answer_input = QTextEdit()
        self.answer_input.setPlaceholderText("Your answer to copy-paste into applications…")
        self.answer_input.setStyleSheet(f"""
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
        self.answer_input.setMinimumHeight(180)
        form_layout.addWidget(self.answer_input)

        form_layout.addStretch()

        scroll.setWidget(form_container)
        layout.addWidget(scroll, 1)

        # Pre-fill
        if self.is_edit:
            self.question_input.setPlainText(self.qa_data.get("question", ""))
            self.answer_input.setPlainText(self.qa_data.get("answer", ""))

        # ── Buttons bar ──
        btn_bar = QWidget()
        btn_bar.setStyleSheet(f"background-color: {COLORS['surface_alt']}; border-top: 1px solid {COLORS['border']};")
        btn_layout = QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(24, 14, 24, 14)
        btn_layout.addStretch()

        cancel = QPushButton("Cancel")
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel.setStyleSheet(SECONDARY_BTN)
        cancel.setFixedHeight(40)
        cancel.clicked.connect(self.reject)
        btn_layout.addWidget(cancel)

        save = QPushButton("💾  Save" if self.is_edit else "✚  Add Answer")
        save.setCursor(Qt.CursorShape.PointingHandCursor)
        save.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white; border: none;
                border-radius: {RADIUS['md']}px;
                padding: 10px 28px;
                font-size: {FONT['md']}px; font-weight: 600;
            }}
            QPushButton:hover {{ background-color: {COLORS['primary_hover']}; }}
        """)
        save.setFixedHeight(40)
        save.clicked.connect(self._save)
        btn_layout.addWidget(save)

        layout.addWidget(btn_bar)

    def _create_category_inline(self):
        """Create a new category from within the answer dialog."""
        name, ok = QInputDialog.getText(self, "New Category", "Category name:")
        if ok and name.strip():
            icon, ok2 = QInputDialog.getText(
                self, "Category Icon", "Emoji icon:", text="📁"
            )
            if ok2:
                new_id = self.db.add_category(name.strip(), icon.strip() or "📁")
                if new_id:
                    # Refresh combo
                    self.cat_combo.clear()
                    categories = self.db.get_categories()
                    for cat in categories:
                        self.cat_combo.addItem(f"{cat['icon']}  {cat['name']}", cat["id"])
                    # Select the new one
                    for i in range(self.cat_combo.count()):
                        if self.cat_combo.itemData(i) == new_id:
                            self.cat_combo.setCurrentIndex(i)
                            break

    def _save(self):
        question = self.question_input.toPlainText().strip()
        answer = self.answer_input.toPlainText().strip()
        cat_id = self.cat_combo.currentData()

        if not question or not answer:
            QMessageBox.warning(self, "Missing Fields",
                                "Both question and answer are required.")
            return

        if self.is_edit:
            self.db.update_quick_answer(
                self.qa_data["id"],
                category_id=cat_id, question=question, answer=answer,
            )
        else:
            self.db.add_quick_answer(cat_id, question, answer)

        self.accept()
