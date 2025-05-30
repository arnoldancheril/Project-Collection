"""
Add Application Dialog
Dialog for adding or editing job applications.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                           QLineEdit, QDateEdit, QComboBox, QPushButton,
                           QTextEdit, QFormLayout, QDialogButtonBox, QFrame,
                           QWidget, QSizePolicy, QCompleter)
from PyQt5.QtCore import Qt, QDate, QSize, QStringListModel
from PyQt5.QtGui import QColor, QLinearGradient, QBrush, QPalette
from datetime import datetime
from models.application import Application
import assets.styles as styles

class AddApplicationDialog(QDialog):
    """Dialog for adding or editing job applications"""
    
    # Predefined software engineering roles for easy selection
    PREDEFINED_ROLES = [
        "Software Engineer",
        "Senior Software Engineer", 
        "Software Developer",
        "Full Stack Developer",
        "Frontend Developer",
        "Backend Developer",
        "Junior Software Engineer",
        "Senior Software Developer",
        "Python Developer",
        "JavaScript Developer",
        "React Developer",
        "Node.js Developer",
        "DevOps Engineer",
        "Data Engineer",
        "Machine Learning Engineer",
        "Site Reliability Engineer",
        "Principal Software Engineer",
        "Staff Software Engineer",
        "Technical Lead",
        "Engineering Manager",
        "Architect",
        "Mobile Developer",
        "iOS Developer",
        "Android Developer"
    ]
    
    def __init__(self, parent=None, application=None):
        """
        Initialize the add/edit application dialog
        
        Args:
            parent: Parent widget
            application: Application object for editing (None for new application)
        """
        super().__init__(parent)
        self.application = application
        self.is_edit_mode = application is not None
        self.db_manager = parent.db_manager if parent else None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI"""
        # Set window properties
        title = "Edit Application" if self.is_edit_mode else "Add Application"
        self.setWindowTitle(title)
        self.setMinimumWidth(450)
        self.setMinimumHeight(550)
        
        # Remove window frame if on supported platform
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        
        # Main layout without margins
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create header with gradient
        header = QWidget()
        header.setFixedHeight(80)
        header.setAutoFillBackground(True)
        
        # Create gradient for header
        gradient = QLinearGradient(0, 0, header.width(), 0)
        gradient.setColorAt(0, QColor("#6c5ce7"))
        gradient.setColorAt(1, QColor("#4834d4"))
        
        # Apply gradient to header
        palette = header.palette()
        palette.setBrush(QPalette.Window, QBrush(gradient))
        header.setPalette(palette)
        
        # Header layout
        header_layout = QVBoxLayout(header)
        header_layout.setAlignment(Qt.AlignCenter)
        
        # Title label
        header_title = QLabel(title)
        header_title.setStyleSheet("""
            color: white;
            font-size: 22px;
            font-weight: bold;
        """)
        header_title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(header_title)
        
        # Add header to main layout
        main_layout.addWidget(header)
        
        # Content widget with white background
        content_widget = QWidget()
        content_widget.setStyleSheet("""
            background-color: white;
            border-radius: 5px;
            color: #333333;
            
            QLabel {
                color: #333333;
            }
            
            QLineEdit, QDateEdit, QComboBox, QTextEdit {
                color: #333333;
            }
        """)
        
        # Content layout with proper margins
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(25, 25, 25, 25)
        content_layout.setSpacing(15)
        
        # Create form fields
        
        # Company
        company_label = QLabel("Company")
        company_label.setStyleSheet("font-weight: bold; color: #333;")
        
        # Get company suggestions and create auto-complete combo
        company_suggestions = self.get_company_suggestions()
        current_company = self.application.company if self.is_edit_mode else ""
        self.company_input = self.create_autocomplete_combo(
            company_suggestions, 
            "Start typing company name...", 
            current_company
        )
        
        # Role
        role_label = QLabel("Role")
        role_label.setStyleSheet("font-weight: bold; color: #333;")
        
        # Get role suggestions and create auto-complete combo
        role_suggestions = self.get_role_suggestions()
        current_role = self.application.role if self.is_edit_mode else ""
        self.role_input = self.create_autocomplete_combo(
            role_suggestions, 
            "Select or type role...", 
            current_role
        )
        
        # Date Applied
        date_label = QLabel("Date Applied")
        date_label.setStyleSheet("font-weight: bold; color: #333;")
        self.date_input = QDateEdit()
        self.date_input.setCalendarPopup(True)
        self.date_input.setDate(QDate.currentDate())
        self.date_input.setStyleSheet("""
            QDateEdit {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background: white;
                color: #333333;
                selection-background-color: #6c5ce7;
            }
            QDateEdit:focus {
                border: 1px solid #6c5ce7;
            }
            QDateEdit::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 25px;
                border-left: 1px solid #ddd;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
        """)
        if self.is_edit_mode and self.application.date_applied:
            try:
                date = datetime.strptime(self.application.date_applied, "%Y-%m-%d").date()
                self.date_input.setDate(QDate(date.year, date.month, date.day))
            except ValueError:
                pass
        
        # Status
        status_label = QLabel("Status")
        status_label.setStyleSheet("font-weight: bold; color: #333;")
        self.status_combo = QComboBox()
        self.status_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background: white;
                color: #333333;
                selection-background-color: #6c5ce7;
            }
            QComboBox:focus {
                border: 1px solid #6c5ce7;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 25px;
                border-left: 1px solid #ddd;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox QAbstractItemView {
                color: #333333;
                background-color: white;
                selection-background-color: #6c5ce7;
            }
        """)
        for status in Application.VALID_STATUSES:
            self.status_combo.addItem(status)
            
        if self.is_edit_mode:
            index = self.status_combo.findText(self.application.status)
            if index >= 0:
                self.status_combo.setCurrentIndex(index)
        
        # Notes
        notes_label = QLabel("Notes")
        notes_label.setStyleSheet("font-weight: bold; color: #333;")
        self.notes_input = QTextEdit()
        self.notes_input.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background: white;
                color: #333333;
                selection-background-color: #6c5ce7;
            }
            QTextEdit:focus {
                border: 1px solid #6c5ce7;
            }
        """)
        self.notes_input.setMinimumHeight(120)
        if self.is_edit_mode and self.application.notes:
            self.notes_input.setText(self.application.notes)
        
        # Add fields to layout
        content_layout.addWidget(company_label)
        content_layout.addWidget(self.company_input)
        content_layout.addWidget(role_label)
        content_layout.addWidget(self.role_input)
        content_layout.addWidget(date_label)
        content_layout.addWidget(self.date_input)
        content_layout.addWidget(status_label)
        content_layout.addWidget(self.status_combo)
        content_layout.addWidget(notes_label)
        content_layout.addWidget(self.notes_input)
        
        # Add spacer
        content_layout.addStretch()
        
        # Buttons at the bottom
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 15, 0, 0)
        
        # Add spacer to push buttons to the right
        button_layout.addStretch()
        
        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        cancel_button.clicked.connect(self.reject)
        
        # Save/Add button
        save_text = "Save" if self.is_edit_mode else "Add"
        save_button = QPushButton(save_text)
        save_button.setStyleSheet("""
            QPushButton {
                background-color: #1abc9c;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #16a085;
            }
        """)
        save_button.clicked.connect(self.accept)
        
        # Add buttons to layout
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(save_button)
        
        # Add button layout to content
        content_layout.addLayout(button_layout)
        
        # Add content widget to main layout
        main_layout.addWidget(content_widget)
        
    def accept(self):
        """Handle the accept action (Save button)"""
        # Get values from form
        company = self.company_input.currentText().strip()
        role = self.role_input.currentText().strip()
        date_applied = self.date_input.date().toString("yyyy-MM-dd")
        status = self.status_combo.currentText()
        notes = self.notes_input.toPlainText().strip()
        
        # Validate
        if not company or not role:
            return  # Could show an error message here
            
        # Save to database
        if self.db_manager:
            if self.is_edit_mode:
                # Update existing application
                self.db_manager.update_application(
                    self.application.id,
                    company=company,
                    role=role,
                    date_applied=date_applied,
                    status=status,
                    notes=notes
                )
            else:
                # Add new application
                self.db_manager.add_application(
                    company=company,
                    role=role,
                    date_applied=date_applied,
                    status=status
                )
                
        # Close dialog
        super().accept()
        
    def create_autocomplete_combo(self, items, placeholder_text="", current_value=""):
        """Create an editable combobox with auto-complete functionality"""
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        
        # Add an empty option first for easier clearing
        if not current_value:
            combo.addItem("")
        
        # Add items to combo
        combo.addItems(items)
        
        # Set up auto-complete
        completer = QCompleter(items)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        combo.setCompleter(completer)
        
        # Set placeholder if provided
        if placeholder_text:
            combo.lineEdit().setPlaceholderText(placeholder_text)
        
        # Set current value if provided
        if current_value:
            combo.setCurrentText(current_value)
        elif not current_value and combo.count() > 0:
            combo.setCurrentIndex(0)  # Select empty option
        
        # Make the dropdown show more items
        combo.setMaxVisibleItems(15)
        
        # Style the combobox
        combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background: white;
                color: #333333;
                selection-background-color: #6c5ce7;
                min-height: 20px;
            }
            QComboBox:focus {
                border: 1px solid #6c5ce7;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 25px;
                border-left: 1px solid #ddd;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
                background: #f8f9fa;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #666;
                margin: 0px;
            }
            QComboBox QAbstractItemView {
                color: #333333;
                background-color: white;
                selection-background-color: #6c5ce7;
                selection-color: white;
                border: 1px solid #ddd;
                outline: none;
            }
            QComboBox QLineEdit {
                color: #333333;
                background: white;
                border: none;
                padding: 0px;
                selection-background-color: #6c5ce7;
            }
        """)
        
        return combo
    
    def get_company_suggestions(self):
        """Get company suggestions from database and return as list"""
        if self.db_manager:
            # Get frequently used companies first
            frequent_companies = self.db_manager.get_frequent_companies(20)
            # Get all companies to ensure completeness
            all_companies = self.db_manager.get_all_companies()
            
            # Combine and deduplicate while preserving order (frequent first)
            seen = set()
            companies = []
            for company in frequent_companies + all_companies:
                if company.lower() not in seen:
                    companies.append(company)
                    seen.add(company.lower())
            
            return companies
        return []
    
    def get_role_suggestions(self):
        """Get role suggestions combining predefined roles and database roles"""
        suggestions = list(self.PREDEFINED_ROLES)  # Start with predefined roles
        
        if self.db_manager:
            # Get frequently used roles from database
            frequent_roles = self.db_manager.get_frequent_roles(15)
            # Get all roles from database
            all_roles = self.db_manager.get_all_roles()
            
            # Add database roles, avoiding duplicates
            seen = {role.lower() for role in suggestions}
            for role in frequent_roles + all_roles:
                if role.lower() not in seen:
                    suggestions.append(role)
                    seen.add(role.lower())
        
        return suggestions 