"""
Filter Panel UI
Provides UI components for filtering job applications with modern, engaging design.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                           QPushButton, QLineEdit, QComboBox, QDateEdit,
                           QFrame, QGroupBox, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, QDate, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor
from datetime import datetime, timedelta

from models.application import Application
from utils.date_helpers import get_date_range
import assets.styles as styles

class ModernCard(QFrame):
    """A modern card widget with subtle shadow and hover effects"""
    
    def __init__(self, title="", icon_text="📁"):
        super().__init__()
        self.setFrameStyle(QFrame.NoFrame)
        self.setup_card(title, icon_text)
        
    def setup_card(self, title, icon_text):
        """Setup the card with title and content area"""
        self.setStyleSheet("""
            ModernCard {
                background-color: white;
                border-radius: 12px;
                border: 1px solid rgba(0, 0, 0, 0.08);
            }
            ModernCard:hover {
                border: 1px solid rgba(108, 92, 231, 0.3);
                background-color: rgba(248, 249, 250, 0.8);
            }
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(8)
        
        if title:
            # Header with icon and title
            header_layout = QHBoxLayout()
            header_layout.setContentsMargins(0, 0, 0, 0)
            
            # Icon
            icon_label = QLabel(icon_text)
            icon_label.setStyleSheet("""
                font-size: 16px;
                color: #6c5ce7;
                font-weight: bold;
                margin-right: 8px;
            """)
            
            # Title
            title_label = QLabel(title)
            title_label.setStyleSheet("""
                font-size: 14px;
                font-weight: 600;
                color: #2d3748;
                margin: 0;
            """)
            
            header_layout.addWidget(icon_label)
            header_layout.addWidget(title_label)
            header_layout.addStretch()
            
            layout.addLayout(header_layout)
            
            # Separator line
            separator = QFrame()
            separator.setFrameStyle(QFrame.HLine)
            separator.setStyleSheet("""
                QFrame {
                    color: rgba(0, 0, 0, 0.1);
                    margin: 4px 0px;
                }
            """)
            layout.addWidget(separator)
        
        # Content area
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(12)
        layout.addLayout(self.content_layout)

class ModernPillButton(QPushButton):
    """Modern pill-shaped button for quick filters"""
    
    def __init__(self, text, is_primary=False):
        super().__init__(text)
        self.is_primary = is_primary
        self.is_active = False
        self.setup_style()
        
    def setup_style(self):
        """Setup the modern pill button styling"""
        if self.is_primary:
            self.setStyleSheet("""
                ModernPillButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                              stop:0 #6c5ce7, stop:1 #a29bfe);
                    color: white;
                    border: none;
                    border-radius: 20px;
                    padding: 12px 24px;
                    font-weight: 600;
                    font-size: 13px;
                    min-width: 120px;
                }
                ModernPillButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                              stop:0 #5641e5, stop:1 #9c88ff);
                }
                ModernPillButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                              stop:0 #4834d4, stop:1 #8c7ae6);
                }
            """)
        else:
            self.setStyleSheet("""
                ModernPillButton {
                    background-color: #f8f9fa;
                    color: #495057;
                    border: 1px solid #e9ecef;
                    border-radius: 18px;
                    padding: 10px 20px;
                    font-weight: 500;
                    font-size: 12px;
                    min-width: 100px;
                }
                ModernPillButton:hover {
                    background-color: #e9ecef;
                    border-color: #6c5ce7;
                    color: #6c5ce7;
                }
                ModernPillButton:pressed {
                    background-color: #dee2e6;
                }
            """)

class FilterPanel(QWidget):
    """Modern filter panel for filtering job applications"""
    
    # Signal emitted when filters are applied
    filters_applied = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the modern UI components"""
        # Set size policy and minimum width to prevent the panel from being cut off
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setMinimumWidth(300)
        self.setMaximumWidth(350)
        
        # Main container with scroll area
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scroll area for better space management
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f8f9fa;
            }
            QScrollBar:vertical {
                border: none;
                background-color: rgba(0, 0, 0, 0.1);
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: rgba(108, 92, 231, 0.5);
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: rgba(108, 92, 231, 0.7);
            }
        """)
        
        # Content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(16)
        
        # Set background color for the entire panel
        self.setStyleSheet("""
            FilterPanel {
                background-color: #f8f9fa;
            }
        """)
        
        # Modern header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(8)
        
        # Title with icon
        title_layout = QHBoxLayout()
        title_icon = QLabel("🎯")
        title_icon.setStyleSheet("font-size: 20px;")
        
        title_label = QLabel("Filters")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: 700;
            color: #4a5568;
            margin: 0;
        """)
        
        title_layout.addWidget(title_icon)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        # Subtitle
        subtitle_label = QLabel("Refine your search results")
        subtitle_label.setStyleSheet("""
            font-size: 13px;
            color: #718096;
            margin: 0;
            margin-bottom: 8px;
        """)
        
        header_layout.addLayout(title_layout)
        header_layout.addWidget(subtitle_label)
        content_layout.addLayout(header_layout)
        
        # Company filter card
        company_card = ModernCard("Company", "🏢")
        self.company_input = QLineEdit()
        self.company_input.setPlaceholderText("Search companies...")
        self.company_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 12px 16px;
                background-color: white;
                color: #2d3748;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #6c5ce7;
                outline: none;
            }
            QLineEdit::placeholder {
                color: #a0aec0;
            }
        """)
        company_card.content_layout.addWidget(self.company_input)
        content_layout.addWidget(company_card)
        
        # Time frame card
        time_card = ModernCard("Time Frame", "📅")
        
        # Quick filter buttons in a grid
        quick_buttons_layout = QVBoxLayout()
        quick_buttons_layout.setSpacing(8)
        
        # First row
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(8)
        self.past_7_days_btn = ModernPillButton("Past 7 days")
        self.past_30_days_btn = ModernPillButton("Past 30 days")
        row1_layout.addWidget(self.past_7_days_btn)
        row1_layout.addWidget(self.past_30_days_btn)
        
        # Second row
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(8)
        self.past_90_days_btn = ModernPillButton("Past 90 days")
        self.all_time_btn = ModernPillButton("All time")
        row2_layout.addWidget(self.past_90_days_btn)
        row2_layout.addWidget(self.all_time_btn)
        
        quick_buttons_layout.addLayout(row1_layout)
        quick_buttons_layout.addLayout(row2_layout)
        
        time_card.content_layout.addLayout(quick_buttons_layout)
        content_layout.addWidget(time_card)
        
        # Status filter card
        status_card = ModernCard("Status", "📊")
        self.status_combo = QComboBox()
        self.status_combo.addItem("All Statuses")
        for status in Application.VALID_STATUSES:
            self.status_combo.addItem(status)
            
        self.status_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 12px 16px;
                background-color: white;
                color: #2d3748;
                font-size: 14px;
                min-height: 20px;
            }
            QComboBox:focus {
                border: 2px solid #6c5ce7;
                outline: none;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 30px;
                border-left: 1px solid #e2e8f0;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
                background: #f7fafc;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #a0aec0;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background-color: white;
                selection-background-color: #6c5ce7;
                selection-color: white;
            }
        """)
        status_card.content_layout.addWidget(self.status_combo)
        content_layout.addWidget(status_card)
        
        # Date range card
        date_card = ModernCard("Date Range", "📆")
        
        # From date
        from_layout = QVBoxLayout()
        from_layout.setSpacing(6)
        from_label = QLabel("From:")
        from_label.setStyleSheet("""
            font-size: 12px;
            font-weight: 600;
            color: #4a5568;
            margin: 0;
        """)
        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDate(QDate.currentDate().addDays(-30))
        self.date_from.setStyleSheet("""
            QDateEdit {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 12px 16px;
                background-color: white;
                color: #2d3748;
                font-size: 14px;
            }
            QDateEdit:focus {
                border: 2px solid #6c5ce7;
                outline: none;
            }
            QDateEdit::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 30px;
                border-left: 1px solid #e2e8f0;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
                background: #f7fafc;
            }
        """)
        
        from_layout.addWidget(from_label)
        from_layout.addWidget(self.date_from)
        
        # To date
        to_layout = QVBoxLayout()
        to_layout.setSpacing(6)
        to_label = QLabel("To:")
        to_label.setStyleSheet("""
            font-size: 12px;
            font-weight: 600;
            color: #4a5568;
            margin: 0;
        """)
        self.date_to = QDateEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setDate(QDate.currentDate())
        self.date_to.setStyleSheet(self.date_from.styleSheet())
        
        to_layout.addWidget(to_label)
        to_layout.addWidget(self.date_to)
        
        # Date layout
        date_row_layout = QHBoxLayout()
        date_row_layout.setSpacing(12)
        date_row_layout.addLayout(from_layout)
        date_row_layout.addLayout(to_layout)
        
        date_card.content_layout.addLayout(date_row_layout)
        content_layout.addWidget(date_card)
        
        # Action buttons
        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(8)
        
        # Apply filters button (primary)
        self.apply_button = ModernPillButton("Apply Filters", is_primary=True)
        self.apply_button.setStyleSheet("""
            ModernPillButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #6c5ce7, stop:1 #a29bfe);
                color: white;
                border: none;
                border-radius: 25px;
                padding: 16px 32px;
                font-weight: 700;
                font-size: 14px;
                min-width: 200px;
            }
            ModernPillButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #5641e5, stop:1 #9c88ff);
            }
            ModernPillButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #4834d4, stop:1 #8c7ae6);
            }
        """)
        
        # Reset filters button
        self.reset_button = QPushButton("Reset All")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #6c5ce7;
                border: 2px solid #6c5ce7;
                border-radius: 20px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 13px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: rgba(108, 92, 231, 0.1);
            }
            QPushButton:pressed {
                background-color: rgba(108, 92, 231, 0.2);
            }
        """)
        
        actions_layout.addWidget(self.apply_button)
        actions_layout.addWidget(self.reset_button)
        
        content_layout.addLayout(actions_layout)
        content_layout.addStretch()
        
        # Set content widget to scroll area
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        
        # Connect signals
        self.apply_button.clicked.connect(self.apply_filters)
        self.reset_button.clicked.connect(self.reset_filters)
        self.past_7_days_btn.clicked.connect(lambda: self.set_date_range_and_apply(7))
        self.past_30_days_btn.clicked.connect(lambda: self.set_date_range_and_apply(30))
        self.past_90_days_btn.clicked.connect(lambda: self.set_date_range_and_apply(90))
        self.all_time_btn.clicked.connect(lambda: self.set_date_range_and_apply(365*10))  # 10 years
        
        # Auto-apply on input changes
        self.company_input.textChanged.connect(self.auto_apply_filters)
        self.status_combo.currentTextChanged.connect(self.auto_apply_filters)
        self.date_from.dateChanged.connect(self.auto_apply_filters)
        self.date_to.dateChanged.connect(self.auto_apply_filters)
        
    def apply_filters(self):
        """Apply the current filters and emit signal"""
        filters = {}
        
        # Company filter
        company = self.company_input.text().strip()
        if company:
            filters['company'] = company
            
        # Status filter
        status = self.status_combo.currentText()
        if status != "All Statuses":
            filters['status'] = status
            
        # Date range
        filters['date_from'] = self.date_from.date().toString("yyyy-MM-dd")
        filters['date_to'] = self.date_to.date().toString("yyyy-MM-dd")
        
        # Emit signal with filters
        self.filters_applied.emit(filters)
        
    def reset_filters(self):
        """Reset all filters to default values"""
        self.company_input.clear()
        self.status_combo.setCurrentIndex(0)  # "All Statuses"
        
        # Reset date range to last 30 days
        self.date_from.setDate(QDate.currentDate().addDays(-30))
        self.date_to.setDate(QDate.currentDate())
        
        # Apply the reset filters
        self.apply_filters()
        
    def set_date_range(self, days):
        """Set the date range to the specified number of days"""
        if days >= 365*10:  # For "All time"
            # Set a very early date for "all time"
            self.date_from.setDate(QDate(2000, 1, 1))
        else:
            self.date_from.setDate(QDate.currentDate().addDays(-days))
        self.date_to.setDate(QDate.currentDate())
        
    def set_date_range_and_apply(self, days):
        """Set the date range and apply filters"""
        self.set_date_range(days)
        
        # Update button states to show which one is active
        self.update_time_button_states(days)
        
        self.apply_filters()
        
    def update_time_button_states(self, active_days):
        """Update visual state of time filter buttons"""
        buttons = [
            (self.past_7_days_btn, 7),
            (self.past_30_days_btn, 30),
            (self.past_90_days_btn, 90),
            (self.all_time_btn, 365*10)
        ]
        
        for button, days in buttons:
            if days == active_days:
                # Active button style
                button.setStyleSheet("""
                    ModernPillButton {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                                  stop:0 #6c5ce7, stop:1 #a29bfe);
                        color: white;
                        border: none;
                        border-radius: 18px;
                        padding: 10px 20px;
                        font-weight: 600;
                        font-size: 12px;
                        min-width: 100px;
                    }
                    ModernPillButton:hover {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                                  stop:0 #5641e5, stop:1 #9c88ff);
                    }
                """)
            else:
                # Inactive button style
                button.setStyleSheet("""
                    ModernPillButton {
                        background-color: #f8f9fa;
                        color: #495057;
                        border: 1px solid #e9ecef;
                        border-radius: 18px;
                        padding: 10px 20px;
                        font-weight: 500;
                        font-size: 12px;
                        min-width: 100px;
                    }
                    ModernPillButton:hover {
                        background-color: #e9ecef;
                        border-color: #6c5ce7;
                        color: #6c5ce7;
                    }
                    ModernPillButton:pressed {
                        background-color: #dee2e6;
                    }
                """)
        
    def auto_apply_filters(self):
        """Auto-apply filters when input changes"""
        # Reset time button states when manual date changes are made
        self.update_time_button_states(None)
        self.apply_filters() 