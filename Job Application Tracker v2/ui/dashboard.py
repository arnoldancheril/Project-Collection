"""
Dashboard UI
Main dashboard interface for the application tracker.
Displays status summary cards and application list.
"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                           QHeaderView, QFrame, QLineEdit, QComboBox, QDateEdit,
                           QMenu, QAction, QSizePolicy, QTabWidget)
from PyQt5.QtCore import Qt, QSize, QDate
from PyQt5.QtGui import QIcon, QColor, QLinearGradient, QBrush, QFont, QPalette

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Qt5Agg')

from database.db_manager import DatabaseManager
from models.application import Application
from ui.add_application import AddApplicationDialog
from ui.analytics import AnalyticsWidget
from ui.filter_panel import FilterPanel
from utils.date_helpers import format_date
import assets.styles as styles

class StatusCard(QFrame):
    """Card widget displaying a status count"""
    
    def __init__(self, title, count, color, parent=None):
        super().__init__(parent)
        self.title = title
        self.count = count
        self.color = color
        self.setup_ui()
        
    def setup_ui(self):
        # Set card style with proper background and text colors
        self.setObjectName("statusCard")
        self.setStyleSheet(f"""
            #statusCard {{
                background-color: {self.color};
                border-radius: 15px;
                min-height: 120px;
                max-height: 120px;
                border: 2px solid rgba(255, 255, 255, 0.3);
                margin: 5px;
            }}
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setAlignment(Qt.AlignCenter)
        
        # Status label with explicit white color and increased font size
        status_label = QLabel(self.title)
        status_label.setAlignment(Qt.AlignCenter)
        status_font = QFont("Arial", 18, QFont.Bold)  # Increased from 14 to 18
        status_label.setFont(status_font)
        status_label.setStyleSheet("""
            color: white;
            background-color: transparent;
            font-weight: bold;
            margin-bottom: 10px;
        """)
        
        # Count label with explicit white color and reduced font size
        count_label = QLabel(str(self.count))
        count_label.setAlignment(Qt.AlignCenter)
        count_font = QFont("Arial", 28, QFont.Bold)  # Reduced from 32 to 28
        count_label.setFont(count_font)
        count_label.setStyleSheet("""
            color: white;
            background-color: transparent;
            font-weight: bold;
        """)
        
        # Add to layout
        layout.addWidget(status_label)
        layout.addWidget(count_label)
        
class ChartWidget(QWidget):
    """Widget containing analytics charts"""
    
    def __init__(self, applications, parent=None):
        super().__init__(parent)
        self.applications = applications
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the analytics charts"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Title with explicit styling
        title = QLabel("📊 Analytics Dashboard")
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #2d3748;
            background-color: transparent;
            margin-bottom: 10px;
        """)
        layout.addWidget(title)
        
        # Subtitle with explicit styling
        subtitle = QLabel("Pipeline health and application insights")
        subtitle.setStyleSheet("""
            font-size: 14px;
            color: #718096;
            background-color: transparent;
            margin-bottom: 20px;
        """)
        layout.addWidget(subtitle)
        
        # Charts container
        charts_layout = QHBoxLayout()
        
        # Status distribution pie chart
        self.pie_chart = self.create_status_pie_chart()
        charts_layout.addWidget(self.pie_chart)
        
        # Applications over time bar chart
        self.timeline_chart = self.create_timeline_chart()
        charts_layout.addWidget(self.timeline_chart)
        
        layout.addLayout(charts_layout)
        
        # Stats summary
        stats_layout = self.create_stats_summary()
        layout.addLayout(stats_layout)
        
    def create_status_pie_chart(self):
        """Create a pie chart showing status distribution"""
        # Count applications by status
        status_counts = {}
        for app in self.applications:
            status = app.get('status', 'Applied')
            status_counts[status] = status_counts.get(status, 0) + 1
            
        # Create figure
        fig = Figure(figsize=(6, 5), dpi=100, facecolor='white')
        canvas = FigureCanvas(fig)
        
        if status_counts:
            # Use app consistent colors
            colors = [Application.STATUS_COLORS.get(status, '#6c5ce7') for status in status_counts.keys()]
            
            ax = fig.add_subplot(111)
            wedges, texts, autotexts = ax.pie(
                status_counts.values(), 
                labels=status_counts.keys(),
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 11, 'color': '#2d3748', 'weight': 'bold'}
            )
            
            # Style the percentage text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
                autotext.set_fontsize(10)
            
            ax.set_title('Application Status Distribution', fontsize=14, fontweight='bold', color='#2d3748', pad=20)
        else:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No applications to display', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, color='#718096')
            ax.set_title('Application Status Distribution', fontsize=14, fontweight='bold', color='#2d3748')
            
        fig.tight_layout()
        return canvas
        
    def create_timeline_chart(self):
        """Create a bar chart showing applications over time"""
        from datetime import datetime, timedelta
        from collections import defaultdict
        
        # Group applications by month
        monthly_counts = defaultdict(int)
        
        for app in self.applications:
            try:
                date_applied = datetime.strptime(app.get('date_applied', ''), '%Y-%m-%d')
                month_key = date_applied.strftime('%Y-%m')
                monthly_counts[month_key] += 1
            except (ValueError, TypeError):
                continue
                
        # Create figure
        fig = Figure(figsize=(6, 5), dpi=100, facecolor='white')
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        if monthly_counts:
            months = sorted(monthly_counts.keys())
            counts = [monthly_counts[month] for month in months]
            
            # Format month labels
            month_labels = [datetime.strptime(month, '%Y-%m').strftime('%b %Y') for month in months]
            
            bars = ax.bar(month_labels, counts, color='#6c5ce7', alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{int(height)}', ha='center', va='bottom', 
                       fontweight='bold', color='#2d3748')
            
            ax.set_title('Applications Over Time', fontsize=14, fontweight='bold', color='#2d3748', pad=20)
            ax.set_ylabel('Number of Applications', fontweight='bold', color='#4a5568')
            ax.set_xlabel('Month', fontweight='bold', color='#4a5568')
            
            # Rotate x labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
        else:
            ax.text(0.5, 0.5, 'No applications to display', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, color='#718096')
            ax.set_title('Applications Over Time', fontsize=14, fontweight='bold', color='#2d3748')
            
        # Style the axes
        ax.tick_params(colors='#4a5568')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e2e8f0')
        ax.spines['bottom'].set_color('#e2e8f0')
        ax.grid(True, alpha=0.3, color='#e2e8f0')
        
        fig.tight_layout()
        return canvas
        
    def create_stats_summary(self):
        """Create summary statistics"""
        layout = QHBoxLayout()
        layout.setSpacing(20)
        
        # Calculate stats
        total_apps = len(self.applications)
        if total_apps > 0:
            status_counts = {}
            for app in self.applications:
                status = app.get('status', 'Applied')
                status_counts[status] = status_counts.get(status, 0) + 1
                
            response_rate = ((status_counts.get('Interviewing', 0) + 
                            status_counts.get('Offer', 0)) / total_apps * 100) if total_apps > 0 else 0
            
            offer_rate = (status_counts.get('Offer', 0) / total_apps * 100) if total_apps > 0 else 0
        else:
            response_rate = 0
            offer_rate = 0
            
        # Create stat cards
        stats = [
            ("Total Applications", str(total_apps), "#6c5ce7"),
            ("Response Rate", f"{response_rate:.1f}%", "#2ecc71"),
            ("Offer Rate", f"{offer_rate:.1f}%", "#e74c3c")
        ]
        
        for title, value, color in stats:
            stat_card = QFrame()
            stat_card.setStyleSheet(f"""
                QFrame {{
                    background-color: white;
                    border-radius: 12px;
                    border: 1px solid #e2e8f0;
                    padding: 15px;
                }}
                QFrame:hover {{
                    border-color: {color};
                    background-color: #f7fafc;
                }}
            """)
            
            card_layout = QVBoxLayout(stat_card)
            card_layout.setAlignment(Qt.AlignCenter)
            
            value_label = QLabel(value)
            value_label.setStyleSheet(f"""
                font-size: 24px;
                font-weight: bold;
                color: {color};
                background-color: transparent;
                margin: 0;
            """)
            value_label.setAlignment(Qt.AlignCenter)
            
            title_label = QLabel(title)
            title_label.setStyleSheet("""
                font-size: 12px;
                font-weight: 600;
                color: #718096;
                background-color: transparent;
                margin: 0;
            """)
            title_label.setAlignment(Qt.AlignCenter)
            
            card_layout.addWidget(value_label)
            card_layout.addWidget(title_label)
            
            layout.addWidget(stat_card)
            
        return layout
        
class Dashboard(QMainWindow):
    """Main dashboard window for the application tracker"""
    
    def __init__(self, db_manager=None):
        super().__init__()
        self.db_manager = db_manager or DatabaseManager()
        self.applications = []
        
        # Set main window background to white and application-wide style
        self.setStyleSheet("""
            QMainWindow {
                background-color: white;
            }
            QWidget {
                background-color: white;
                color: #333333;
            }
            QLabel {
                background-color: transparent;
                color: #2d3748;
            }
        """)
        
        self.setup_ui()
        self.load_applications()
        
    def setup_ui(self):
        # Set window properties
        self.setWindowTitle("Application Tracker")
        self.setMinimumSize(1400, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #f8f9fa;")
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create header
        header_layout = self.create_header()
        main_layout.addLayout(header_layout)
        
        # Create tab widget with modern styling
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid #e2e8f0;
                border-radius: 0px 8px 8px 8px;
                background-color: white;
                margin-top: 0px;
                padding: 0px;
            }}
            QTabBar {{
                qproperty-drawBase: 0;
                background-color: transparent;
            }}
            QTabBar::tab {{
                background-color: #f8f9fa;
                color: #495057;
                padding: 12px 24px;
                margin-right: 2px;
                margin-bottom: 0px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 600;
                font-size: 14px;
                min-width: 120px;
                border: 1px solid #e2e8f0;
                border-bottom: none;
            }}
            QTabBar::tab:selected {{
                background-color: white;
                color: {styles.COLORS['primary']};
                border: 1px solid #e2e8f0;
                border-bottom: 1px solid white;
                margin-bottom: -1px;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: #e9ecef;
                color: {styles.COLORS['primary']};
                border-color: {styles.COLORS['primary_light']};
            }}
            QTabBar::tab:first {{
                margin-left: 8px;
            }}
        """)
        
        # Dashboard tab
        dashboard_tab = QWidget()
        dashboard_tab.setStyleSheet("background-color: white;")
        dashboard_layout = QVBoxLayout(dashboard_tab)
        dashboard_layout.setContentsMargins(15, 15, 15, 0)
        dashboard_layout.setSpacing(15)
        
        # Create status cards for dashboard tab
        status_cards_layout = self.create_status_cards()
        dashboard_layout.addLayout(status_cards_layout)
        
        # Create content area with filter rail and table
        content_layout = QHBoxLayout()
        content_layout.setSpacing(0)  # Remove spacing for cleaner look
        
        # Modern filter panel
        self.filter_panel = FilterPanel(self)
        self.filter_panel.filters_applied.connect(self.apply_filters)
        content_layout.addWidget(self.filter_panel)
        
        # Main content area with table
        main_content = QWidget()
        main_content.setStyleSheet("background-color: white; padding-left: 20px;")
        main_content_layout = QVBoxLayout(main_content)
        main_content_layout.setContentsMargins(20, 15, 20, 20)
        
        # Applications table
        table_layout = self.create_applications_table()
        main_content_layout.addLayout(table_layout)
        
        content_layout.addWidget(main_content, 1)
        dashboard_layout.addLayout(content_layout)
        
        # Analytics tab
        self.analytics_tab = QWidget()
        self.analytics_tab.setStyleSheet("background-color: white;")
        self.analytics_layout = QVBoxLayout(self.analytics_tab)
        self.analytics_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(dashboard_tab, "📊 Dashboard")
        self.tab_widget.addTab(self.analytics_tab, "📈 Analytics")
        
        # Connect tab change to update analytics
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        main_layout.addWidget(self.tab_widget)
        
    def create_header(self):
        """Create the header with gradient background and search"""
        header_widget = QWidget()
        header_widget.setObjectName("headerWidget")
        header_widget.setStyleSheet("""
            #headerWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6c5ce7, stop:1 #a29bfe);
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 10px;
            }
        """)
        
        header_layout = QHBoxLayout(header_widget)
        
        # Title with explicit white color
        title_label = QLabel("APPLICATION TRACKER")
        title_label.setStyleSheet("""
            color: white; 
            font-size: 20px; 
            font-weight: bold;
            background-color: transparent;
        """)
        
        # Search box
        search_box = QLineEdit()
        search_box.setPlaceholderText("Search applications...")
        search_box.setStyleSheet("""
            QLineEdit {
                border-radius: 20px;
                padding: 8px 15px;
                background: white;
                min-width: 250px;
                color: #333333;
                border: none;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid rgba(255, 255, 255, 0.8);
            }
        """)
        
        # Add New button
        add_button = QPushButton("+ Add Application")
        add_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border-radius: 20px;
                padding: 8px 20px;
                font-weight: bold;
                font-size: 14px;
                border: none;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #229954;
            }
        """)
        add_button.clicked.connect(self.open_add_application_dialog)
        
        # Add widgets to header
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(search_box)
        header_layout.addWidget(add_button)
        
        # Create container layout for the header
        container_layout = QVBoxLayout()
        container_layout.addWidget(header_widget)
        
        return container_layout
        
    def create_status_cards(self):
        """Create status summary cards"""
        status_layout = QHBoxLayout()
        status_layout.setSpacing(15)
        
        # Get status counts from database
        status_counts = self.db_manager.get_status_counts()
        
        # Create cards for each status with explicit counts
        applied_count = status_counts.get(Application.STATUS_APPLIED, 0)
        applied_card = StatusCard("Applied", applied_count, Application.STATUS_COLORS[Application.STATUS_APPLIED])
        status_layout.addWidget(applied_card)
        
        interviewing_count = status_counts.get(Application.STATUS_INTERVIEWING, 0)
        interviewing_card = StatusCard("Interviewing", interviewing_count, Application.STATUS_COLORS[Application.STATUS_INTERVIEWING])
        status_layout.addWidget(interviewing_card)
        
        offer_count = status_counts.get(Application.STATUS_OFFER, 0)
        offer_card = StatusCard("Offer", offer_count, Application.STATUS_COLORS[Application.STATUS_OFFER])
        status_layout.addWidget(offer_card)
        
        rejected_count = status_counts.get(Application.STATUS_REJECTED, 0)
        rejected_card = StatusCard("Rejected", rejected_count, Application.STATUS_COLORS[Application.STATUS_REJECTED])
        status_layout.addWidget(rejected_card)
        
        return status_layout
        
    def create_applications_table(self):
        """Create applications table"""
        table_layout = QVBoxLayout()
        
        # Table header with explicit styling
        table_header = QLabel("Applications")
        table_header.setStyleSheet("""
            font-weight: bold; 
            font-size: 18px;
            color: #2d3748;
            background-color: transparent;
            margin-bottom: 10px;
        """)
        table_layout.addWidget(table_header)
        
        # Create table with location column
        self.table = QTableWidget()
        self.table.setColumnCount(6)  # Added location column
        self.table.setHorizontalHeaderLabels(["Company", "Role", "Location", "Date Applied", "Status", "Actions"])
        
        # Set column widths to properly accommodate content
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)  # Company - Fixed to reduce width
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # Role
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Location
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Date Applied
        header.setSectionResizeMode(4, QHeaderView.Fixed)  # Status - Fixed width
        header.setSectionResizeMode(5, QHeaderView.Fixed)  # Actions - Fixed width
        
        # Set specific column widths for better alignment and visibility
        self.table.setColumnWidth(0, 200)  # Company column - fixed reasonable width
        self.table.setColumnWidth(4, 130)  # Status column width - slightly wider
        self.table.setColumnWidth(5, 180)  # Actions column width - increased for button visibility
        
        # Style the table
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                alternate-background-color: #f8f9fa;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                gridline-color: #e2e8f0;
                color: #333333;
                font-size: 13px;
            }
            QHeaderView::section {
                background-color: #f1f3f4;
                padding: 12px 12px 12px 0px;
                border: none;
                border-bottom: 2px solid #e2e8f0;
                font-weight: bold;
                color: #2d3748;
                font-size: 14px;
                text-align: left;
                margin: 0px;
            }
            QTableWidget::item {
                color: #333333;
                padding: 12px;
                border-bottom: 1px solid #f0f0f0;
                text-align: left;
                vertical-align: middle;
            }
            QTableWidget::item:selected {
                background-color: rgba(108, 92, 231, 0.1);
                color: #2d3748;
            }
        """)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(400)
        # Set optimal row height for the updated widgets and alignment
        self.table.verticalHeader().setDefaultSectionSize(50)
        
        table_layout.addWidget(self.table)
        
        return table_layout
        
    def load_applications(self):
        """Load applications from database"""
        self.applications = self.db_manager.get_applications()
        self.update_table(self.applications)
        self.update_status_cards()
        
    def update_table(self, applications):
        """Update the table with applications data"""
        self.table.setRowCount(0)
        
        for row, app_data in enumerate(applications):
            app = Application.from_dict(app_data)
            self.table.insertRow(row)
            
            # Company
            company_item = QTableWidgetItem(app.company)
            company_item.setData(Qt.UserRole, app.id)
            company_item.setForeground(QBrush(QColor("#333333")))
            self.table.setItem(row, 0, company_item)
            
            # Role
            role_item = QTableWidgetItem(app.role)
            role_item.setForeground(QBrush(QColor("#333333")))
            self.table.setItem(row, 1, role_item)
            
            # Location
            location_item = QTableWidgetItem(app.location or "Not specified")
            location_item.setForeground(QBrush(QColor("#666666")))
            self.table.setItem(row, 2, location_item)
            
            # Date Applied
            date_item = QTableWidgetItem(app.date_applied)
            date_item.setForeground(QBrush(QColor("#333333")))
            self.table.setItem(row, 3, date_item)
            
            # Status with proper styling - simplified and better aligned
            status_container = QWidget()
            status_container.setStyleSheet("background-color: transparent;")
            status_layout = QHBoxLayout(status_container)
            status_layout.setContentsMargins(12, 0, 12, 0)  # Horizontal alignment with header padding
            status_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            
            status_widget = QLabel(app.status)
            status_widget.setStyleSheet(f"""
                QLabel {{
                    background-color: {app.get_status_color()};
                    color: white;
                    border-radius: 12px;
                    padding: 6px 12px;
                    font-size: 11px;
                    font-weight: bold;
                    text-align: center;
                }}
            """)
            status_widget.setAlignment(Qt.AlignCenter)
            status_widget.setFixedSize(100, 24)  # Fixed size for consistent appearance
            status_layout.addWidget(status_widget)
            status_layout.addStretch()  # Push status badge to the left
            
            self.table.setCellWidget(row, 4, status_container)
            
            # Actions with properly sized buttons - improved layout and alignment
            actions_widget = QWidget()
            actions_widget.setStyleSheet("background-color: transparent;")
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(12, 0, 12, 0)  # Horizontal alignment with header padding
            actions_layout.setSpacing(8)
            actions_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            
            edit_button = QPushButton("Edit")
            edit_button.setStyleSheet("""
                QPushButton {
                    background-color: #28a78e;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 6px 12px;
                    font-size: 11px;
                    font-weight: bold;
                    min-width: 55px;
                    max-width: 70px;
                    height: 28px;
                }
                QPushButton:hover {
                    background-color: #28a78e;
                }
                QPushButton:pressed {
                    background-color: #1e7e34;
                }
            """)
            edit_button.clicked.connect(lambda _, app_id=app.id: self.edit_application(app_id))
            
            delete_button = QPushButton("Delete")
            delete_button.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 6px 12px;
                    font-size: 11px;
                    font-weight: bold;
                    min-width: 55px;
                    max-width: 70px;
                    height: 28px;
                }
                QPushButton:hover {
                    background-color: #c82333;
                }
                QPushButton:pressed {
                    background-color: #bd2130;
                }
            """)
            delete_button.clicked.connect(lambda _, app_id=app.id: self.delete_application(app_id))
            
            actions_layout.addWidget(edit_button)
            actions_layout.addWidget(delete_button)
            actions_layout.addStretch()  # Push buttons to the left
            
            self.table.setCellWidget(row, 5, actions_widget)
    
    def update_status_cards(self):
        """Update status cards with current counts"""
        status_counts = self.db_manager.get_status_counts()
        self.update_status_cards_with_data(self.applications)
        
    def update_status_cards_with_data(self, applications):
        """Update status cards with data from specific applications list"""
        # Count statuses from the provided applications
        status_counts = {}
        for app in applications:
            status = app.get('status', 'Applied')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Update the status cards if they exist
        # Note: This would require storing references to the status cards
        # For now, we'll just pass - this can be enhanced later
        pass
        
    def open_add_application_dialog(self):
        """Open dialog to add a new application"""
        dialog = AddApplicationDialog(self)
        if dialog.exec_():
            # Reload applications after adding
            self.load_applications()
            
    def edit_application(self, application_id):
        """Open dialog to edit an application"""
        # Find application data
        app_data = next((a for a in self.applications if a['id'] == application_id), None)
        if not app_data:
            return
            
        dialog = AddApplicationDialog(self, application=Application.from_dict(app_data))
        if dialog.exec_():
            # Reload applications after editing
            self.load_applications()
            
    def delete_application(self, application_id):
        """Delete an application"""
        self.db_manager.delete_application(application_id)
        self.load_applications()
        
    def apply_filters(self, filters):
        """Apply filters from the filter panel"""
        filtered_applications = self.applications.copy()
        
        # Apply company filter
        if 'company' in filters and filters['company']:
            company_filter = filters['company'].lower()
            filtered_applications = [
                app for app in filtered_applications 
                if company_filter in app.get('company', '').lower()
            ]
        
        # Apply status filter
        if 'status' in filters and filters['status'] != "All Statuses":
            filtered_applications = [
                app for app in filtered_applications 
                if app.get('status') == filters['status']
            ]
        
        # Apply date range filter
        if 'date_from' in filters and 'date_to' in filters:
            from datetime import datetime
            try:
                date_from = datetime.strptime(filters['date_from'], '%Y-%m-%d').date()
                date_to = datetime.strptime(filters['date_to'], '%Y-%m-%d').date()
                
                filtered_applications = [
                    app for app in filtered_applications 
                    if date_from <= datetime.strptime(app.get('date_applied', ''), '%Y-%m-%d').date() <= date_to
                ]
            except (ValueError, TypeError):
                # If date parsing fails, skip date filtering
                pass
        
        # Update table with filtered applications
        self.update_table(filtered_applications)
        
        # Update status cards with filtered data
        self.update_status_cards_with_data(filtered_applications)
        
    def on_tab_changed(self, index):
        """Handle tab change to update analytics"""
        if index == 1:  # Analytics tab
            self.update_analytics()
            
    def update_analytics(self):
        """Update the analytics tab with current data"""
        # Clear existing analytics content
        for i in reversed(range(self.analytics_layout.count())):
            self.analytics_layout.itemAt(i).widget().setParent(None)
            
        # Create new analytics widget with database manager
        analytics_widget = AnalyticsWidget(self.db_manager, self)
        self.analytics_layout.addWidget(analytics_widget) 