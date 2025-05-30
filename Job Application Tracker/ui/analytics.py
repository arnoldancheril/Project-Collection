"""
Analytics UI
Provides data visualization and analytics for job applications.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                           QPushButton, QFrame, QTabWidget, QComboBox, 
                           QSizePolicy, QGridLayout, QTableWidget, 
                           QTableWidgetItem, QHeaderView, QScrollArea,
                           QSpacerItem)
from PyQt5.QtCore import Qt, QSize, QDate, QMargins
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont, QPainterPath, QLinearGradient, QGradient
from PyQt5.QtChart import (QChart, QChartView, QBarSeries, QBarSet, 
                          QBarCategoryAxis, QValueAxis, QPieSeries, 
                          QLineSeries, QSplineSeries, QDateTimeAxis,
                          QAreaSeries)

from database.db_manager import DatabaseManager
from models.application import Application
from utils.date_helpers import get_date_range
import assets.styles as styles
from datetime import datetime, timedelta
import random  # For demo data if needed

class FunnelChart(QWidget):
    """Funnel chart for application pipeline visualization"""
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(300)
        self.data = self._get_data()
        
    def _get_data(self):
        """Get funnel data from database"""
        status_counts = self.db_manager.get_status_counts()
        
        data = [
            {"status": Application.STATUS_APPLIED, "count": status_counts.get(Application.STATUS_APPLIED, 0), 
             "color": QColor(Application.STATUS_COLORS[Application.STATUS_APPLIED])},
            {"status": Application.STATUS_INTERVIEWING, "count": status_counts.get(Application.STATUS_INTERVIEWING, 0), 
             "color": QColor(Application.STATUS_COLORS[Application.STATUS_INTERVIEWING])},
            {"status": Application.STATUS_OFFER, "count": status_counts.get(Application.STATUS_OFFER, 0), 
             "color": QColor(Application.STATUS_COLORS[Application.STATUS_OFFER])},
            {"status": Application.STATUS_REJECTED, "count": status_counts.get(Application.STATUS_REJECTED, 0), 
             "color": QColor("#f74d4d")}  # Darker red for rejected
        ]
        
        return data
    
    def paintEvent(self, event):
        """Draw the funnel chart"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Calculate total to determine relative sizes
        total = max(sum(item["count"] for item in self.data), 1)
        
        # Calculate funnel segments
        y_offset = 20
        segment_height = (height - y_offset - 40) / len(self.data)
        
        for i, item in enumerate(self.data):
            ratio = item["count"] / total
            top_width = width * 0.8 * (1 - (i * 0.15))
            bottom_width = width * 0.8 * (1 - ((i + 1) * 0.15))
            
            x_start = (width - top_width) / 2
            y_start = y_offset + (i * segment_height)
            
            # Draw trapezoid for funnel segment
            path = QPainterPath()
            path.moveTo(x_start, y_start)
            path.lineTo(x_start + top_width, y_start)
            path.lineTo(x_start + top_width - ((top_width - bottom_width) / 2), y_start + segment_height)
            path.lineTo(x_start + ((top_width - bottom_width) / 2), y_start + segment_height)
            path.closeSubpath()
            
            painter.setBrush(QBrush(item["color"]))
            painter.setPen(Qt.NoPen)
            painter.drawPath(path)
            
            # Draw text
            painter.setPen(Qt.white)
            font = QFont()
            font.setBold(True)
            font.setPointSize(12)
            painter.setFont(font)
            
            text_rect = path.boundingRect()
            painter.drawText(text_rect, Qt.AlignCenter, str(item["count"]))
            
            # Draw status label below the number
            font.setPointSize(10)
            font.setBold(False)
            painter.setFont(font)
            status_rect = text_rect
            status_rect.translate(0, 20)
            painter.drawText(status_rect, Qt.AlignCenter, item["status"])

class ApplicationsOverTime(QWidget):
    """Line chart showing applications over time"""
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create chart
        chart = QChart()
        chart.setTitle("")
        chart.setAnimationOptions(QChart.SeriesAnimations)
        chart.setTheme(QChart.ChartThemeLight)
        chart.setBackgroundVisible(False)
        chart.setMargins(QMargins(0, 0, 0, 0))
        chart.layout().setContentsMargins(0, 0, 0, 0)
        
        # Create chart view
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        chart_view.setBackgroundBrush(QBrush(QColor("#ffffff")))
        chart_view.setFrameShape(QFrame.NoFrame)
        
        layout.addWidget(chart_view)
        
        self.chart = chart
        self.chart_view = chart_view
        
        # Update chart with data
        self.update_chart()
        
    def update_chart(self):
        """Update the chart with current data"""
        # Clear existing series
        self.chart.removeAllSeries()
        
        # Create line series for applications over time
        series = QLineSeries()
        series.setName("Applications")
        series.setColor(QColor(styles.COLORS["primary"]))
        pen = series.pen()
        pen.setWidth(3)
        series.setPen(pen)
        
        # Demo data - replace with actual data from database in production
        # In a real implementation, you would query the database for applications by date
        # Example: 
        # applications = self.db_manager.get_applications_by_month()
        
        # For now, using mock data spanning 6 months
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        
        # Generate random data points
        cumulative = 0
        for i in range(6):
            # Random increase between 1-5 applications per month
            increase = random.randint(1, 5)
            cumulative += increase
            # Add point to series with x,y coordinates
            series.append(i, cumulative)
        
        self.chart.addSeries(series)
        
        # Create axes
        axis_x = QBarCategoryAxis()
        axis_x.append(months)
        
        axis_y = QValueAxis()
        axis_y.setRange(0, cumulative + 2)
        axis_y.setTickCount(5)
        axis_y.setLabelFormat("%d")
        
        self.chart.addAxis(axis_x, Qt.AlignBottom)
        self.chart.addAxis(axis_y, Qt.AlignLeft)
        
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)
        
        # Set legend visibility
        self.chart.legend().setVisible(False)

class ActivityHeatmap(QWidget):
    """Heatmap showing application activity over time"""
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # This would be replaced with actual data in a real implementation
        # For now, we'll display a placeholder
        label = QLabel("Application Activity")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        # Grid for the heatmap
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(4)
        
        # Sample months and days
        months = ["Feb", "Mar", "Apr", "May"]
        days = ["", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        
        # Create month labels (columns)
        for i, month in enumerate(months):
            month_label = QLabel(month)
            month_label.setAlignment(Qt.AlignCenter)
            month_label.setStyleSheet("color: #666; font-size: 12px;")
            grid_layout.addWidget(month_label, 0, i + 1)
        
        # Create day labels (rows)
        for i, day in enumerate(days):
            if i > 0:  # Skip the first empty cell
                day_label = QLabel(day)
                day_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                day_label.setStyleSheet("color: #666; font-size: 12px;")
                grid_layout.addWidget(day_label, i, 0)
        
        # Create colored cells for the heatmap (random data for demo)
        for row in range(1, len(days)):
            for col in range(1, len(months) + 1):
                # Random activity level
                activity = random.randint(0, 3)
                
                cell = QFrame()
                cell.setFixedSize(30, 30)
                
                # Color based on activity
                if activity == 0:
                    color = "#f5f5f5"  # No activity
                elif activity == 1:
                    color = "#c4b5fd"  # Low activity
                elif activity == 2:
                    color = "#a78bfa"  # Medium activity
                else:
                    color = "#8b5cf6"  # High activity
                
                cell.setStyleSheet(f"""
                    background-color: {color};
                    border-radius: 4px;
                """)
                
                grid_layout.addWidget(cell, row, col)
        
        layout.addWidget(label)
        layout.addWidget(grid_widget)
        layout.addStretch()

class TopCompaniesTable(QWidget):
    """Table showing top companies applied to"""
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("Top Companies")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        count_label = QLabel("# Applications")
        count_label.setStyleSheet("color: #666; font-size: 12px;")
        count_label.setAlignment(Qt.AlignRight)
        
        header.addWidget(title)
        header.addStretch()
        header.addWidget(count_label)
        
        # Table for companies
        table = QTableWidget(0, 2)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Company", "# Applications"])
        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        table.horizontalHeader().resizeSection(1, 100)
        table.setShowGrid(False)
        table.setFrameShape(QFrame.NoFrame)
        table.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                border: none;
            }
            QTableWidget::item {
                padding: 5px 0;
                border-bottom: 1px solid #f0f0f0;
            }
        """)
        
        # Get actual data from database
        companies = self.db_manager.get_company_counts(limit=6)
        
        # If no data, show placeholder data
        if not companies:
            companies = [
                ("Google", 5),
                ("Microsoft", 4),
                ("Amazon", 3),
                ("OpenAI", 2),
                ("LinkedIn", 2),
                ("Meta", 2)
            ]
        
        # Populate table
        table.setRowCount(len(companies))
        for row, (company, count) in enumerate(companies):
            # Company name
            name_item = QTableWidgetItem(company)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 0, name_item)
            
            # Application count
            count_item = QTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            count_item.setFlags(count_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 1, count_item)
        
        layout.addLayout(header)
        layout.addWidget(table)

class FollowUpsWidget(QWidget):
    """Widget showing pending follow-ups"""
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("Follow Ups")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        count_label = QLabel("1")  # Mock data
        count_label.setStyleSheet("color: #666; font-size: 12px;")
        count_label.setAlignment(Qt.AlignRight)
        
        header.addWidget(title)
        header.addStretch()
        header.addWidget(count_label)
        
        # Sample follow-up item
        follow_up = QFrame()
        follow_up.setFrameShape(QFrame.StyledPanel)
        follow_up.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #f0f0f0;
                padding: 8px;
            }
        """)
        
        follow_up_layout = QHBoxLayout(follow_up)
        
        icon = QLabel("📋")
        icon.setStyleSheet("font-size: 16px;")
        
        content = QLabel("Send follow-up email")
        content.setStyleSheet("font-size: 13px;")
        
        date = QLabel("Jun 7")
        date.setStyleSheet("color: #666; font-size: 12px;")
        date.setAlignment(Qt.AlignRight)
        
        follow_up_layout.addWidget(icon)
        follow_up_layout.addWidget(content)
        follow_up_layout.addStretch()
        follow_up_layout.addWidget(date)
        
        layout.addLayout(header)
        layout.addWidget(follow_up)
        layout.addStretch()

class AverageTimeInStageWidget(QWidget):
    """Widget showing average time in each application stage"""
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel("Avg. Time in Stage")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Create chart
        chart = QChart()
        chart.setTitle("")
        chart.setAnimationOptions(QChart.SeriesAnimations)
        chart.setBackgroundVisible(False)
        chart.setMargins(QMargins(0, 0, 0, 0))
        chart.layout().setContentsMargins(0, 0, 0, 0)
        
        # Bar series
        bar_set = QBarSet("")
        bar_set.setColor(QColor(styles.COLORS["primary"]))
        
        # Mock data for time in stages (in days)
        # Replace with actual data from database
        stages = ["Applied", "Screening", "Technical", "Interview", "Offer", "Accepted"]
        times = [4, 7, 12, 9, 10, 14]
        
        for time in times:
            bar_set.append(time)
        
        series = QBarSeries()
        series.append(bar_set)
        chart.addSeries(series)
        
        # Create axes
        axis_x = QBarCategoryAxis()
        axis_x.append(stages)
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setRange(0, max(times) + 2)
        axis_y.setTickCount(5)
        axis_y.setLabelFormat("%d")
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)
        
        # Create chart view
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        chart_view.setMinimumHeight(200)
        
        layout.addWidget(chart_view)

class StatsCard(QFrame):
    """Card displaying a key statistic"""
    
    def __init__(self, title, value, suffix="", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #f0f0f0;
                padding: 12px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Value
        value_label = QLabel(f"{value}{suffix}")
        value_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        value_label.setAlignment(Qt.AlignCenter)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #666; font-size: 14px;")
        title_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(value_label)
        layout.addWidget(title_label)

class AnalyticsWidget(QWidget):
    """Analytics dashboard widget"""
    
    def __init__(self, db_manager=None, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager or DatabaseManager()
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title section
        icon_label = QLabel("📊")
        icon_label.setStyleSheet("font-size: 24px;")
        
        title_label = QLabel("Analytics")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        
        new_btn = QPushButton("+ New Application")
        new_btn.setStyleSheet(styles.MODERN_PRIMARY_BUTTON_STYLE)
        new_btn.setCursor(Qt.PointingHandCursor)
        
        title_layout = QHBoxLayout()
        title_layout.addWidget(icon_label)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(new_btn)
        main_layout.addLayout(title_layout)
        
        # Stats cards - top row
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)
        
        # Get counts from database
        status_counts = self.db_manager.get_status_counts()
        total_applications = sum(status_counts.values())
        interviewing_count = status_counts.get(Application.STATUS_INTERVIEWING, 0)
        response_rate = int(interviewing_count / total_applications * 100) if total_applications > 0 else 0
        
        # Get last application date
        last_applied_date = self.db_manager.get_last_application_date()
        days_since_last_app = 0
        if last_applied_date:
            from datetime import datetime, date
            last_date = datetime.strptime(last_applied_date, "%Y-%m-%d").date()
            today = date.today()
            days_since_last_app = (today - last_date).days
        
        # Create stat cards
        total_card = StatsCard("Total Applications", total_applications)
        response_card = StatsCard("Response Rate", response_rate, "%")
        last_applied_card = StatsCard("Last Application", days_since_last_app, " days ago")
        last_activity_card = StatsCard("Last Application", days_since_last_app, " days")
        avg_start_card = StatsCard("Avg. Days", "5,6", " days")
        avg_end_card = StatsCard("Avg. Days", "28", " days")
        
        stats_layout.addWidget(total_card)
        stats_layout.addWidget(response_card)
        stats_layout.addWidget(last_applied_card)
        stats_layout.addWidget(last_activity_card)
        stats_layout.addWidget(avg_start_card)
        stats_layout.addWidget(avg_end_card)
        
        main_layout.addLayout(stats_layout)
        
        # Create scrollable area for charts and tables
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(16)
        
        # First row - pipeline and charts
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(16)
        
        # Pipeline funnel
        pipeline_frame = QFrame()
        pipeline_frame.setFrameShape(QFrame.StyledPanel)
        pipeline_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #f0f0f0;
                padding: 16px;
            }
        """)
        
        pipeline_layout = QVBoxLayout(pipeline_frame)
        
        pipeline_title = QLabel("Pipeline")
        pipeline_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        
        funnel_chart = FunnelChart(self.db_manager)
        
        pipeline_layout.addWidget(pipeline_title)
        pipeline_layout.addWidget(funnel_chart)
        
        # Applications over time chart
        applications_frame = QFrame()
        applications_frame.setFrameShape(QFrame.StyledPanel)
        applications_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #f0f0f0;
                padding: 16px;
            }
        """)
        
        applications_layout = QVBoxLayout(applications_frame)
        
        applications_title = QLabel("Applications Over Time")
        applications_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        
        applications_chart = ApplicationsOverTime(self.db_manager)
        
        applications_layout.addWidget(applications_title)
        applications_layout.addWidget(applications_chart)
        
        # Activity heatmap
        activity_frame = QFrame()
        activity_frame.setFrameShape(QFrame.StyledPanel)
        activity_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #f0f0f0;
                padding: 16px;
            }
        """)
        
        activity_layout = QVBoxLayout(activity_frame)
        
        activity_title = QLabel("Application Activity")
        activity_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        
        activity_chart = ActivityHeatmap(self.db_manager)
        
        activity_layout.addWidget(activity_title)
        activity_layout.addWidget(activity_chart)
        
        row1_layout.addWidget(pipeline_frame, 1)
        row1_layout.addWidget(applications_frame, 2)
        row1_layout.addWidget(activity_frame, 2)
        
        # Second row - tables and chart
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(16)
        
        # Top companies
        companies_frame = QFrame()
        companies_frame.setFrameShape(QFrame.StyledPanel)
        companies_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #f0f0f0;
                padding: 16px;
            }
        """)
        
        companies_layout = QVBoxLayout(companies_frame)
        companies_widget = TopCompaniesTable(self.db_manager)
        companies_layout.addWidget(companies_widget)
        
        # Follow-ups
        followups_frame = QFrame()
        followups_frame.setFrameShape(QFrame.StyledPanel)
        followups_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #f0f0f0;
                padding: 16px;
            }
        """)
        
        followups_layout = QVBoxLayout(followups_frame)
        followups_widget = FollowUpsWidget(self.db_manager)
        followups_layout.addWidget(followups_widget)
        
        # Avg time in stage
        avgtime_frame = QFrame()
        avgtime_frame.setFrameShape(QFrame.StyledPanel)
        avgtime_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #f0f0f0;
                padding: 16px;
            }
        """)
        
        avgtime_layout = QVBoxLayout(avgtime_frame)
        avgtime_widget = AverageTimeInStageWidget(self.db_manager)
        avgtime_layout.addWidget(avgtime_widget)
        
        row2_layout.addWidget(companies_frame, 1)
        row2_layout.addWidget(followups_frame, 1)
        
        row3_layout = QHBoxLayout()
        row3_layout.addWidget(avgtime_frame)
        
        # Add rows to scroll layout
        scroll_layout.addLayout(row1_layout)
        scroll_layout.addLayout(row2_layout)
        scroll_layout.addLayout(row3_layout)
        scroll_layout.addStretch()
        
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
    def refresh_analytics(self):
        """Refresh all analytics data"""
        # This would update all charts and tables with fresh data
        self.setup_ui() 