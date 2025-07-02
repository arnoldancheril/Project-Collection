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

class EnhancedApplicationsOverTime(QWidget):
    """Enhanced line chart showing applications over time with detailed metrics"""
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the enhanced chart"""
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
        chart_view.setMinimumHeight(350)
        
        layout.addWidget(chart_view)
        
        self.chart = chart
        self.chart_view = chart_view
        
        # Update chart with enhanced data
        self.update_chart()
        
    def update_chart(self):
        """Update the chart with enhanced data visualization"""
        # Clear existing series
        self.chart.removeAllSeries()
        
        # Get applications data
        applications = self.db_manager.get_applications()
        
        if not applications:
            # Show empty state
            series = QLineSeries()
            series.setName("No Data")
            for i in range(6):
                series.append(i, 0)
            self.chart.addSeries(series)
            return
        
        # Process applications by month for better granularity
        from collections import defaultdict
        from datetime import datetime, timedelta
        
        monthly_data = defaultdict(int)
        monthly_cumulative = defaultdict(int)
        
        # Parse and group applications by month
        for app in applications:
            try:
                date_obj = datetime.strptime(app['date_applied'], "%Y-%m-%d")
                month_key = date_obj.strftime("%Y-%m")
                monthly_data[month_key] += 1
            except (ValueError, TypeError):
                continue
        
        # Create sorted list of months
        sorted_months = sorted(monthly_data.keys())
        
        if not sorted_months:
            return
            
        # Generate more detailed time series
        series_monthly = QLineSeries()
        series_monthly.setName("Applications per Month")
        series_monthly.setColor(QColor("#6c5ce7"))
        pen = series_monthly.pen()
        pen.setWidth(3)
        series_monthly.setPen(pen)
        
        # Cumulative series
        series_cumulative = QLineSeries()
        series_cumulative.setName("Total Applications")
        series_cumulative.setColor(QColor("#2ecc71"))
        pen_cum = series_cumulative.pen()
        pen_cum.setWidth(2)
        pen_cum.setStyle(Qt.DashLine)
        series_cumulative.setPen(pen_cum)
        
        # Calculate data points
        cumulative_total = 0
        month_labels = []
        
        for i, month in enumerate(sorted_months):
            monthly_count = monthly_data[month]
            cumulative_total += monthly_count
            
            # Add points to series
            series_monthly.append(i, monthly_count)
            series_cumulative.append(i, cumulative_total)
            
            # Format month label
            try:
                date_obj = datetime.strptime(month, "%Y-%m")
                month_labels.append(date_obj.strftime("%b %Y"))
            except (ValueError, TypeError):
                month_labels.append(month)
        
        # Add series to chart
        self.chart.addSeries(series_monthly)
        self.chart.addSeries(series_cumulative)
        
        # Create enhanced axes
        axis_x = QBarCategoryAxis()
        axis_x.append(month_labels)
        axis_x.setLabelsColor(QColor("#4a5568"))
        self.chart.addAxis(axis_x, Qt.AlignBottom)
        series_monthly.attachAxis(axis_x)
        series_cumulative.attachAxis(axis_x)
        
        # Y-axis for monthly applications
        max_monthly = max(monthly_data.values()) if monthly_data else 1
        axis_y_left = QValueAxis()
        axis_y_left.setRange(0, max_monthly + 2)
        axis_y_left.setTickCount(min(max_monthly + 1, 8))
        axis_y_left.setLabelsColor(QColor("#4a5568"))
        axis_y_left.setTitleText("Monthly Applications")
        self.chart.addAxis(axis_y_left, Qt.AlignLeft)
        series_monthly.attachAxis(axis_y_left)
        
        # Y-axis for cumulative (right side)
        axis_y_right = QValueAxis()
        axis_y_right.setRange(0, cumulative_total + 5)
        axis_y_right.setTickCount(8)
        axis_y_right.setLabelsColor(QColor("#4a5568"))
        axis_y_right.setTitleText("Total Applications")
        self.chart.addAxis(axis_y_right, Qt.AlignRight)
        series_cumulative.attachAxis(axis_y_right)
        
        # Style the chart
        self.chart.legend().setAlignment(Qt.AlignTop)
        self.chart.legend().setBackgroundVisible(False)
        self.chart.legend().setLabelColor(QColor("#2d3748"))

class ActivityHeatmap(QWidget):
    """Heatmap showing application activity over time"""
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create application activity label
        label = QLabel("Application Activity")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        # Grid for the heatmap
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(4)
        
        # Get real application data
        applications = self.db_manager.get_applications()
        
        # Process application data by date
        date_activity = {}
        for app in applications:
            date_str = app.get('date_applied')
            if not date_str:
                continue
                
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                date_key = date_obj.strftime("%Y-%m-%d")
                
                if date_key not in date_activity:
                    date_activity[date_key] = 0
                date_activity[date_key] += 1
            except (ValueError, TypeError):
                # Skip invalid dates
                continue
        
        # Get the months from existing data or use default
        months_set = set()
        for date_key in date_activity.keys():
            try:
                date_obj = datetime.strptime(date_key, "%Y-%m-%d")
                months_set.add(date_obj.strftime("%b"))
            except (ValueError, TypeError):
                continue
        
        # Sample months and days
        months = sorted(list(months_set)) if months_set else ["Feb", "Mar", "Apr", "May"]
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
        
        # Create colored cells for the heatmap based on real data
        for row in range(1, len(days)):
            day_of_week = row  # 1=Monday, 7=Sunday
            
            for col in range(1, len(months) + 1):
                month_name = months[col-1]
                
                # Find activity level for this day/month combination
                activity = 0
                
                # Look through all dates in our activity data
                for date_key in date_activity.keys():
                    try:
                        date_obj = datetime.strptime(date_key, "%Y-%m-%d")
                        if (date_obj.strftime("%b") == month_name and 
                            date_obj.weekday() + 1 == day_of_week):
                            # This date matches our current cell
                            activity = date_activity[date_key]
                            break
                    except (ValueError, TypeError):
                        continue
                
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
                
                cell.setStyleSheet(f"background-color: {color}; border-radius: 4px;")
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
        
        # Title
        title = QLabel("Top Companies")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        
        # Create table
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["", "# Applications"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setShowGrid(False)
        table.setStyleSheet("""
            QTableWidget {
                border: none;
                background-color: transparent;
            }
            QHeaderView::section {
                background-color: transparent;
                padding: 6px;
                border: none;
                font-weight: bold;
                color: #666;
            }
            QTableWidget::item {
                padding: 6px;
                border-bottom: 1px solid #f0f0f0;
            }
        """)
        
        # Get data from database
        companies = self.db_manager.get_company_counts(limit=6)
        
        if companies:
            table.setRowCount(len(companies))
            
            for i, (company, count) in enumerate(companies):
                # Company name
                name_item = QTableWidgetItem(company)
                
                # Application count
                count_item = QTableWidgetItem(str(count))
                count_item.setTextAlignment(Qt.AlignCenter)
                
                table.setItem(i, 0, name_item)
                table.setItem(i, 1, count_item)
        else:
            # No data, add a placeholder row
            table.setRowCount(1)
            table.setItem(0, 0, QTableWidgetItem("No applications yet"))
            table.setItem(0, 1, QTableWidgetItem(""))
        
        layout.addWidget(title)
        layout.addWidget(table)

class FollowUpsWidget(QWidget):
    """Widget showing applications that need follow-up"""
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title = QLabel("Follow Ups")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title)
        
        # Get follow-up data from database
        follow_ups = self.db_manager.get_follow_ups(limit=5)
        
        if follow_ups:
            # Create a list for follow-ups
            for follow_up in follow_ups:
                item = QFrame()
                item.setFrameShape(QFrame.StyledPanel)
                item.setStyleSheet("""
                    QFrame {
                        background-color: #f9f9f9;
                        border-radius: 4px;
                        border: 1px solid #f0f0f0;
                        margin-bottom: 8px;
                    }
                """)
                
                item_layout = QHBoxLayout(item)
                item_layout.setContentsMargins(10, 10, 10, 10)
                
                # Checkbox
                checkbox = QFrame()
                checkbox.setFixedSize(20, 20)
                checkbox.setStyleSheet("""
                    background-color: white;
                    border: 1px solid #e0e0e0;
                    border-radius: 3px;
                """)
                
                # Company and role
                company = follow_up.get('company', '')
                role = follow_up.get('role', '')
                text = QLabel(f"Send follow-up email")
                text.setStyleSheet("color: #333;")
                
                # Date applied - calculate days since
                date_applied = follow_up.get('date_applied', '')
                days_ago = ''
                if date_applied:
                    try:
                        applied_date = datetime.strptime(date_applied, "%Y-%m-%d").date()
                        today = datetime.now().date()
                        days = (today - applied_date).days
                        days_ago = f"Jun {days}"
                    except (ValueError, TypeError):
                        days_ago = "Unknown"
                
                date_label = QLabel(days_ago)
                date_label.setStyleSheet("color: #888;")
                date_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                
                item_layout.addWidget(checkbox)
                item_layout.addWidget(text)
                item_layout.addStretch()
                item_layout.addWidget(date_label)
                
                layout.addWidget(item)
        else:
            # No follow-ups
            no_data = QLabel("No follow-ups needed")
            no_data.setStyleSheet("color: #888; font-size: 12px; margin-top: 10px;")
            no_data.setAlignment(Qt.AlignCenter)
            layout.addWidget(no_data)
        
        layout.addStretch()

class EnhancedTimeInStageWidget(QWidget):
    """Enhanced widget showing detailed time analysis for application stages"""
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the enhanced time analysis widget"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        # Get applications data
        applications = self.db_manager.get_applications()
        time_data = self.calculate_detailed_time_metrics(applications)
        
        # Create metrics summary cards
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(12)
        
        for metric in time_data['summary']:
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    border: 1px solid #e9ecef;
                    padding: 12px;
                }
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setAlignment(Qt.AlignCenter)
            
            value_label = QLabel(f"{metric['value']}")
            value_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2d3748;")
            value_label.setAlignment(Qt.AlignCenter)
            
            title_label = QLabel(metric['title'])
            title_label.setStyleSheet("font-size: 10px; color: #718096; margin-top: 4px;")
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setWordWrap(True)
            
            card_layout.addWidget(value_label)
            card_layout.addWidget(title_label)
            metrics_layout.addWidget(card)
        
        layout.addLayout(metrics_layout)
        
        # Create detailed time chart
        chart = self.create_detailed_time_chart(time_data['stages'])
        layout.addWidget(chart)
        
        # Add time-based insights
        insights = self.generate_time_insights(time_data)
        if insights:
            insight_label = QLabel(f"⏰ {insights}")
            insight_label.setStyleSheet("""
                background-color: #fff3cd;
                color: #856404;
                padding: 12px;
                border-radius: 6px;
                border-left: 4px solid #ffc107;
                font-size: 12px;
                margin-top: 8px;
            """)
            insight_label.setWordWrap(True)
            layout.addWidget(insight_label)
            
    def calculate_detailed_time_metrics(self, applications):
        """Calculate detailed time metrics for applications"""
        from datetime import datetime, timedelta
        from collections import defaultdict
        
        if not applications:
            return {
                'summary': [
                    {'title': 'Avg Response Time', 'value': '0 days'},
                    {'title': 'Fastest Response', 'value': '0 days'},
                    {'title': 'Applications This Week', 'value': '0'},
                    {'title': 'Response Rate', 'value': '0%'}
                ],
                'stages': {}
            }
        
        today = datetime.now().date()
        week_start = today - timedelta(days=today.weekday())
        
        response_times = []
        this_week_count = 0
        total_responses = 0
        stage_times = defaultdict(list)
        
        for app in applications:
            try:
                app_date = datetime.strptime(app['date_applied'], "%Y-%m-%d").date()
                status = app.get('status', Application.STATUS_APPLIED)
                
                # Count this week applications
                if app_date >= week_start:
                    this_week_count += 1
                
                # Calculate time in current stage (days since application)
                days_since_applied = (today - app_date).days
                stage_times[status].append(days_since_applied)
                
                # Count responses (not just applied or rejected immediately)
                if status in [Application.STATUS_INTERVIEWING, Application.STATUS_OFFER]:
                    total_responses += 1
                    response_times.append(days_since_applied)
                    
            except (ValueError, TypeError):
                continue
        
        # Calculate summary metrics
        avg_response = sum(response_times) // len(response_times) if response_times else 0
        fastest_response = min(response_times) if response_times else 0
        response_rate = int((total_responses / len(applications)) * 100) if applications else 0
        
        # Calculate average time in each stage
        stage_averages = {}
        for status, times in stage_times.items():
            if times:
                stage_averages[status] = sum(times) // len(times)
        
        return {
            'summary': [
                {'title': 'Avg Response Time', 'value': f'{avg_response} days'},
                {'title': 'Fastest Response', 'value': f'{fastest_response} days'},
                {'title': 'Applications This Week', 'value': str(this_week_count)},
                {'title': 'Response Rate', 'value': f'{response_rate}%'}
            ],
            'stages': stage_averages
        }
        
    def create_detailed_time_chart(self, stage_data):
        """Create a detailed chart showing time spent in each stage"""
        # Create chart
        chart = QChart()
        chart.setTitle("")
        chart.setAnimationOptions(QChart.SeriesAnimations)
        chart.setBackgroundVisible(False)
        chart.setMargins(QMargins(10, 10, 10, 10))
        
        # Create bar series
        bar_series = QBarSeries()
        bar_set = QBarSet("Average Days")
        bar_set.setColor(QColor("#6c5ce7"))
        
        # Prepare data
        statuses = [Application.STATUS_APPLIED, Application.STATUS_INTERVIEWING, Application.STATUS_OFFER, Application.STATUS_REJECTED]
        status_labels = ["Applied", "Interviewing", "Offer", "Rejected"]
        
        for status in statuses:
            avg_days = stage_data.get(status, 0)
            bar_set.append(avg_days)
            
        bar_series.append(bar_set)
        chart.addSeries(bar_series)
        
        # Create axes
        axis_x = QBarCategoryAxis()
        axis_x.append(status_labels)
        axis_x.setLabelsColor(QColor("#4a5568"))
        chart.addAxis(axis_x, Qt.AlignBottom)
        bar_series.attachAxis(axis_x)
        
        max_days = max(stage_data.values()) if stage_data else 1
        axis_y = QValueAxis()
        axis_y.setRange(0, max_days + 5)
        axis_y.setTickCount(min(max_days + 1, 8))
        axis_y.setLabelsColor(QColor("#4a5568"))
        axis_y.setTitleText("Days")
        chart.addAxis(axis_y, Qt.AlignLeft)
        bar_series.attachAxis(axis_y)
        
        # Hide legend
        chart.legend().setVisible(False)
        
        # Create chart view
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        chart_view.setFrameShape(QFrame.NoFrame)
        chart_view.setMinimumHeight(250)
        chart_view.setMaximumHeight(300)
        
        return chart_view
        
    def generate_time_insights(self, time_data):
        """Generate actionable insights from time analysis"""
        stages = time_data['stages']
        
        if not stages:
            return "Start tracking applications to see time-based insights."
        
        # Find longest stage
        if stages:
            longest_stage = max(stages, key=stages.get)
            longest_time = stages[longest_stage]
            
            if longest_time > 30:
                return f"Applications in '{longest_stage}' status average {longest_time} days. Consider following up after 2-3 weeks."
            elif longest_time > 14:
                return f"Average time in '{longest_stage}' is {longest_time} days. This is within normal range."
            else:
                return "Your application process timing looks healthy. Keep up the good pace!"
        
        return ""

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

class EnhancedActivityChart(QWidget):
    """Enhanced chart showing application activity patterns with meaningful insights"""
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the enhanced activity chart"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Activity metrics row
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(16)
        
        # Get applications for analysis
        applications = self.db_manager.get_applications()
        
        # Calculate activity metrics
        activity_metrics = self.calculate_activity_metrics(applications)
        
        # Create metric cards
        for metric in activity_metrics:
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    border: 1px solid #e9ecef;
                    padding: 12px;
                }
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setAlignment(Qt.AlignCenter)
            
            value_label = QLabel(str(metric['value']))
            value_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #2d3748;")
            value_label.setAlignment(Qt.AlignCenter)
            
            title_label = QLabel(metric['title'])
            title_label.setStyleSheet("font-size: 11px; color: #718096; margin-top: 4px;")
            title_label.setAlignment(Qt.AlignCenter)
            
            card_layout.addWidget(value_label)
            card_layout.addWidget(title_label)
            metrics_layout.addWidget(card)
        
        layout.addLayout(metrics_layout)
        
        # Weekly pattern chart
        weekly_chart = self.create_weekly_pattern_chart(applications)
        layout.addWidget(weekly_chart)
        
    def calculate_activity_metrics(self, applications):
        """Calculate meaningful activity metrics"""
        from datetime import datetime, timedelta
        
        if not applications:
            return [
                {'title': 'This Week', 'value': 0},
                {'title': 'Last Week', 'value': 0},
                {'title': 'This Month', 'value': 0},
                {'title': 'Most Active Day', 'value': 'N/A'}
            ]
        
        today = datetime.now().date()
        week_start = today - timedelta(days=today.weekday())
        last_week_start = week_start - timedelta(days=7)
        month_start = today.replace(day=1)
        
        this_week_count = 0
        last_week_count = 0
        this_month_count = 0
        day_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}  # Mon-Sun
        
        for app in applications:
            try:
                app_date = datetime.strptime(app['date_applied'], "%Y-%m-%d").date()
                
                # This week
                if app_date >= week_start:
                    this_week_count += 1
                    
                # Last week
                elif app_date >= last_week_start and app_date < week_start:
                    last_week_count += 1
                    
                # This month
                if app_date >= month_start:
                    this_month_count += 1
                    
                # Day of week counting
                day_counts[app_date.weekday()] += 1
                
            except (ValueError, TypeError):
                continue
        
        # Find most active day
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        most_active_day = max(day_counts, key=day_counts.get)
        most_active_day_name = days[most_active_day] if day_counts[most_active_day] > 0 else 'N/A'
        
        return [
            {'title': 'This Week', 'value': this_week_count},
            {'title': 'Last Week', 'value': last_week_count},
            {'title': 'This Month', 'value': this_month_count},
            {'title': 'Most Active Day', 'value': most_active_day_name}
        ]
        
    def create_weekly_pattern_chart(self, applications):
        """Create a bar chart showing application patterns by day of week"""
        from datetime import datetime
        
        # Create chart
        chart = QChart()
        chart.setTitle("")
        chart.setAnimationOptions(QChart.SeriesAnimations)
        chart.setBackgroundVisible(False)
        chart.setMargins(QMargins(10, 10, 10, 10))
        
        # Create bar series
        bar_series = QBarSeries()
        bar_set = QBarSet("Applications")
        bar_set.setColor(QColor("#6c5ce7"))
        
        # Count applications by day of week
        day_counts = [0] * 7  # Mon-Sun
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        for app in applications:
            try:
                app_date = datetime.strptime(app['date_applied'], "%Y-%m-%d").date()
                day_counts[app_date.weekday()] += 1
            except (ValueError, TypeError):
                continue
        
        # Add data to bar set
        for count in day_counts:
            bar_set.append(count)
            
        bar_series.append(bar_set)
        chart.addSeries(bar_series)
        
        # Create axes
        axis_x = QBarCategoryAxis()
        axis_x.append(days)
        axis_x.setLabelsColor(QColor("#4a5568"))
        chart.addAxis(axis_x, Qt.AlignBottom)
        bar_series.attachAxis(axis_x)
        
        max_count = max(day_counts) if day_counts else 1
        axis_y = QValueAxis()
        axis_y.setRange(0, max_count + 1)
        axis_y.setTickCount(min(max_count + 1, 6))
        axis_y.setLabelsColor(QColor("#4a5568"))
        chart.addAxis(axis_y, Qt.AlignLeft)
        bar_series.attachAxis(axis_y)
        
        # Hide legend
        chart.legend().setVisible(False)
        
        # Create chart view
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        chart_view.setFrameShape(QFrame.NoFrame)
        chart_view.setMinimumHeight(200)
        chart_view.setMaximumHeight(250)
        
        return chart_view

class CompanySuccessChart(QWidget):
    """Chart showing success rates by company to help identify which companies to target"""
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the company success rate chart"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Get applications data
        applications = self.db_manager.get_applications()
        company_data = self.calculate_company_success_rates(applications)
        
        if not company_data:
            # Show empty state
            empty_label = QLabel("No data available yet")
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setStyleSheet("color: #718096; font-size: 14px; padding: 40px;")
            layout.addWidget(empty_label)
            return
        
        # Create success rate table with visual indicators
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Company", "Applied", "Success Rate", "Status"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setShowGrid(False)
        table.setStyleSheet("""
            QTableWidget {
                border: none;
                background-color: transparent;
                alternate-background-color: #f8f9fa;
            }
            QHeaderView::section {
                background-color: #f1f3f4;
                padding: 10px;
                border: none;
                border-bottom: 1px solid #e2e8f0;
                font-weight: bold;
                color: #2d3748;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f0f0f0;
            }
        """)
        table.setAlternatingRowColors(True)
        
        # Sort companies by success rate (descending) and total applications
        sorted_companies = sorted(company_data, key=lambda x: (x['success_rate'], x['total']), reverse=True)
        
        # Show top companies (limit to reasonable number)
        display_companies = sorted_companies[:8]
        table.setRowCount(len(display_companies))
        
        for i, company in enumerate(display_companies):
            # Company name
            name_item = QTableWidgetItem(company['name'])
            table.setItem(i, 0, name_item)
            
            # Total applications
            total_item = QTableWidgetItem(str(company['total']))
            total_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(i, 1, total_item)
            
            # Success rate with visual indicator
            success_rate = company['success_rate']
            rate_widget = QWidget()
            rate_layout = QHBoxLayout(rate_widget)
            rate_layout.setContentsMargins(8, 4, 8, 4)
            
            # Progress bar for visual representation
            progress_bg = QFrame()
            progress_bg.setFixedHeight(8)
            progress_bg.setStyleSheet("background-color: #e2e8f0; border-radius: 4px;")
            
            progress_fill = QFrame(progress_bg)
            fill_width = max(int(success_rate * 0.8), 8) if success_rate > 0 else 0  # Scale for visual appeal
            progress_fill.setFixedSize(fill_width, 8)
            progress_fill.move(0, 0)
            
            # Color based on success rate
            if success_rate >= 60:
                color = "#2ecc71"  # Green for high success
            elif success_rate >= 30:
                color = "#f39c12"  # Orange for medium success
            else:
                color = "#e74c3c"  # Red for low success
                
            progress_fill.setStyleSheet(f"background-color: {color}; border-radius: 4px;")
            
            # Percentage text
            rate_label = QLabel(f"{success_rate:.0f}%")
            rate_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11px;")
            
            rate_layout.addWidget(progress_bg)
            rate_layout.addWidget(rate_label)
            rate_layout.addStretch()
            
            table.setCellWidget(i, 2, rate_widget)
            
            # Recommendation
            if success_rate >= 50:
                recommendation = "🎯 High priority"
                rec_color = "#2ecc71"
            elif success_rate >= 25:
                recommendation = "⚡ Good target"
                rec_color = "#f39c12"
            elif company['total'] >= 3:
                recommendation = "🔄 Try different approach"
                rec_color = "#e74c3c"
            else:
                recommendation = "📊 Need more data"
                rec_color = "#6c5ce7"
                
            rec_item = QTableWidgetItem(recommendation)
            rec_item.setForeground(QBrush(QColor(rec_color)))
            table.setItem(i, 3, rec_item)
        
        # Adjust row height for better appearance
        for i in range(table.rowCount()):
            table.setRowHeight(i, 45)
            
        layout.addWidget(table)
        
        # Add insight summary
        insights = self.generate_insights(company_data)
        if insights:
            insight_label = QLabel(f"💡 Insight: {insights}")
            insight_label.setStyleSheet("""
                background-color: #f0f9ff;
                color: #1e40af;
                padding: 12px;
                border-radius: 6px;
                border-left: 4px solid #3b82f6;
                font-size: 12px;
                margin-top: 8px;
            """)
            insight_label.setWordWrap(True)
            layout.addWidget(insight_label)
            
    def calculate_company_success_rates(self, applications):
        """Calculate success rates for each company"""
        from collections import defaultdict
        
        company_stats = defaultdict(lambda: {'total': 0, 'successful': 0})
        
        for app in applications:
            company = app.get('company', '')
            status = app.get('status', '')
            
            if not company:
                continue
                
            company_stats[company]['total'] += 1
            
            # Consider interviewing and offer as success
            if status in [Application.STATUS_INTERVIEWING, Application.STATUS_OFFER]:
                company_stats[company]['successful'] += 1
        
        # Calculate success rates and create result list
        company_data = []
        for company, stats in company_stats.items():
            if stats['total'] >= 1:  # Only include companies with at least 1 application
                success_rate = (stats['successful'] / stats['total']) * 100
                company_data.append({
                    'name': company,
                    'total': stats['total'],
                    'successful': stats['successful'],
                    'success_rate': success_rate
                })
        
        return company_data
        
    def generate_insights(self, company_data):
        """Generate actionable insights from company success data"""
        if not company_data:
            return ""
            
        # Find best performing company
        best_company = max(company_data, key=lambda x: x['success_rate'])
        
        # Find companies with most applications
        most_applied = max(company_data, key=lambda x: x['total'])
        
        if best_company['success_rate'] > 50:
            return f"Consider targeting companies similar to {best_company['name']} (success rate: {best_company['success_rate']:.0f}%)"
        elif most_applied['total'] >= 3 and most_applied['success_rate'] < 25:
            return f"You've applied to {most_applied['name']} {most_applied['total']} times with low success. Consider a different approach."
        else:
            return "Track more applications to identify successful patterns and improve targeting strategy."

class AnalyticsWidget(QWidget):
    """Analytics dashboard widget"""
    
    def __init__(self, db_manager=None, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager or DatabaseManager()
        self.parent_window = parent
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title section with functional buttons
        title_layout = QHBoxLayout()
        
        icon_label = QLabel("📊")
        icon_label.setStyleSheet("font-size: 24px;")
        
        title_label = QLabel("Analytics Dashboard")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2d3748;")
        
        # Button container
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c5ce7;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 13px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #5641e5;
            }
            QPushButton:pressed {
                background-color: #4834d4;
            }
        """)
        refresh_btn.clicked.connect(self.refresh_analytics)
        refresh_btn.setCursor(Qt.PointingHandCursor)
        
        # Export button
        export_btn = QPushButton("📊 Export Data")
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 13px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        export_btn.clicked.connect(self.export_data)
        export_btn.setCursor(Qt.PointingHandCursor)
        
        # Add New Application button
        new_btn = QPushButton("+ Add Application")
        new_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 13px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #229954;
            }
        """)
        new_btn.clicked.connect(self.open_add_application_dialog)
        new_btn.setCursor(Qt.PointingHandCursor)
        
        button_layout.addWidget(refresh_btn)
        button_layout.addWidget(export_btn)
        button_layout.addWidget(new_btn)
        
        title_layout.addWidget(icon_label)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addLayout(button_layout)
        main_layout.addLayout(title_layout)
        
        # Stats cards - top row
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)
        
        # Get counts from database
        status_counts = self.db_manager.get_status_counts()
        total_applications = sum(status_counts.values())
        interviewing_count = status_counts.get(Application.STATUS_INTERVIEWING, 0)
        offer_count = status_counts.get(Application.STATUS_OFFER, 0)
        
        # Calculate meaningful metrics
        response_rate = int((interviewing_count + offer_count) / total_applications * 100) if total_applications > 0 else 0
        offer_rate = int(offer_count / total_applications * 100) if total_applications > 0 else 0
        
        # Get last application date
        last_applied_date = self.db_manager.get_last_application_date()
        days_since_last_app = 0
        if last_applied_date:
            from datetime import datetime, date
            last_date = datetime.strptime(last_applied_date, "%Y-%m-%d").date()
            today = date.today()
            days_since_last_app = (today - last_date).days
        
        # Get average time metrics
        avg_time_to_response = self.db_manager.get_avg_time_to_response()
        avg_time_to_resolution = self.db_manager.get_avg_time_to_resolution()
        
        # Create stat cards with better metrics
        total_card = StatsCard("Total Applications", total_applications, "")
        response_card = StatsCard("Response Rate", response_rate, "%")
        offer_card = StatsCard("Offer Rate", offer_rate, "%")
        last_applied_card = StatsCard("Days Since Last App", days_since_last_app, "")
        avg_response_card = StatsCard("Avg Days to Response", avg_time_to_response, "")
        avg_resolution_card = StatsCard("Avg Days to Resolution", avg_time_to_resolution, "")
        
        stats_layout.addWidget(total_card)
        stats_layout.addWidget(response_card)
        stats_layout.addWidget(offer_card)
        stats_layout.addWidget(last_applied_card)
        stats_layout.addWidget(avg_response_card)
        stats_layout.addWidget(avg_resolution_card)
        
        main_layout.addLayout(stats_layout)
        
        # Create scrollable area for charts and tables
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(20)
        
        # First row - Enhanced charts
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(16)
        
        # Enhanced Applications Over Time
        applications_frame = self.create_chart_frame("📈 Applications Over Time", EnhancedApplicationsOverTime(self.db_manager))
        
        # Enhanced Activity Chart (replacing basic heatmap)
        activity_frame = self.create_chart_frame("📊 Weekly Application Activity", EnhancedActivityChart(self.db_manager))
        
        row1_layout.addWidget(applications_frame, 1)
        row1_layout.addWidget(activity_frame, 1)
        
        # Second row - Pipeline and Success Rate
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(16)
        
        # Pipeline funnel
        pipeline_frame = self.create_chart_frame("🏗️ Application Pipeline", FunnelChart(self.db_manager))
        
        # Company Success Rate (replaces Follow Ups)
        success_frame = self.create_chart_frame("🎯 Success Rate by Company", CompanySuccessChart(self.db_manager))
        
        row2_layout.addWidget(pipeline_frame, 1)
        row2_layout.addWidget(success_frame, 1)
        
        # Third row - Enhanced Time Analysis and Top Companies
        row3_layout = QHBoxLayout()
        row3_layout.setSpacing(16)
        
        # Enhanced Average Time in Stage
        avgtime_frame = self.create_chart_frame("⏱️ Time Analysis by Stage", EnhancedTimeInStageWidget(self.db_manager))
        
        # Top companies
        companies_frame = self.create_chart_frame("🏢 Top Companies Applied", TopCompaniesTable(self.db_manager))
        
        row3_layout.addWidget(avgtime_frame, 2)
        row3_layout.addWidget(companies_frame, 1)
        
        # Add rows to scroll layout
        scroll_layout.addLayout(row1_layout)
        scroll_layout.addLayout(row2_layout)
        scroll_layout.addLayout(row3_layout)
        scroll_layout.addStretch()
        
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
    def create_chart_frame(self, title, widget):
        """Create a standardized frame for charts"""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                padding: 0px;
            }
        """)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 16, 20, 20)
        layout.setSpacing(16)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #2d3748; margin-bottom: 8px;")
        layout.addWidget(title_label)
        
        # Widget
        layout.addWidget(widget)
        
        return frame
        
    def open_add_application_dialog(self):
        """Open dialog to add a new application"""
        if self.parent_window and hasattr(self.parent_window, 'open_add_application_dialog'):
            # Call the parent's method to open the dialog
            self.parent_window.open_add_application_dialog()
        else:
            # Fallback: create dialog directly
            from ui.add_application import AddApplicationDialog
            dialog = AddApplicationDialog(self)
            if dialog.exec_():
                self.refresh_analytics()
                
    def export_data(self):
        """Export analytics data to CSV"""
        try:
            from datetime import datetime
            import csv
            
            applications = self.db_manager.get_applications()
            if not applications:
                return
                
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"application_analytics_{timestamp}.csv"
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['company', 'role', 'location', 'date_applied', 'status']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for app in applications:
                    writer.writerow({
                        'company': app.get('company', ''),
                        'role': app.get('role', ''),
                        'location': app.get('location', ''),
                        'date_applied': app.get('date_applied', ''),
                        'status': app.get('status', '')
                    })
            
            print(f"Data exported to {filename}")
        except Exception as e:
            print(f"Export error: {e}")
        
    def refresh_analytics(self):
        """Refresh all analytics data"""
        # Re-setup the entire UI with fresh data
        # Clear the current layout
        for i in reversed(range(self.layout().count())):
            child = self.layout().itemAt(i).widget()
            if child:
                child.setParent(None)
                
        # Recreate the UI
        self.setup_ui() 