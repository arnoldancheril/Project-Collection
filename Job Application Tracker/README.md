# Application Tracker

A local GUI application to track job applications with a modern, colorful interface.

## Features

- Track job applications with details like company, role, date applied, and status
- **Smart auto-complete**: Company and role fields with intelligent suggestions based on your application history
- **Predefined software engineering roles**: Quick selection from common software engineering positions
- **Modern filter panel**: Card-based design with icons, real-time search, and intuitive controls
- Modern, colorful UI with status cards and a clean dashboard
- Filter applications by company, status, or date range
- View analytics and statistics about your job search
- Data is stored locally in SQLite database
- Application state is preserved between sessions

## Screenshots

![Application Tracker Dashboard](docs/dashboard_preview.png)

## Requirements

- Python 3.6 or later
- PyQt5
- PyQtChart (for analytics visualizations)

## Installation

1. Clone this repository:
```
git clone https://github.com/arnoldancheril/Project-Collection/tree/main/Job%20Application%20Tracker
cd into the folder
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
python main.py
```

## 🚀 Easy Launcher Options

For convenient access, several launcher options are available:

### 📱 Desktop App (Recommended)
- **`ApplicationTracker.app`** is available on your desktop
- **Double-click** to launch instantly
- Perfect for daily use

### 💻 Terminal Command
```bash
# Install global launcher:
./install_launcher.sh

# Then launch from anywhere:
apptracker
```

### 📋 Direct Script
```bash
# Launch directly:
./launch_app_tracker.sh
```

See **[LAUNCHER_SETUP.md](LAUNCHER_SETUP.md)** for detailed setup instructions.

## Usage

### Dashboard

The main dashboard shows your application stats at a glance:
- Status summary cards (Applied, Interviewing, Offer, Rejected)
- Filter rail on the left for quick filtering
- Table of applications with status indicators

### Adding Applications

Click the "Add Application" button to add a new job application:
- **Company field**: Features auto-complete with your frequently used companies. Start typing to see suggestions, or click the dropdown to browse companies you've applied to before.
- **Role field**: Includes predefined software engineering roles (Software Engineer, Full Stack Developer, etc.) plus your previously used roles. You can select from the dropdown or type a custom role.
- Enter date applied and select application status
- Add optional notes

The auto-complete functionality learns from your application history, showing your most frequently used companies and roles first for faster data entry.

### Filtering

Filter your applications with the modern, card-based filter panel:
- **Company Search**: Real-time search with auto-apply as you type
- **Quick Time Filters**: Modern pill buttons for common time ranges (Past 7 days, Past 30 days, Past 90 days, All time)
- **Status Filter**: Dropdown to filter by application status (Applied, Interviewing, Offer, Rejected)
- **Custom Date Range**: Precise date range selection with calendar popups
- **Modern Design**: Card-based layout with icons, smooth hover effects, and intuitive visual hierarchy
- **Auto-Apply**: Intelligent filtering that updates results as you interact with the controls

### Analytics

The Analytics tab provides insights into your job search:
- Status distribution
- Success rates
- Trends over time

## Development

The application is structured in a modular way:

- `main.py` - Application entry point
- `database/` - Database handling
- `ui/` - User interface components
- `models/` - Data models
- `utils/` - Utility functions
- `assets/` - Styling and resources
