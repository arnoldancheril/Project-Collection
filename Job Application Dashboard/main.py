#!/usr/bin/env python3
"""
Application Tracker v3.0
Modern job application tracker with Quick Answers support.
Upgraded to PyQt6.
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from database.db_manager_v2 import DatabaseManager
from ui.main_window import MainWindow


def main():
    # Ensure working directory is the project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    app = QApplication(sys.argv)
    app.setApplicationName("Job Application Tracker")

    # Initialize database
    db = DatabaseManager()
    db.initialize_database()

    # Create and show main window
    window = MainWindow(db)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()