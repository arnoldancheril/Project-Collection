#!/usr/bin/env python3
"""
Application Tracker - Main Application
Main entry point for the job application tracker. Initializes the database,
sets up the UI, and starts the application.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from database.db_manager import DatabaseManager
from ui.dashboard import Dashboard
from utils.config import load_config

def main():
    """Main entry point for the application"""
    # Create the application
    app = QApplication(sys.argv)
    app.setApplicationName("Application Tracker")
    
    # Set up database
    db_manager = DatabaseManager()
    db_manager.initialize_database()
    
    # Load configuration
    config = load_config()
    
    # Create and show the main window
    main_window = Dashboard(db_manager)
    main_window.show()
    
    # Start the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 