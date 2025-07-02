#!/usr/bin/env python3
"""
Test script to verify the application launches without errors
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer
from database.db_manager import DatabaseManager
from ui.dashboard import Dashboard

def test_app():
    """Test the application launch"""
    print("Testing Application Tracker...")
    
    # Create the application
    app = QApplication(sys.argv)
    app.setApplicationName("Application Tracker Test")
    
    # Set up database
    db_manager = DatabaseManager()
    db_manager.initialize_database()
    print("Database initialized successfully")
    
    # Create and show the main window
    main_window = Dashboard(db_manager)
    main_window.show()
    print("Main window created and shown")
    
    # Switch to analytics tab to test it
    main_window.tab_widget.setCurrentIndex(1)
    print("Switched to analytics tab")
    
    # Close after 3 seconds for testing
    QTimer.singleShot(3000, app.quit)
    
    # Start the application
    result = app.exec_()
    print(f"Application closed with result: {result}")
    return result

if __name__ == "__main__":
    test_app() 