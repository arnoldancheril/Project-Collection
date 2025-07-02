#!/bin/bash

# Application Tracker Launcher Script
# This script ensures the app launches from the correct directory with the right Python version

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the application directory
cd "$SCRIPT_DIR"

# Check if python3 is available
if command -v python3 &> /dev/null; then
    echo "Launching Application Tracker..."
    python3 main.py
elif command -v python &> /dev/null; then
    echo "Launching Application Tracker with python..."
    python main.py
else
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python 3 to run the Application Tracker"
    read -p "Press Enter to close..."
    exit 1
fi 