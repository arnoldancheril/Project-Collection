#!/bin/bash
# Job Application Tracker - Launch Script
# Double-click this file to start the application

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Use system python3 or the one in PATH
PYTHON=$(command -v python3 || echo "/usr/bin/python3")

# Install dependencies if needed (first run)
if ! "$PYTHON" -c "import PyQt6" 2>/dev/null; then
    echo "Installing dependencies..."
    "$PYTHON" -m pip install -r requirements.txt --quiet
fi

# Launch the app
exec "$PYTHON" main.py
