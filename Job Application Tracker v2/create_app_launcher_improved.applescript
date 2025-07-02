-- Improved Application Tracker Launcher AppleScript
-- This script creates a robust macOS application launcher with automatic path detection

on run
    try
        -- Get the current directory where this script is located
        set scriptPath to POSIX path of (path to me)
        set appTrackerPath to do shell script "dirname '" & scriptPath & "'"
        
        -- Alternative paths to check in case the script isn't in the project directory
        set possiblePaths to {appTrackerPath, "/Users/arnoldancheril/Desktop/apptrackerv4.nosync", "/Users/arnoldancheril/Desktop/apptracker.nosync"}
        
        set foundPath to ""
        repeat with currentPath in possiblePaths
            try
                -- Check if main.py exists in this path
                do shell script "test -f '" & currentPath & "/main.py'"
                set foundPath to currentPath
                exit repeat
            on error
                -- Continue to next path
            end try
        end repeat
        
        if foundPath is "" then
            display dialog "Application Tracker not found!" & return & return & "Please make sure the Application Tracker is installed and main.py exists." buttons {"OK"} default button "OK" with icon stop
            return
        end if
        
        -- Check if Python is available
        try
            do shell script "command -v python3"
        on error
            display dialog "Python 3 is not installed!" & return & return & "Please install Python 3 to run the Application Tracker." & return & return & "You can download it from: https://python.org" buttons {"OK"} default button "OK" with icon stop
            return
        end try
        
        -- Launch the application using Terminal with improved error handling
        tell application "Terminal"
            -- Create a new window with a descriptive title
            set newWindow to do script "echo '🚀 Starting Application Tracker...' && cd '" & foundPath & "' && chmod +x launch_app_tracker.sh && ./launch_app_tracker.sh"
            
            -- Set window title
            set custom title of newWindow to "Application Tracker"
            
            -- Bring Terminal to front
            activate
            
            -- Show success notification
            display notification "Application Tracker is starting..." with title "🎯 App Tracker" subtitle "Opening in Terminal"
        end tell
        
    on error errMsg
        -- Show detailed error dialog
        display dialog "Error launching Application Tracker:" & return & return & errMsg & return & return & "Troubleshooting:" & return & "1. Make sure Python 3 is installed" & return & "2. Check that the app files are in the correct location" & return & "3. Try running from Terminal manually" buttons {"OK"} default button "OK" with icon stop
    end try
end run 