-- Application Tracker Dynamic Launcher AppleScript
-- This script finds the Application Tracker directory dynamically

on run
    try
        -- Try to find the Application Tracker directory
        set appTrackerPath to ""
        
        -- Check common locations
        set possiblePaths to {"/Users/arnoldancheril/Desktop/apptracker.nosync", "/Users/arnoldancheril/Applications/apptracker.nosync", "/Applications/apptracker"}
        
        repeat with currentPath in possiblePaths
            try
                tell application "System Events"
                    if exists folder currentPath then
                        if exists file (currentPath & "/main.py") then
                            set appTrackerPath to currentPath
                            exit repeat
                        end if
                    end if
                end tell
            end try
        end repeat
        
        -- If not found in common locations, try to find it
        if appTrackerPath = "" then
            try
                set findResult to do shell script "find /Users/arnoldancheril -name 'main.py' -path '*/apptracker*' -type f 2>/dev/null | head -1"
                if findResult ≠ "" then
                    set appTrackerPath to do shell script "dirname '" & findResult & "'"
                end if
            end try
        end if
        
        -- If still not found, use default
        if appTrackerPath = "" then
            set appTrackerPath to "/Users/arnoldancheril/Desktop/apptracker.nosync"
        end if
        
        -- Launch the application using Terminal
        tell application "Terminal"
            -- Check if Terminal is already running
            if not (exists window 1) then
                -- Open a new window if none exists
                do script "cd '" & appTrackerPath & "' && ./launch_app_tracker.sh"
            else
                -- Use existing window
                do script "cd '" & appTrackerPath & "' && ./launch_app_tracker.sh" in window 1
            end if
            
            -- Bring Terminal to front
            activate
        end tell
        
    on error errMsg
        -- Show error dialog if something goes wrong
        display dialog "Error launching Application Tracker: " & errMsg & return & return & "Please make sure the Application Tracker is installed properly." buttons {"OK"} default button "OK" with icon stop
    end try
end run 