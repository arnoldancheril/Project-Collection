-- Application Tracker Launcher AppleScript
-- This script creates a proper macOS application launcher

on run
    try
        -- Set the correct path to your Application Tracker project
        set appTrackerPath to "/Users/arnoldancheril/Desktop/apptracker.nosync"
        
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
        display dialog "Error launching Application Tracker: " & errMsg & return & return & "Make sure the app is located at: /Users/arnoldancheril/Desktop/apptracker.nosync" buttons {"OK"} default button "OK" with icon stop
    end try
end run 