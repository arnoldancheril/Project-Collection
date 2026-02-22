#!/bin/bash

# 🚀 Application Tracker Desktop Launcher Creator
# Creates a beautiful macOS app bundle that can be double-clicked to launch the tracker

echo "🎯 Creating Application Tracker Desktop Launcher..."

# Get the current directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="Application Tracker"
APP_BUNDLE="${APP_NAME}.app"
DESKTOP_PATH="$HOME/Desktop"

# Remove existing app if it exists
if [ -d "$DESKTOP_PATH/$APP_BUNDLE" ]; then
    echo "🗑️  Removing existing app..."
    rm -rf "$DESKTOP_PATH/$APP_BUNDLE"
fi

# Create the app bundle structure
echo "📁 Creating app bundle structure..."
mkdir -p "$DESKTOP_PATH/$APP_BUNDLE/Contents/MacOS"
mkdir -p "$DESKTOP_PATH/$APP_BUNDLE/Contents/Resources"

# Create the Info.plist file
echo "📄 Creating Info.plist..."
cat > "$DESKTOP_PATH/$APP_BUNDLE/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>Application Tracker</string>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundleIconFile</key>
    <string>app_icon</string>
    <key>CFBundleIdentifier</key>
    <string>com.apptracker.main</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Application Tracker</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>2.0</string>
    <key>CFBundleVersion</key>
    <string>2024.1</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.productivity</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.12</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSHumanReadableCopyright</key>
    <string>© 2024 Application Tracker</string>
</dict>
</plist>
EOF

# Create the launcher executable script
echo "🔧 Creating launcher executable..."
cat > "$DESKTOP_PATH/$APP_BUNDLE/Contents/MacOS/launcher" << EOF
#!/bin/bash

# Application Tracker Launcher
# This script launches the Python application from the desktop

# Set up environment
export PATH="/usr/local/bin:/usr/bin:/bin:\$PATH"

# Find the application directory
APP_DIR="$CURRENT_DIR"

# Alternative search paths
SEARCH_PATHS=(
    "$CURRENT_DIR"
    "\$HOME/Desktop/apptrackerv4.nosync"
    "\$HOME/Desktop/apptracker.nosync"
    "\$HOME/Documents/apptrackerv4.nosync"
    "\$HOME/Applications/apptrackerv4.nosync"
)

FOUND_PATH=""
for path in "\${SEARCH_PATHS[@]}"; do
    if [ -f "\$path/main.py" ]; then
        FOUND_PATH="\$path"
        break
    fi
done

if [ -z "\$FOUND_PATH" ]; then
    osascript -e 'display alert "Application Tracker Not Found" message "Could not locate the Application Tracker files. Please make sure it is installed in your Desktop folder." as critical'
    exit 1
fi

# Check Python installation
if ! command -v python3 &> /dev/null; then
    osascript -e 'display alert "Python 3 Required" message "Python 3 is not installed. Please install Python 3 from python.org to run the Application Tracker." as critical'
    open "https://www.python.org/downloads/"
    exit 1
fi

# Check PyQt5 installation
cd "\$FOUND_PATH"
if ! python3 -c "import PyQt5" 2>/dev/null; then
    osascript -e 'display notification "Installing required packages..." with title "Application Tracker" subtitle "First time setup..."'
    
    # Try to install requirements
    if [ -f "requirements.txt" ]; then
        python3 -m pip install -r requirements.txt
    else
        python3 -m pip install PyQt5 matplotlib
    fi
fi

# Launch the application
osascript -e 'display notification "Starting Application Tracker..." with title "🎯 App Tracker" subtitle "Opening application..."'

# Launch in Terminal with a nice title
osascript -e "
tell application \"Terminal\"
    set newWindow to do script \"cd '\$FOUND_PATH' && echo '🎯 Application Tracker' && echo '=================' && python3 main.py\"
    set custom title of newWindow to \"Application Tracker\"
    activate
end tell
"
EOF

# Make the launcher executable
chmod +x "$DESKTOP_PATH/$APP_BUNDLE/Contents/MacOS/launcher"

# Copy or create an app icon
echo "🎨 Setting up app icon..."
if [ -f "$CURRENT_DIR/assets/icons/app_icon.icns" ]; then
    cp "$CURRENT_DIR/assets/icons/app_icon.icns" "$DESKTOP_PATH/$APP_BUNDLE/Contents/Resources/app_icon.icns"
elif [ -d "$CURRENT_DIR/Application Tracker.app/Contents/Resources" ]; then
    # Copy from existing app if available
    cp "$CURRENT_DIR/Application Tracker.app/Contents/Resources/app_icon.icns" "$DESKTOP_PATH/$APP_BUNDLE/Contents/Resources/app_icon.icns" 2>/dev/null || true
else
    # Create a simple app icon using SF Symbols or system icons
    echo "📱 Creating default app icon..."
    # This creates a simple icon - you can replace with a custom design
    /usr/bin/python3 -c "
import os
# Create a basic icon using available system tools
os.system('sips -s format icns /System/Library/CoreServices/CoreTypes.bundle/Contents/Resources/ToolbarApplicationsFolderIcon.icns --out \"$DESKTOP_PATH/$APP_BUNDLE/Contents/Resources/app_icon.icns\" 2>/dev/null || echo \"Using default icon\"')
" || echo "Will use default system icon"
fi

# Set proper permissions
echo "🔐 Setting permissions..."
chmod -R 755 "$DESKTOP_PATH/$APP_BUNDLE"

# Clear any quarantine attributes (for security)
xattr -cr "$DESKTOP_PATH/$APP_BUNDLE" 2>/dev/null || true

echo ""
echo "✅ Desktop Launcher Created Successfully!"
echo ""
echo "📍 Location: $DESKTOP_PATH/$APP_BUNDLE"
echo ""
echo "🚀 How to use:"
echo "   1. Look for 'Application Tracker' on your Desktop"
echo "   2. Double-click the icon to launch"
echo "   3. The app will open in Terminal automatically"
echo ""
echo "💡 Note: On first launch, macOS may ask for permission to run the app."
echo "   If so, right-click the app and select 'Open' to bypass security."
echo ""
echo "🎉 Enjoy your Application Tracker!" 