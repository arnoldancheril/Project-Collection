# 🚀 Application Tracker Launcher Setup

This guide explains how to set up desktop and terminal launchers for your Application Tracker app.

## 📱 Desktop Launcher (macOS App)

I've created a proper macOS application that you can launch from your desktop!

### ✅ Already Set Up:
- **Desktop App**: `ApplicationTracker.app` is already copied to your desktop
- **Double-click** the app icon on your desktop to launch the Application Tracker

### 🔧 How It Works:
- The app automatically finds and launches your Application Tracker
- It opens in Terminal with the correct Python environment
- No need to remember file paths or commands

## 💻 Terminal Launcher (Global Command)

You can also install a global terminal command to launch the app from anywhere.

### 🛠️ Installation:
```bash
# Run this command in your Application Tracker directory:
./install_launcher.sh
```

### 🎯 Usage After Installation:
```bash
# Launch from anywhere in terminal:
apptracker

# Or use the desktop app:
open ~/Desktop/ApplicationTracker.app
```

## 📋 Quick Start Options

Choose your preferred method:

### Option 1: Desktop Icon (Easiest)
1. **Look for `ApplicationTracker.app` on your desktop**
2. **Double-click** to launch
3. **Done!** The app will open in a Terminal window

### Option 2: Terminal Command (For Power Users)
1. **Run**: `./install_launcher.sh` (requires sudo password)
2. **Use**: `apptracker` from any terminal location
3. **Enjoy**: Launch from anywhere on your system

### Option 3: Direct Script (Manual)
1. **Navigate** to your app directory
2. **Run**: `./launch_app_tracker.sh`
3. **Works** without any installation

## 🔍 File Overview

| File | Purpose |
|------|---------|
| `ApplicationTracker.app` | macOS desktop application (on your desktop) |
| `launch_app_tracker.sh` | Direct launcher script |
| `install_launcher.sh` | Installs global `apptracker` command |
| `create_app_launcher.applescript` | Source code for the desktop app |

## 🛟 Troubleshooting

### Desktop App Not Working?
- **Try**: Right-click → Open (first time only, due to macOS security)
- **Check**: Make sure you're in the correct directory
- **Path Issue**: If you see "no such file or directory", the app has been **fixed** - try the updated version on your desktop

### "no such file or directory: ./launch_app_tracker.sh"?
- **Fixed!** This was a path issue that has been resolved
- **Solution**: The updated `ApplicationTracker.app` on your desktop now points to the correct location
- **Re-download**: If still having issues, the app was recompiled with the correct path

### Terminal Command Not Found?
- **Run**: `./install_launcher.sh` to install the global command
- **Check**: `/usr/local/bin` is in your PATH

### Python Errors?
- **Ensure**: Python 3 is installed (`python3 --version`)
- **Install**: PyQt5 requirements (`pip3 install -r requirements.txt`)

## 🎉 Recommended Setup

For the best experience:

1. **Use the desktop app** for daily use (just double-click)
2. **Install the terminal command** for quick access when coding
3. **Keep both** - they work independently!

The desktop app is perfect for regular use, while the terminal command is great when you're already working in Terminal.

---

**Enjoy your modernized Application Tracker! 🎯** 