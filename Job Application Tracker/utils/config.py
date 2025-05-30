"""
Configuration Module
Manages application configuration and settings.
"""

import os
import json
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    "app_name": "Application Tracker",
    "theme": "default",
    "database_path": "application_tracker.db",
    "window_size": {
        "width": 900,
        "height": 600
    }
}

# Configuration file path
CONFIG_FILE = "config.json"

def load_config():
    """
    Load application configuration from file
    
    Returns:
        dict: Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    # Try to load from file if it exists
    config_path = Path(CONFIG_FILE)
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading config: {e}")
            
    return config

def save_config(config):
    """
    Save configuration to file
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except IOError as e:
        print(f"Error saving config: {e}")
        return False

def get_setting(key, default=None):
    """
    Get a specific setting
    
    Args:
        key (str): Setting key
        default: Default value if setting not found
        
    Returns:
        Setting value or default
    """
    config = load_config()
    return config.get(key, default)

def update_setting(key, value):
    """
    Update a specific setting
    
    Args:
        key (str): Setting key
        value: New value
        
    Returns:
        bool: True if successful, False otherwise
    """
    config = load_config()
    config[key] = value
    return save_config(config) 