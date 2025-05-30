"""
Styles Module
Defines application-wide styling constants and theme settings.
"""

# Colors
COLORS = {
    # Primary colors
    "primary": "#6c5ce7",
    "primary_light": "#a29bfe",
    "primary_dark": "#5641e5",
    
    # Status colors
    "applied": "#3498db",       # Blue
    "interviewing": "#1abc9c",  # Teal
    "offer": "#e74c3c",         # Red
    "rejected": "#f39c12",      # Yellow
    
    # UI colors
    "background": "#f8f9fa",
    "surface": "#ffffff",
    "text_primary": "#333333",
    "text_secondary": "#666666",
    "border": "#e0e0e0",
    
    # Button colors
    "success": "#2ecc71",
    "danger": "#e74c3c",
    "warning": "#f39c12",
    "info": "#3498db",
}

# Font sizes
FONT_SIZES = {
    "small": "10px",
    "medium": "12px",
    "large": "14px",
    "x_large": "16px",
    "xx_large": "18px",
    "xxx_large": "24px",
}

# Styling snippets
CARD_STYLE = f"""
    background-color: {COLORS['surface']};
    border-radius: 10px;
    padding: 15px;
    border: 1px solid {COLORS['border']};
"""

# Modern card with hover effects
MODERN_CARD_STYLE = f"""
    background-color: {COLORS['surface']};
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(0, 0, 0, 0.08);
    margin-bottom: 12px;
"""

MODERN_CARD_HOVER_STYLE = f"""
    {MODERN_CARD_STYLE}
    border: 1px solid rgba(108, 92, 231, 0.3);
    background-color: rgba(248, 249, 250, 0.8);
"""

# Modern pill button styles
MODERN_PILL_BUTTON_STYLE = f"""
    QPushButton {{
        background-color: #f8f9fa;
        color: #495057;
        border: 1px solid #e9ecef;
        border-radius: 18px;
        padding: 10px 20px;
        font-weight: 500;
        font-size: 12px;
        min-width: 100px;
    }}
    QPushButton:hover {{
        background-color: #e9ecef;
        border-color: {COLORS['primary']};
        color: {COLORS['primary']};
    }}
    QPushButton:pressed {{
        background-color: #dee2e6;
    }}
"""

MODERN_PRIMARY_BUTTON_STYLE = f"""
    QPushButton {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                  stop:0 {COLORS['primary']}, stop:1 {COLORS['primary_light']});
        color: white;
        border: none;
        border-radius: 20px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 13px;
        min-width: 120px;
    }}
    QPushButton:hover {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                  stop:0 {COLORS['primary_dark']}, stop:1 #9c88ff);
    }}
    QPushButton:pressed {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                  stop:0 #4834d4, stop:1 #8c7ae6);
    }}
"""

BUTTON_STYLE = f"""
    QPushButton {{
        background-color: {COLORS['primary']};
        color: white;
        border-radius: 5px;
        padding: 5px 15px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: {COLORS['primary_dark']};
    }}
"""

SUCCESS_BUTTON_STYLE = f"""
    QPushButton {{
        background-color: {COLORS['success']};
        color: white;
        border-radius: 5px;
        padding: 5px 15px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: #27ae60;
    }}
"""

DANGER_BUTTON_STYLE = f"""
    QPushButton {{
        background-color: {COLORS['danger']};
        color: white;
        border-radius: 5px;
        padding: 5px 15px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: #c0392b;
    }}
"""

# Header with gradient
HEADER_STYLE = f"""
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                              stop:0 {COLORS['primary']}, 
                              stop:1 {COLORS['primary_light']});
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    color: white;
"""

# Input field styling
INPUT_STYLE = """
    QLineEdit, QDateEdit, QComboBox, QTextEdit {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
        background-color: white;
    }
    QLineEdit:focus, QDateEdit:focus, QComboBox:focus, QTextEdit:focus {
        border: 1px solid #6c5ce7;
    }
"""

# Table styling
TABLE_STYLE = """
    QTableWidget {
        background-color: white;
        alternate-background-color: #f5f5f5;
        border: none;
        border-radius: 5px;
        gridline-color: #e0e0e0;
    }
    QHeaderView::section {
        background-color: #f0f0f0;
        padding: 5px;
        border: none;
        border-bottom: 1px solid #ddd;
        font-weight: bold;
    }
    QTableWidget::item {
        padding: 5px;
    }
"""

# Main application style sheet
MAIN_STYLE = f"""
    QWidget {{
        font-family: Arial, sans-serif;
        color: {COLORS['text_primary']};
        background-color: {COLORS['background']};
    }}
    
    QLabel {{
        color: {COLORS['text_primary']};
    }}
    
    {INPUT_STYLE}
    
    {TABLE_STYLE}
"""

def get_status_style(status_color):
    """Generate a style for status badges"""
    return f"""
        background-color: {status_color};
        color: white;
        border-radius: 10px;
        padding: 3px 10px;
        font-size: 12px;
    """

def get_application_style():
    """Get the main application style sheet"""
    return MAIN_STYLE 