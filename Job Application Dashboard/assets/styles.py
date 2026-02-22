"""
Styles Module
Defines application-wide styling constants and theme settings.
"""

# Colors
COLORS = {
    # Primary brand colors (modern indigo palette)
    "primary": "#4F46E5",          # indigo-600
    "primary_light": "#6366F1",    # indigo-500
    "primary_dark": "#4338CA",     # indigo-700

    # Status colors (tailwind-style palette)
    "Applied": "#3B82F6",          # blue-500
    "Interviewing": "#F59E0B",     # amber-500
    "Offer": "#10B981",            # emerald-500
    "Rejected": "#EF4444",         # red-500

    # UI colors
    "background": "#F7F8FC",       # very light background
    "surface": "#FFFFFF",
    "text_primary": "#1F2937",     # gray-800
    "text_secondary": "#6B7280",   # gray-500
    "border": "#E5E7EB",          # gray-200

    # Interaction colors
    "selection": "#F1F5F9",       # slate-100 (subtle selection)
    "row_hover": "#F8FAFC",       # slate-50

    # Button colors
    "success": "#22C55E",         # green-500
    "danger": "#EF4444",          # red-500
    "warning": "#F59E0B",         # amber-500
    "info": "#3B82F6",            # blue-500
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
        background-color: white;
        color: #495057;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 10px 20px;
        font-weight: 500;
        font-size: 12px;
        min-width: 100px;
    }}
    QPushButton:hover {{
        background-color: #f7fafc;
        border-color: {COLORS['primary']};
        color: {COLORS['primary']};
    }}
    QPushButton:pressed {{
        background-color: #edf2f7;
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
    QMainWindow {{
        background-color: {COLORS['background']};
    }}
    QWidget {{
        font-family: "Arial", "Helvetica", "sans-serif";
        color: {COLORS['text_primary']};
        background-color: {COLORS['background']};
    }}
    
    QLabel {{
        color: {COLORS['text_primary']};
        background-color: transparent;
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