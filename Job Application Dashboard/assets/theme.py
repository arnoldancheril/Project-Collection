"""
Theme v2.0 – Modern design system for Application Tracker.
Flat, clean, spacious design with a cohesive color palette.
"""

# ── Color Palette ──────────────────────────────────────────────────────
COLORS = {
    # Brand
    "primary":        "#6366F1",   # indigo-500
    "primary_hover":  "#4F46E5",   # indigo-600
    "primary_light":  "#EEF2FF",   # indigo-50

    # Sidebar
    "sidebar_bg":     "#1E1B4B",   # indigo-950
    "sidebar_text":   "#C7D2FE",   # indigo-200
    "sidebar_active": "#4F46E5",   # indigo-600
    "sidebar_hover":  "#312E81",   # indigo-900

    # Surfaces
    "bg":             "#F8FAFC",   # slate-50
    "surface":        "#FFFFFF",
    "surface_alt":    "#F1F5F9",   # slate-100
    "border":         "#E2E8F0",   # slate-200
    "border_light":   "#F1F5F9",   # slate-100

    # Text
    "text":           "#0F172A",   # slate-900
    "text_secondary": "#64748B",   # slate-500
    "text_muted":     "#94A3B8",   # slate-400

    # Status
    "applied":        "#3B82F6",   # blue-500
    "interviewing":   "#F59E0B",   # amber-500
    "offer":          "#10B981",   # emerald-500
    "rejected":       "#EF4444",   # red-500

    # Accents
    "success":        "#22C55E",
    "warning":        "#F59E0B",
    "danger":         "#EF4444",
    "info":           "#3B82F6",

    # Misc
    "copy_btn":       "#8B5CF6",   # violet-500
    "copy_btn_hover": "#7C3AED",   # violet-600
}

STATUS_COLORS = {
    "Applied":       COLORS["applied"],
    "Interviewing":  COLORS["interviewing"],
    "Offer":         COLORS["offer"],
    "Rejected":      COLORS["rejected"],
}

# ── Font Sizes ─────────────────────────────────────────────────────────
FONT = {
    "xs": 11,
    "sm": 12,
    "md": 13,
    "lg": 15,
    "xl": 18,
    "xxl": 24,
    "hero": 32,
}

RADIUS = {
    "sm": 6,
    "md": 10,
    "lg": 14,
    "xl": 20,
    "pill": 100,
}

# ── Global Stylesheet ─────────────────────────────────────────────────
GLOBAL_STYLE = f"""
* {{
    font-family: "Helvetica Neue", Arial;
}}

QMainWindow {{
    background-color: {COLORS['bg']};
}}

QScrollBar:vertical {{
    border: none;
    background: {COLORS['surface_alt']};
    width: 8px;
    border-radius: 4px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['text_muted']};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS['text_secondary']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    height: 0px;
}}

QToolTip {{
    background-color: {COLORS['text']};
    color: white;
    border: none;
    padding: 6px 10px;
    border-radius: {RADIUS['sm']}px;
    font-size: {FONT['sm']}px;
}}
"""

# ── Component Styles ───────────────────────────────────────────────────

SIDEBAR_STYLE = f"""
#sidebar {{
    background-color: {COLORS['sidebar_bg']};
    min-width: 220px;
    max-width: 220px;
}}
"""

def sidebar_button_style(active=False):
    bg = COLORS["sidebar_active"] if active else "transparent"
    text = "#FFFFFF" if active else COLORS["sidebar_text"]
    weight = "600" if active else "400"
    return f"""
        QPushButton {{
            background-color: {bg};
            color: {text};
            border: none;
            border-radius: {RADIUS['md']}px;
            padding: 12px 16px;
            text-align: left;
            font-size: {FONT['md']}px;
            font-weight: {weight};
        }}
        QPushButton:hover {{
            background-color: {COLORS['sidebar_hover'] if not active else COLORS['sidebar_active']};
            color: #FFFFFF;
        }}
    """

CARD_STYLE = f"""
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: {RADIUS['lg']}px;
    padding: 20px;
"""

def status_badge_style(status):
    color = STATUS_COLORS.get(status, COLORS["primary"])
    return f"""
        background-color: {color};
        color: white;
        border: none;
        border-radius: {RADIUS['pill']}px;
        padding: 4px 14px;
        font-size: {FONT['xs']}px;
        font-weight: 600;
    """

def status_card_style(status):
    color = STATUS_COLORS.get(status, COLORS["primary"])
    return f"""
        background-color: {color};
        border-radius: {RADIUS['lg']}px;
        padding: 20px;
        min-height: 90px;
    """

PRIMARY_BTN = f"""
    QPushButton {{
        background-color: {COLORS['primary']};
        color: white;
        border: none;
        border-radius: {RADIUS['md']}px;
        padding: 10px 22px;
        font-size: {FONT['md']}px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {COLORS['primary_hover']};
    }}
    QPushButton:pressed {{
        background-color: #4338CA;
    }}
"""

SECONDARY_BTN = f"""
    QPushButton {{
        background-color: {COLORS['surface']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        border-radius: {RADIUS['md']}px;
        padding: 10px 22px;
        font-size: {FONT['md']}px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: {COLORS['surface_alt']};
        border-color: {COLORS['primary']};
    }}
"""

DANGER_BTN = f"""
    QPushButton {{
        background-color: {COLORS['danger']};
        color: white;
        border: none;
        border-radius: {RADIUS['md']}px;
        padding: 10px 22px;
        font-size: {FONT['md']}px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: #DC2626;
    }}
"""

SUCCESS_BTN = f"""
    QPushButton {{
        background-color: {COLORS['success']};
        color: white;
        border: none;
        border-radius: {RADIUS['md']}px;
        padding: 10px 22px;
        font-size: {FONT['md']}px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: #16A34A;
    }}
"""

COPY_BTN = f"""
    QPushButton {{
        background-color: {COLORS['copy_btn']};
        color: white;
        border: none;
        border-radius: {RADIUS['sm']}px;
        padding: 6px 16px;
        font-size: {FONT['sm']}px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {COLORS['copy_btn_hover']};
    }}
"""

INPUT_STYLE = f"""
    QLineEdit, QTextEdit {{
        background-color: {COLORS['surface']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        border-radius: {RADIUS['sm']}px;
        padding: 10px 14px;
        font-size: {FONT['md']}px;
        selection-background-color: {COLORS['primary_light']};
    }}
    QLineEdit:focus, QTextEdit:focus {{
        border-color: {COLORS['primary']};
        outline: none;
    }}
    QLineEdit::placeholder {{
        color: {COLORS['text_muted']};
    }}
"""

COMBO_STYLE = f"""
    QComboBox {{
        background-color: {COLORS['surface']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        border-radius: {RADIUS['sm']}px;
        padding: 10px 14px;
        font-size: {FONT['md']}px;
        min-height: 20px;
    }}
    QComboBox:focus {{
        border-color: {COLORS['primary']};
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: right center;
        width: 30px;
        border: none;
    }}
    QComboBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {COLORS['text_secondary']};
    }}
    QComboBox QAbstractItemView {{
        background-color: {COLORS['surface']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        border-radius: {RADIUS['sm']}px;
        selection-background-color: {COLORS['primary_light']};
        selection-color: {COLORS['primary']};
        outline: none;
        padding: 4px;
    }}
"""

DATE_STYLE = f"""
    QDateEdit {{
        background-color: {COLORS['surface']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        border-radius: {RADIUS['sm']}px;
        padding: 10px 14px;
        font-size: {FONT['md']}px;
    }}
    QDateEdit:focus {{
        border-color: {COLORS['primary']};
    }}
    QDateEdit::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: right center;
        width: 30px;
        border: none;
    }}
"""

TABLE_STYLE = f"""
    QTableWidget {{
        background-color: {COLORS['surface']};
        alternate-background-color: {COLORS['surface_alt']};
        border: 1px solid {COLORS['border']};
        border-radius: {RADIUS['md']}px;
        gridline-color: {COLORS['border_light']};
        color: {COLORS['text']};
        font-size: {FONT['md']}px;
        outline: none;
    }}
    QHeaderView::section {{
        background-color: {COLORS['surface_alt']};
        color: {COLORS['text']};
        font-weight: 600;
        font-size: {FONT['sm']}px;
        padding: 12px 16px;
        border: none;
        border-bottom: 2px solid {COLORS['border']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    QTableWidget::item {{
        padding: 10px 16px;
        border-bottom: 1px solid {COLORS['border_light']};
    }}
    QTableWidget::item:selected {{
        background-color: {COLORS['primary_light']};
        color: {COLORS['text']};
    }}
    QTableWidget::item:hover {{
        background-color: {COLORS['surface_alt']};
    }}
"""

SEARCH_STYLE = f"""
    QLineEdit {{
        background-color: {COLORS['surface']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        border-radius: {RADIUS['xl']}px;
        padding: 10px 20px 10px 40px;
        font-size: {FONT['md']}px;
    }}
    QLineEdit:focus {{
        border-color: {COLORS['primary']};
    }}
    QLineEdit::placeholder {{
        color: {COLORS['text_muted']};
    }}
"""

DIALOG_STYLE = f"""
    QDialog {{
        background-color: {COLORS['surface']};
        border-radius: {RADIUS['lg']}px;
    }}
"""

TAB_STYLE = f"""
    QTabWidget::pane {{
        border: none;
        background-color: transparent;
    }}
    QTabBar::tab {{
        background-color: transparent;
        color: {COLORS['text_secondary']};
        padding: 10px 20px;
        margin-right: 4px;
        border: none;
        border-bottom: 2px solid transparent;
        font-size: {FONT['md']}px;
        font-weight: 500;
    }}
    QTabBar::tab:selected {{
        color: {COLORS['primary']};
        border-bottom: 2px solid {COLORS['primary']};
        font-weight: 600;
    }}
    QTabBar::tab:hover:!selected {{
        color: {COLORS['text']};
        border-bottom: 2px solid {COLORS['border']};
    }}
"""
