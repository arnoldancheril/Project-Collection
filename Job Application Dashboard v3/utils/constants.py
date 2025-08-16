"""
Constants Module
Application-wide constants and enumerations.
"""

# Application version
APP_VERSION = "1.0.0"

# Date formats
DATE_FORMAT_ISO = "%Y-%m-%d"
DATE_FORMAT_DISPLAY = "%b %d, %Y"  # e.g. Jan 15, 2023

# Filter types
FILTER_ALL = "All"
FILTER_COMPANY = "Company"
FILTER_STATUS = "Status"
FILTER_DATE_RANGE = "Date Range"

# Time filters
TIME_FILTER_7_DAYS = "Past 7 days"
TIME_FILTER_30_DAYS = "Past 30 days"
TIME_FILTER_90_DAYS = "Past 90 days"
TIME_FILTER_CUSTOM = "Custom range"

# Table columns
COLUMN_COMPANY = 0
COLUMN_ROLE = 1
COLUMN_DATE_APPLIED = 2
COLUMN_STATUS = 3
COLUMN_ACTIONS = 4

# Dialog results
RESULT_ADDED = 1
RESULT_UPDATED = 2
RESULT_DELETED = 3
RESULT_CANCELLED = 0 