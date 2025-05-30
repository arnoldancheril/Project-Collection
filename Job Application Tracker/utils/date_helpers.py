"""
Date Helper Functions
Utility functions for working with dates in the application.
"""

from datetime import datetime, timedelta

def format_date(date_str, format_str="%Y-%m-%d"):
    """
    Format a date string to a different format
    
    Args:
        date_str (str): Date string in ISO format (YYYY-MM-DD)
        format_str (str): Target format string
        
    Returns:
        str: Formatted date string or original string if conversion fails
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime(format_str)
    except (ValueError, TypeError):
        return date_str
        
def get_date_range(days):
    """
    Get a date range from days ago to today
    
    Args:
        days (int): Number of days in the past
        
    Returns:
        tuple: (from_date, to_date) in ISO format (YYYY-MM-DD)
    """
    today = datetime.now().date()
    from_date = today - timedelta(days=days)
    return from_date.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
    
def days_between(start_date, end_date=None):
    """
    Calculate the number of days between two dates
    
    Args:
        start_date (str): Start date in ISO format (YYYY-MM-DD)
        end_date (str, optional): End date in ISO format. Defaults to today.
        
    Returns:
        int: Number of days between the dates or 0 if conversion fails
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        
        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            end = datetime.now().date()
            
        delta = end - start
        return delta.days
    except (ValueError, TypeError):
        return 0 