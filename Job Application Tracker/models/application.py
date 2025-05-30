"""
Application Model
Defines the Application class representing a job application entry.
"""

from datetime import datetime

class Application:
    """Represents a job application entry"""
    
    STATUS_APPLIED = "Applied"
    STATUS_INTERVIEWING = "Interviewing"
    STATUS_OFFER = "Offer"
    STATUS_REJECTED = "Rejected"
    
    # List of valid statuses
    VALID_STATUSES = [
        STATUS_APPLIED,
        STATUS_INTERVIEWING,
        STATUS_OFFER,
        STATUS_REJECTED
    ]
    
    # Status colors
    STATUS_COLORS = {
        STATUS_APPLIED: "#3498db",      # Blue
        STATUS_INTERVIEWING: "#1abc9c",  # Teal
        STATUS_OFFER: "#e74c3c",        # Red
        STATUS_REJECTED: "#f39c12",     # Yellow
    }
    
    def __init__(self, company, role, date_applied, status, notes=None, id=None, 
                 created_at=None, updated_at=None):
        """
        Initialize an Application object
        
        Args:
            company (str): Company name
            role (str): Job role/title
            date_applied (str): Date applied in YYYY-MM-DD format
            status (str): Application status
            notes (str, optional): Additional notes
            id (int, optional): Database ID
            created_at (str, optional): Creation timestamp
            updated_at (str, optional): Last update timestamp
        """
        self.id = id
        self.company = company
        self.role = role
        self.date_applied = date_applied
        self.status = status if status in self.VALID_STATUSES else self.STATUS_APPLIED
        self.notes = notes
        self.created_at = created_at
        self.updated_at = updated_at
        
    @classmethod
    def from_dict(cls, data):
        """Create an Application instance from a dictionary"""
        return cls(
            company=data.get('company'),
            role=data.get('role'),
            date_applied=data.get('date_applied'),
            status=data.get('status'),
            notes=data.get('notes'),
            id=data.get('id'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )
        
    def to_dict(self):
        """Convert Application to dictionary"""
        return {
            'id': self.id,
            'company': self.company,
            'role': self.role,
            'date_applied': self.date_applied,
            'status': self.status,
            'notes': self.notes,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
        
    def get_status_color(self):
        """Get the color associated with the application's status"""
        return self.STATUS_COLORS.get(self.status, "#999999")  # Default gray if status not found
        
    def days_since_applied(self):
        """Calculate days since application was submitted"""
        if not self.date_applied:
            return 0
            
        applied_date = datetime.strptime(self.date_applied, "%Y-%m-%d").date()
        today = datetime.now().date()
        delta = today - applied_date
        return delta.days 