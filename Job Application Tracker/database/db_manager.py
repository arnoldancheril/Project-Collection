"""
Database Manager
Handles database connection, initialization, and CRUD operations for the application tracker.
Uses SQLite for local storage of job application data.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, db_path="application_tracker.db"):
        """Initialize the database manager with the path to the SQLite database file"""
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        
    def connect(self):
        """Create a connection to the SQLite database"""
        self.connection = sqlite3.connect(self.db_path)
        # Enable foreign keys
        self.connection.execute("PRAGMA foreign_keys = ON")
        # Use Row objects for query results
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()
        return self.connection
        
    def close(self):
        """Close the database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.cursor = None
            
    def initialize_database(self):
        """Create database tables if they don't exist"""
        try:
            connection = self.connect()
            
            # Read schema file
            schema_path = Path(__file__).parent / "schema.sql"
            with open(schema_path, "r") as schema_file:
                schema_sql = schema_file.read()
                
            # Execute schema
            connection.executescript(schema_sql)
            connection.commit()
            
            print("Database initialized successfully")
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
        finally:
            self.close()
            
    # Application CRUD operations
    def add_application(self, company, role, date_applied, status):
        """Add a new job application to the database"""
        try:
            connection = self.connect()
            cursor = connection.cursor()
            
            cursor.execute(
                """
                INSERT INTO applications (company, role, date_applied, status)
                VALUES (?, ?, ?, ?)
                """,
                (company, role, date_applied, status)
            )
            
            connection.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error adding application: {e}")
            return None
        finally:
            self.close()
            
    def update_application(self, application_id, **kwargs):
        """Update an existing job application"""
        valid_fields = ['company', 'role', 'date_applied', 'status', 'notes']
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        
        if not updates:
            return False
            
        try:
            connection = self.connect()
            cursor = connection.cursor()
            
            set_clause = ", ".join([f"{field} = ?" for field in updates.keys()])
            values = list(updates.values())
            values.append(application_id)
            
            cursor.execute(
                f"""
                UPDATE applications
                SET {set_clause}
                WHERE id = ?
                """,
                values
            )
            
            connection.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Error updating application: {e}")
            return False
        finally:
            self.close()
            
    def get_applications(self, filters=None):
        """
        Get applications with optional filtering
        
        Args:
            filters: Dictionary with filter conditions
                    e.g. {'company': 'Google', 'status': 'Applied'}
        """
        try:
            connection = self.connect()
            cursor = connection.cursor()
            
            query = "SELECT * FROM applications"
            params = []
            
            if filters and len(filters) > 0:
                where_clauses = []
                
                if 'company' in filters:
                    where_clauses.append("company LIKE ?")
                    params.append(f"%{filters['company']}%")
                    
                if 'role' in filters:
                    where_clauses.append("role LIKE ?")
                    params.append(f"%{filters['role']}%")
                    
                if 'status' in filters:
                    where_clauses.append("status = ?")
                    params.append(filters['status'])
                    
                if 'date_from' in filters and 'date_to' in filters:
                    where_clauses.append("date_applied BETWEEN ? AND ?")
                    params.append(filters['date_from'])
                    params.append(filters['date_to'])
                    
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
            
            query += " ORDER BY date_applied DESC"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error fetching applications: {e}")
            return []
        finally:
            self.close()
            
    def delete_application(self, application_id):
        """Delete a job application from the database"""
        try:
            connection = self.connect()
            cursor = connection.cursor()
            
            cursor.execute(
                "DELETE FROM applications WHERE id = ?",
                (application_id,)
            )
            
            connection.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Error deleting application: {e}")
            return False
        finally:
            self.close()
            
    def get_status_counts(self):
        """Get counts of applications by status"""
        try:
            connection = self.connect()
            cursor = connection.cursor()
            
            cursor.execute(
                """
                SELECT status, COUNT(*) as count
                FROM applications
                GROUP BY status
                """
            )
            
            return {row['status']: row['count'] for row in cursor.fetchall()}
        except sqlite3.Error as e:
            print(f"Error getting status counts: {e}")
            return {}
        finally:
            self.close()
    
    def get_frequent_companies(self, limit=10):
        """Get frequently used companies ordered by usage count"""
        try:
            self.connect()
            cursor = self.cursor
            
            cursor.execute(
                """
                SELECT company, COUNT(*) as usage_count
                FROM applications
                GROUP BY LOWER(company)
                ORDER BY usage_count DESC, company ASC
                LIMIT ?
                """,
                (limit,)
            )
            
            return [row['company'] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error getting frequent companies: {e}")
            return []
        finally:
            self.close()
    
    def get_frequent_roles(self, limit=10):
        """Get frequently used roles ordered by usage count"""
        try:
            self.connect()
            cursor = self.cursor
            
            cursor.execute(
                """
                SELECT role, COUNT(*) as usage_count
                FROM applications
                GROUP BY LOWER(role)
                ORDER BY usage_count DESC, role ASC
                LIMIT ?
                """,
                (limit,)
            )
            
            return [row['role'] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error getting frequent roles: {e}")
            return []
        finally:
            self.close()
    
    def get_all_companies(self):
        """Get all unique companies for auto-complete"""
        try:
            self.connect()
            cursor = self.cursor
            
            cursor.execute(
                """
                SELECT DISTINCT company
                FROM applications
                ORDER BY company ASC
                """
            )
            
            return [row['company'] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error getting all companies: {e}")
            return []
        finally:
            self.close()
    
    def get_all_roles(self):
        """Get all unique roles for auto-complete"""
        try:
            self.connect()
            cursor = self.cursor
            
            cursor.execute(
                """
                SELECT DISTINCT role
                FROM applications
                ORDER BY role ASC
                """
            )
            
            return [row['role'] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error getting all roles: {e}")
            return []
        finally:
            self.close()
            
    def get_company_counts(self, limit=6):
        """Get counts of applications by company (for analytics dashboard)"""
        try:
            connection = self.connect()
            cursor = connection.cursor()
            
            cursor.execute(
                """
                SELECT company, COUNT(*) as count
                FROM applications
                GROUP BY company
                ORDER BY count DESC, company ASC
                LIMIT ?
                """,
                (limit,)
            )
            
            return [(row['company'], row['count']) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error getting company counts: {e}")
            return []
        finally:
            self.close()
            
    def get_last_application_date(self):
        """Get the date of the most recent application"""
        try:
            connection = self.connect()
            cursor = connection.cursor()
            
            cursor.execute(
                """
                SELECT date_applied
                FROM applications
                ORDER BY date_applied DESC
                LIMIT 1
                """
            )
            
            result = cursor.fetchone()
            return result['date_applied'] if result else None
        except sqlite3.Error as e:
            print(f"Error getting last application date: {e}")
            return None
        finally:
            self.close() 