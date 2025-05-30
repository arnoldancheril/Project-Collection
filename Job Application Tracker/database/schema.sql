-- Application Tracker Database Schema
-- This file defines the database tables for the job application tracker

-- Applications table - stores information about job applications
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT NOT NULL,
    role TEXT NOT NULL,
    date_applied TEXT NOT NULL,  -- ISO format date: YYYY-MM-DD
    status TEXT NOT NULL,        -- Applied, Interviewing, Offer, Rejected, etc.
    notes TEXT,                  -- Optional notes about the application
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trigger to update the updated_at timestamp when a row is modified
CREATE TRIGGER IF NOT EXISTS update_applications_timestamp 
AFTER UPDATE ON applications
BEGIN
    UPDATE applications SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END; 