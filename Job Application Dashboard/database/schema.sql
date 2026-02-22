-- Application Tracker Database Schema v2.0
-- Complete redesign with Quick Answers support

-- Applications table - stores information about job applications
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT NOT NULL,
    role TEXT NOT NULL,
    location TEXT DEFAULT '',
    job_url TEXT DEFAULT '',
    salary_range TEXT DEFAULT '',
    date_applied TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'Applied',
    notes TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Quick Answer Categories (Education, Work Experience, Skills, etc.)
CREATE TABLE IF NOT EXISTS quick_answer_categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    icon TEXT DEFAULT '📁',
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Quick Answers - question/answer pairs organized by category
CREATE TABLE IF NOT EXISTS quick_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sort_order INTEGER DEFAULT 0,
    times_copied INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES quick_answer_categories(id) ON DELETE CASCADE
);

-- Trigger to update the updated_at timestamp when applications are modified
CREATE TRIGGER IF NOT EXISTS update_applications_timestamp 
AFTER UPDATE ON applications
BEGIN
    UPDATE applications SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Trigger to update the updated_at timestamp when quick_answers are modified
CREATE TRIGGER IF NOT EXISTS update_quick_answers_timestamp 
AFTER UPDATE ON quick_answers
BEGIN
    UPDATE quick_answers SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;