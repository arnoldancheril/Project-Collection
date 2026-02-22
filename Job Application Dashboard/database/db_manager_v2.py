"""
Database Manager v2.0
Handles all database operations for the Application Tracker.
Supports applications, quick answer categories, and quick answers.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path


class DatabaseManager:
    """Manages SQLite database connections and CRUD operations."""

    def __init__(self, db_path=None):
        if db_path is None:
            # Store DB next to this file's parent directory
            db_path = str(Path(__file__).parent.parent / "application_tracker.db")
        self.db_path = db_path

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_database(self):
        """Create tables from schema.sql and run migrations."""
        conn = self._connect()
        try:
            schema_path = Path(__file__).parent / "schema.sql"
            with open(schema_path, "r") as f:
                conn.executescript(f.read())
            conn.commit()
            self._run_migrations(conn)
            self._seed_default_categories(conn)
        except sqlite3.Error as e:
            print(f"DB init error: {e}")
        finally:
            conn.close()

    def _run_migrations(self, conn):
        """Add columns that may be missing from an older schema."""
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(applications)")
        cols = {row[1] for row in cursor.fetchall()}
        migrations = {
            "location": "ALTER TABLE applications ADD COLUMN location TEXT DEFAULT ''",
            "job_url": "ALTER TABLE applications ADD COLUMN job_url TEXT DEFAULT ''",
            "salary_range": "ALTER TABLE applications ADD COLUMN salary_range TEXT DEFAULT ''",
        }
        for col, sql in migrations.items():
            if col not in cols:
                try:
                    cursor.execute(sql)
                except sqlite3.Error:
                    pass
        conn.commit()

    def _seed_default_categories(self, conn):
        """Insert default quick-answer categories if the table is empty."""
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM quick_answer_categories")
        if cursor.fetchone()[0] > 0:
            return
        defaults = [
            ("Personal Info", "👤", 1),
            ("Education", "🎓", 2),
            ("Work Experience", "💼", 3),
            ("Skills & Technologies", "🛠️", 4),
            ("Projects", "📂", 5),
            ("Certifications & Clearances", "📜", 6),
            ("Cover Letter Snippets", "✉️", 7),
            ("Behavioral Questions", "🗣️", 8),
            ("Other", "📌", 99),
        ]
        cursor.executemany(
            "INSERT INTO quick_answer_categories (name, icon, sort_order) VALUES (?, ?, ?)",
            defaults,
        )
        conn.commit()

    # ==================================================================
    #  APPLICATION CRUD
    # ==================================================================
    def add_application(self, company, role, date_applied, status,
                        location="", job_url="", salary_range="", notes=""):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO applications
                   (company, role, location, job_url, salary_range, date_applied, status, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (company, role, location, job_url, salary_range, date_applied, status, notes),
            )
            conn.commit()
            return cur.lastrowid
        except sqlite3.Error as e:
            print(f"Error adding application: {e}")
            return None
        finally:
            conn.close()

    def update_application(self, application_id, **kwargs):
        valid = {"company", "role", "location", "job_url", "salary_range",
                 "date_applied", "status", "notes"}
        updates = {k: v for k, v in kwargs.items() if k in valid}
        if not updates:
            return False
        conn = self._connect()
        try:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [application_id]
            conn.execute(
                f"UPDATE applications SET {set_clause} WHERE id = ?", values
            )
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating application: {e}")
            return False
        finally:
            conn.close()

    def delete_application(self, application_id):
        conn = self._connect()
        try:
            conn.execute("DELETE FROM applications WHERE id = ?", (application_id,))
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error deleting application: {e}")
            return False
        finally:
            conn.close()

    def get_applications(self, filters=None):
        conn = self._connect()
        try:
            query = "SELECT * FROM applications"
            params = []
            if filters:
                clauses = []
                if filters.get("company"):
                    clauses.append("company LIKE ?")
                    params.append(f"%{filters['company']}%")
                if filters.get("role"):
                    clauses.append("role LIKE ?")
                    params.append(f"%{filters['role']}%")
                if filters.get("status") and filters["status"] != "All":
                    clauses.append("status = ?")
                    params.append(filters["status"])
                if filters.get("date_from") and filters.get("date_to"):
                    clauses.append("date_applied BETWEEN ? AND ?")
                    params.extend([filters["date_from"], filters["date_to"]])
                if clauses:
                    query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY date_applied DESC"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as e:
            print(f"Error fetching applications: {e}")
            return []
        finally:
            conn.close()

    def get_application(self, application_id):
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM applications WHERE id = ?", (application_id,)
            ).fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return None
        finally:
            conn.close()

    def get_status_counts(self):
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT status, COUNT(*) as count FROM applications GROUP BY status"
            ).fetchall()
            return {r["status"]: r["count"] for r in rows}
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return {}
        finally:
            conn.close()

    def get_total_count(self):
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) as c FROM applications").fetchone()
            return row["c"] if row else 0
        except sqlite3.Error:
            return 0
        finally:
            conn.close()

    def get_recent_applications(self, limit=10):
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM applications ORDER BY date_applied DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def get_applications_by_month(self):
        """Return list of (month_str, count) for charting."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT strftime('%Y-%m', date_applied) as month, COUNT(*) as count
                   FROM applications GROUP BY month ORDER BY month"""
            ).fetchall()
            return [(r["month"], r["count"]) for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def get_unique_companies(self):
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT company FROM applications ORDER BY company"
            ).fetchall()
            return [r["company"] for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def get_unique_roles(self):
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT role FROM applications ORDER BY role"
            ).fetchall()
            return [r["role"] for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    # ==================================================================
    #  QUICK ANSWER CATEGORIES
    # ==================================================================
    def get_categories(self):
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM quick_answer_categories ORDER BY sort_order, name"
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return []
        finally:
            conn.close()

    def add_category(self, name, icon="📁", sort_order=0):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO quick_answer_categories (name, icon, sort_order) VALUES (?, ?, ?)",
                (name, icon, sort_order),
            )
            conn.commit()
            return cur.lastrowid
        except sqlite3.Error as e:
            print(f"Error adding category: {e}")
            return None
        finally:
            conn.close()

    def update_category(self, category_id, **kwargs):
        valid = {"name", "icon", "sort_order"}
        updates = {k: v for k, v in kwargs.items() if k in valid}
        if not updates:
            return False
        conn = self._connect()
        try:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [category_id]
            conn.execute(
                f"UPDATE quick_answer_categories SET {set_clause} WHERE id = ?", values
            )
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return False
        finally:
            conn.close()

    def delete_category(self, category_id):
        conn = self._connect()
        try:
            conn.execute("DELETE FROM quick_answer_categories WHERE id = ?", (category_id,))
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return False
        finally:
            conn.close()

    # ==================================================================
    #  QUICK ANSWERS
    # ==================================================================
    def get_quick_answers(self, category_id=None):
        conn = self._connect()
        try:
            if category_id:
                rows = conn.execute(
                    """SELECT qa.*, qac.name as category_name, qac.icon as category_icon
                       FROM quick_answers qa
                       JOIN quick_answer_categories qac ON qa.category_id = qac.id
                       WHERE qa.category_id = ?
                       ORDER BY qa.sort_order, qa.question""",
                    (category_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT qa.*, qac.name as category_name, qac.icon as category_icon
                       FROM quick_answers qa
                       JOIN quick_answer_categories qac ON qa.category_id = qac.id
                       ORDER BY qac.sort_order, qa.sort_order, qa.question"""
                ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return []
        finally:
            conn.close()

    def add_quick_answer(self, category_id, question, answer, sort_order=0):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO quick_answers (category_id, question, answer, sort_order)
                   VALUES (?, ?, ?, ?)""",
                (category_id, question, answer, sort_order),
            )
            conn.commit()
            return cur.lastrowid
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return None
        finally:
            conn.close()

    def update_quick_answer(self, answer_id, **kwargs):
        valid = {"category_id", "question", "answer", "sort_order"}
        updates = {k: v for k, v in kwargs.items() if k in valid}
        if not updates:
            return False
        conn = self._connect()
        try:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [answer_id]
            conn.execute(
                f"UPDATE quick_answers SET {set_clause} WHERE id = ?", values
            )
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return False
        finally:
            conn.close()

    def delete_quick_answer(self, answer_id):
        conn = self._connect()
        try:
            conn.execute("DELETE FROM quick_answers WHERE id = ?", (answer_id,))
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return False
        finally:
            conn.close()

    def increment_copy_count(self, answer_id):
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE quick_answers SET times_copied = times_copied + 1 WHERE id = ?",
                (answer_id,),
            )
            conn.commit()
        except sqlite3.Error:
            pass
        finally:
            conn.close()

    def search_quick_answers(self, query):
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT qa.*, qac.name as category_name, qac.icon as category_icon
                   FROM quick_answers qa
                   JOIN quick_answer_categories qac ON qa.category_id = qac.id
                   WHERE qa.question LIKE ? OR qa.answer LIKE ?
                   ORDER BY qa.times_copied DESC, qa.question""",
                (f"%{query}%", f"%{query}%"),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    # ==================================================================
    #  FREQUENCY-BASED QUERIES (for autocomplete)
    # ==================================================================
    def get_companies_by_frequency(self):
        """Return companies sorted by frequency (most applied first)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT company, COUNT(*) as cnt
                   FROM applications
                   GROUP BY company
                   ORDER BY cnt DESC, company"""
            ).fetchall()
            return [r["company"] for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def get_roles_by_frequency(self):
        """Return roles sorted by frequency (most applied first)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT role, COUNT(*) as cnt
                   FROM applications
                   GROUP BY role
                   ORDER BY cnt DESC, role"""
            ).fetchall()
            return [r["role"] for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def get_locations_by_frequency(self):
        """Return locations sorted by frequency (most used first)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT location, COUNT(*) as cnt
                   FROM applications
                   WHERE location IS NOT NULL AND location != ''
                   GROUP BY location
                   ORDER BY cnt DESC, location"""
            ).fetchall()
            return [r["location"] for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def get_applications_for_export(self, date_from=None, date_to=None):
        """Get applications filtered by date range for CSV/clipboard export."""
        conn = self._connect()
        try:
            query = "SELECT company, role, location, date_applied, status, job_url, notes FROM applications"
            params = []
            if date_from and date_to:
                query += " WHERE date_applied BETWEEN ? AND ?"
                params = [date_from, date_to]
            elif date_from:
                query += " WHERE date_applied >= ?"
                params = [date_from]
            elif date_to:
                query += " WHERE date_applied <= ?"
                params = [date_to]
            query += " ORDER BY date_applied DESC"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def get_weekly_stats(self):
        """Get application count per week for the last 12 weeks."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT strftime('%Y-W%W', date_applied) as week,
                          COUNT(*) as count
                   FROM applications
                   WHERE date_applied >= date('now', '-84 days')
                   GROUP BY week ORDER BY week"""
            ).fetchall()
            return [(r["week"], r["count"]) for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def get_daily_stats(self, days=30):
        """Get application count per day for the last N days."""
        conn = self._connect()
        try:
            rows = conn.execute(
                f"""SELECT date_applied, COUNT(*) as count
                    FROM applications
                    WHERE date_applied >= date('now', '-{days} days')
                    GROUP BY date_applied ORDER BY date_applied""",
            ).fetchall()
            return [(r["date_applied"], r["count"]) for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()
