import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import Calendar, DateEntry
import datetime
import sqlite3
import bcrypt
import threading
import time
import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plyer import notification

class CollegeScheduler(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("College Scheduler")
        self.geometry("1200x800")
        self.style = ttk.Style(self)
        self.style.theme_use('clam')  # Use a modern theme

        # Initialize Database
        self.conn = sqlite3.connect('scheduler.db')
        self.create_tables()

        # User Authentication
        self.current_user = None
        self.show_login_screen()

    # ---------------------- Database Methods ---------------------- #
    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                task TEXT,
                due_date DATE,
                priority TEXT,
                completed INTEGER DEFAULT 0,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                title TEXT,
                event_date DATE,
                event_type TEXT,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
        ''')
        self.conn.commit()

    # ---------------------- User Authentication ---------------------- #
    def show_login_screen(self):
        self.clear_window()

        login_frame = ttk.Frame(self)
        login_frame.pack(pady=100)

        ttk.Label(login_frame, text="Login", font=("Helvetica", 24)).grid(row=0, column=0, columnspan=2, pady=20)

        ttk.Label(login_frame, text="Username:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)
        self.login_username_entry = ttk.Entry(login_frame)
        self.login_username_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(login_frame, text="Password:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.E)
        self.login_password_entry = ttk.Entry(login_frame, show='*')
        self.login_password_entry.grid(row=2, column=1, padx=5, pady=5)

        login_btn = ttk.Button(login_frame, text="Login", command=self.login)
        login_btn.grid(row=3, column=0, columnspan=2, pady=10)

        register_btn = ttk.Button(login_frame, text="Register", command=self.show_registration_screen)
        register_btn.grid(row=4, column=0, columnspan=2)

    def show_registration_screen(self):
        self.clear_window()

        register_frame = ttk.Frame(self)
        register_frame.pack(pady=100)

        ttk.Label(register_frame, text="Register", font=("Helvetica", 24)).grid(row=0, column=0, columnspan=2, pady=20)

        ttk.Label(register_frame, text="Username:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)
        self.register_username_entry = ttk.Entry(register_frame)
        self.register_username_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(register_frame, text="Password:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.E)
        self.register_password_entry = ttk.Entry(register_frame, show='*')
        self.register_password_entry.grid(row=2, column=1, padx=5, pady=5)

        register_btn = ttk.Button(register_frame, text="Register", command=self.register)
        register_btn.grid(row=3, column=0, columnspan=2, pady=10)

        back_btn = ttk.Button(register_frame, text="Back to Login", command=self.show_login_screen)
        back_btn.grid(row=4, column=0, columnspan=2)

    def login(self):
        username = self.login_username_entry.get()
        password = self.login_password_entry.get().encode('utf-8')

        cursor = self.conn.cursor()
        cursor.execute("SELECT user_id, password_hash FROM users WHERE username=?", (username,))
        result = cursor.fetchone()

        if result and bcrypt.checkpw(password, result[1]):
            self.current_user = result[0]
            self.show_main_application()
            threading.Thread(target=self.notification_worker, daemon=True).start()
        else:
            messagebox.showerror("Login Failed", "Invalid username or password.")

    def register(self):
        username = self.register_username_entry.get()
        password = self.register_password_entry.get().encode('utf-8')
        password_hash = bcrypt.hashpw(password, bcrypt.gensalt())

        cursor = self.conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
            self.conn.commit()
            messagebox.showinfo("Registration Successful", "You can now log in.")
            self.show_login_screen()
        except sqlite3.IntegrityError:
            messagebox.showerror("Registration Failed", "Username already exists.")

    # ---------------------- Main Application ---------------------- #
    def show_main_application(self):
        self.clear_window()

        # Create Notebook for Tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create Frames for Tabs
        self.scheduler_frame = ttk.Frame(self.notebook)
        self.tasks_frame = ttk.Frame(self.notebook)
        self.analytics_frame = ttk.Frame(self.notebook)
        self.help_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.scheduler_frame, text='Scheduler')
        self.notebook.add(self.tasks_frame, text='Tasks')
        self.notebook.add(self.analytics_frame, text='Analytics')
        self.notebook.add(self.help_frame, text='Help')

        self.create_scheduler_tab()
        self.create_tasks_tab()
        self.create_analytics_tab()
        self.create_help_tab()

    # ---------------------- Scheduler Tab ---------------------- #
    def create_scheduler_tab(self):
        # Calendar and Event Management
        self.scheduler_frame.columnconfigure(0, weight=1)
        self.scheduler_frame.rowconfigure(0, weight=1)

        calendar_frame = ttk.Frame(self.scheduler_frame)
        calendar_frame.grid(row=0, column=0, sticky='nsew')

        ttk.Label(calendar_frame, text="Calendar", font=("Helvetica", 18)).pack(pady=10)

        self.calendar = Calendar(
            calendar_frame,
            selectmode='day',
            date_pattern='yyyy-mm-dd',
            font=("Helvetica", 12),
            cursor="hand1"
        )
        self.calendar.pack(pady=10, padx=10, expand=True, fill=tk.BOTH)
        self.calendar.bind("<<CalendarSelected>>", self.update_event_list)

        # Event List
        event_list_frame = ttk.Frame(self.scheduler_frame)
        event_list_frame.grid(row=0, column=1, sticky='nsew')

        self.events_listbox = tk.Listbox(event_list_frame, font=("Helvetica", 12))
        self.events_listbox.pack(pady=10, padx=10, expand=True, fill=tk.BOTH)

        # Event Management
        event_mgmt_frame = ttk.Frame(self.scheduler_frame)
        event_mgmt_frame.grid(row=1, column=0, columnspan=2, sticky='ew')

        ttk.Label(event_mgmt_frame, text="Add Event", font=("Helvetica", 16)).grid(row=0, column=0, columnspan=4, pady=10)

        ttk.Label(event_mgmt_frame, text="Title:").grid(row=1, column=0, padx=5, pady=5)
        self.event_title_entry = ttk.Entry(event_mgmt_frame)
        self.event_title_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(event_mgmt_frame, text="Date:").grid(row=1, column=2, padx=5, pady=5)
        self.event_date_entry = DateEntry(event_mgmt_frame, date_pattern='yyyy-mm-dd')
        self.event_date_entry.grid(row=1, column=3, padx=5, pady=5)

        ttk.Label(event_mgmt_frame, text="Type:").grid(row=2, column=0, padx=5, pady=5)
        self.event_type_var = tk.StringVar()
        event_types = ['Exam', 'Meeting', 'Assignment', 'Other']
        self.event_type_combo = ttk.Combobox(event_mgmt_frame, textvariable=self.event_type_var, values=event_types, state='readonly')
        self.event_type_combo.grid(row=2, column=1, padx=5, pady=5)

        add_event_btn = ttk.Button(event_mgmt_frame, text="Add Event", command=self.add_event)
        add_event_btn.grid(row=2, column=2, columnspan=2, padx=5, pady=5)

        self.update_calendar_events()

    def add_event(self):
        title = self.event_title_entry.get()
        event_date = self.event_date_entry.get_date()
        event_type = self.event_type_var.get()

        if not title or not event_type:
            messagebox.showwarning("Input Error", "Please fill in all fields.")
            return

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO events (user_id, title, event_date, event_type)
            VALUES (?, ?, ?, ?)
        ''', (self.current_user, title, event_date, event_type))
        self.conn.commit()

        self.event_title_entry.delete(0, tk.END)
        self.event_type_combo.set('')
        self.event_date_entry.set_date(datetime.date.today())

        self.update_calendar_events()
        self.update_event_list()

    def update_calendar_events(self):
        self.calendar.calevent_remove('all')

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT event_date, event_type FROM events
            WHERE user_id=?
        ''', (self.current_user,))
        events = cursor.fetchall()

        for event_date_str, event_type in events:
            event_date = datetime.datetime.strptime(event_date_str, '%Y-%m-%d').date()
            color = self.get_event_color(event_type)
            self.calendar.calevent_create(event_date, event_type, tags=event_type)
            self.calendar.tag_config(event_type, background=color)

    def update_event_list(self, event=None):
        selected_date = self.calendar.selection_get()
        self.events_listbox.delete(0, tk.END)

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT title, event_type FROM events
            WHERE user_id=? AND event_date=?
        ''', (self.current_user, selected_date))
        events = cursor.fetchall()

        if events:
            for title, event_type in events:
                display_text = f"{event_type}: {title}"
                self.events_listbox.insert(tk.END, display_text)
        else:
            self.events_listbox.insert(tk.END, "No events on this date.")

    def get_event_color(self, event_type):
        color_map = {
            'Exam': 'red',
            'Meeting': 'blue',
            'Assignment': 'green',
            'Other': 'grey'
        }
        return color_map.get(event_type, 'black')

    # ---------------------- Tasks Tab ---------------------- #
    def create_tasks_tab(self):
        self.tasks_frame.columnconfigure(0, weight=1)
        self.tasks_frame.rowconfigure(0, weight=1)

        # Task List
        task_list_frame = ttk.Frame(self.tasks_frame)
        task_list_frame.grid(row=0, column=0, sticky='nsew')

        self.tasks_tree = ttk.Treeview(task_list_frame, columns=('Due Date', 'Priority'), show='headings')
        self.tasks_tree.heading('Due Date', text='Due Date')
        self.tasks_tree.heading('Priority', text='Priority')
        self.tasks_tree.pack(fill=tk.BOTH, expand=True)

        # Task Management
        task_mgmt_frame = ttk.Frame(self.tasks_frame)
        task_mgmt_frame.grid(row=1, column=0, sticky='ew')

        ttk.Label(task_mgmt_frame, text="Add Task", font=("Helvetica", 16)).grid(row=0, column=0, columnspan=4, pady=10)

        ttk.Label(task_mgmt_frame, text="Task:").grid(row=1, column=0, padx=5, pady=5)
        self.task_entry = ttk.Entry(task_mgmt_frame)
        self.task_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(task_mgmt_frame, text="Due Date:").grid(row=1, column=2, padx=5, pady=5)
        self.task_due_date_entry = DateEntry(task_mgmt_frame, date_pattern='yyyy-mm-dd')
        self.task_due_date_entry.grid(row=1, column=3, padx=5, pady=5)

        ttk.Label(task_mgmt_frame, text="Priority:").grid(row=2, column=0, padx=5, pady=5)
        self.task_priority_var = tk.StringVar()
        priorities = ['High', 'Medium', 'Low']
        self.task_priority_combo = ttk.Combobox(task_mgmt_frame, textvariable=self.task_priority_var, values=priorities, state='readonly')
        self.task_priority_combo.grid(row=2, column=1, padx=5, pady=5)
        self.task_priority_combo.set('Medium')

        add_task_btn = ttk.Button(task_mgmt_frame, text="Add Task", command=self.add_task)
        add_task_btn.grid(row=2, column=2, columnspan=2, padx=5, pady=5)

        mark_complete_btn = ttk.Button(task_mgmt_frame, text="Mark as Completed", command=self.mark_task_completed)
        mark_complete_btn.grid(row=3, column=0, columnspan=4, pady=10)

        self.update_tasks_list()

    def add_task(self):
        task = self.task_entry.get()
        due_date = self.task_due_date_entry.get_date()
        priority = self.task_priority_var.get()

        if not task:
            messagebox.showwarning("Input Error", "Please enter a task description.")
            return

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO tasks (user_id, task, due_date, priority)
            VALUES (?, ?, ?, ?)
        ''', (self.current_user, task, due_date, priority))
        self.conn.commit()

        self.task_entry.delete(0, tk.END)
        self.task_due_date_entry.set_date(datetime.date.today())
        self.task_priority_combo.set('Medium')

        self.update_tasks_list()

    def update_tasks_list(self):
        for item in self.tasks_tree.get_children():
            self.tasks_tree.delete(item)

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT task_id, task, due_date, priority FROM tasks
            WHERE user_id=? AND completed=0
            ORDER BY due_date ASC
        ''', (self.current_user,))
        tasks = cursor.fetchall()

        for task_id, task, due_date, priority in tasks:
            self.tasks_tree.insert('', 'end', iid=task_id, values=(due_date, priority), text=task)

    def mark_task_completed(self):
        selected_items = self.tasks_tree.selection()
        cursor = self.conn.cursor()
        for item in selected_items:
            cursor.execute('''
                UPDATE tasks SET completed=1 WHERE task_id=?
            ''', (item,))
        self.conn.commit()
        self.update_tasks_list()
        self.update_analytics()

    # ---------------------- Analytics Tab ---------------------- #
    def create_analytics_tab(self):
        ttk.Label(self.analytics_frame, text="Analytics", font=("Helvetica", 18)).pack(pady=10)

        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.chart_canvas = FigureCanvasTkAgg(self.figure, self.analytics_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_analytics()

    def update_analytics(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM tasks WHERE user_id=? AND completed=1
        ''', (self.current_user,))
        completed_tasks = cursor.fetchone()[0]

        cursor.execute('''
            SELECT COUNT(*) FROM tasks WHERE user_id=? AND completed=0
        ''', (self.current_user,))
        pending_tasks = cursor.fetchone()[0]

        total_tasks = completed_tasks + pending_tasks

        if total_tasks == 0:
            return

        self.figure.clear()

        labels = 'Completed', 'Pending'
        sizes = [completed_tasks, pending_tasks]
        colors = ['green', 'orange']

        ax = self.figure.add_subplot(111)
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')

        self.figure.suptitle('Task Completion Overview')
        self.chart_canvas.draw()

    # ---------------------- Help Tab ---------------------- #
    def create_help_tab(self):
        ttk.Label(self.help_frame, text="Help and Support", font=("Helvetica", 18)).pack(pady=10)
        help_text = """
        **Getting Started**

        - **Login/Register**: Create an account or log in to access your personalized scheduler.
        - **Scheduler Tab**: View and manage your events on the calendar.
        - **Tasks Tab**: Add tasks, set priorities, and mark them as completed.
        - **Analytics Tab**: View your productivity statistics.
        - **Notifications**: Receive reminders for upcoming tasks and events.

        **Adding Events**

        - Fill in the event title, select the date, and choose the event type.
        - Click 'Add Event' to save it to your calendar.

        **Adding Tasks**

        - Enter the task description, due date, and select the priority.
        - Click 'Add Task' to add it to your task list.

        **Export/Import Data**

        - Use the 'Export Data' and 'Import Data' options in the 'File' menu to back up or restore your data.
        """
        help_label = ttk.Label(self.help_frame, text=help_text, justify=tk.LEFT)
        help_label.pack(pady=10, padx=10)

    # ---------------------- Notifications ---------------------- #
    def notification_worker(self):
        while True:
            now = datetime.datetime.now()
            current_date = now.date()
            current_time = now.time()

            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT task FROM tasks
                WHERE user_id=? AND due_date=? AND completed=0
            ''', (self.current_user, current_date))
            tasks_due = cursor.fetchall()

            cursor.execute('''
                SELECT title FROM events
                WHERE user_id=? AND event_date=?
            ''', (self.current_user, current_date))
            events_today = cursor.fetchall()

            notification_text = ""
            if tasks_due:
                tasks_list = ', '.join([task[0] for task in tasks_due])
                notification_text += f"Tasks Due Today: {tasks_list}\n"
            if events_today:
                events_list = ', '.join([event[0] for event in events_today])
                notification_text += f"Events Today: {events_list}"

            if notification_text:
                notification.notify(
                    title="College Scheduler Reminder",
                    message=notification_text,
                    timeout=10
                )
            time.sleep(3600)  # Check every hour

    # ---------------------- Utility Methods ---------------------- #
    def clear_window(self):
        for widget in self.winfo_children():
            widget.destroy()

    def on_closing(self):
        self.conn.close()
        self.destroy()

if __name__ == "__main__":
    app = CollegeScheduler()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
