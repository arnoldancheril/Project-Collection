# gui.py

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QMenuBar, QAction, QMessageBox, QFileDialog, QHeaderView
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from main import SimpleDBMS

class DBMSGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dbms = SimpleDBMS()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Custom DBMS GUI')
        self.setGeometry(100, 100, 800, 600)
        self.create_menu()
        self.create_widgets()
        self.show()

    def create_menu(self):
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu('File')
        export_action = QAction('Export Results', self)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help Menu
        help_menu = menu_bar.addMenu('Help')
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_widgets(self):
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layouts
        main_layout = QVBoxLayout()
        input_layout = QHBoxLayout()
        result_layout = QVBoxLayout()

        # Query Input
        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText('Enter SQL Query...')
        execute_button = QPushButton('Execute')
        execute_button.clicked.connect(self.execute_query)
        input_layout.addWidget(self.query_input)
        input_layout.addWidget(execute_button)

        # Result Display
        self.result_table = QTableWidget()
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        result_layout.addWidget(QLabel('Result:'))
        result_layout.addWidget(self.result_table)

        # Combine Layouts
        main_layout.addLayout(input_layout)
        main_layout.addLayout(result_layout)
        central_widget.setLayout(main_layout)

    def execute_query(self):
        query = self.query_input.toPlainText().strip()
        if not query.endswith(';'):
            query += ';'
        result = self.dbms.execute(query)

        if isinstance(result, str):
            if result.startswith('Error:'):
                QMessageBox.critical(self, 'Error', result)
            else:
                QMessageBox.information(self, 'Result', result)
            # Clear the result table if necessary
            self.result_table.clear()
            self.result_table.setRowCount(0)
            self.result_table.setColumnCount(0)
        elif isinstance(result, list):
            self.display_result(result)
        else:
            QMessageBox.warning(self, 'Warning', 'Unexpected result type.')

    def display_result(self, data):
        if not data:
            self.result_table.clear()
            self.result_table.setRowCount(0)
            self.result_table.setColumnCount(0)
            QMessageBox.information(self, 'Result', 'No data found.')
            return

        headers = list(data[0].keys())
        self.result_table.setColumnCount(len(headers))
        self.result_table.setHorizontalHeaderLabels(headers)
        self.result_table.setRowCount(len(data))

        for row_idx, row_data in enumerate(data):
            for col_idx, header in enumerate(headers):
                value = row_data.get(header, '')
                item = QTableWidgetItem(str(value))
                self.result_table.setItem(row_idx, col_idx, item)

    def export_results(self):
        if self.result_table.rowCount() == 0:
            QMessageBox.warning(self, 'No Data', 'No data to export.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save File', '', 'CSV Files (*.csv)')
        if path:
            with open(path, 'w') as file:
                # Write headers
                headers = [self.result_table.horizontalHeaderItem(i).text() for i in range(self.result_table.columnCount())]
                file.write(','.join(headers) + '\n')
                # Write data
                for row in range(self.result_table.rowCount()):
                    row_data = []
                    for column in range(self.result_table.columnCount()):
                        item = self.result_table.item(row, column)
                        row_data.append(item.text() if item else '')
                    file.write(','.join(row_data) + '\n')
            QMessageBox.information(self, 'Export Successful', f'Results exported to {path}.')

    def show_about(self):
        QMessageBox.about(self, 'About', 'Custom DBMS GUI\nDeveloped using PyQt5.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = DBMSGUI()
    sys.exit(app.exec_())