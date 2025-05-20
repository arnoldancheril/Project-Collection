import uuid
import logging
from copy import deepcopy

from storage_engine import StorageEngine
from sql_parser import SQLParser
from transaction_manager import TransactionManager
from concurrency_control import LockManager
from index import BTreeIndex

# Configure logging
logging.basicConfig(
    filename='dbms.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

class SimpleDBMS:
    def __init__(self):
        self.storage = StorageEngine()
        self.parser = SQLParser()
        self.tx_manager = TransactionManager()
        self.lock_manager = LockManager()
        self.indexes = {}

    def execute(self, query):
        transaction_id = str(uuid.uuid4())
        self.tx_manager.begin_transaction(transaction_id)
        logging.info(f"Executing query: {query}")
        try:
            command = self.parser.parse(query)
            action = command['action']

            if action == 'select':
                result = self.handle_select(command)
            elif action == 'insert':
                result = self.handle_insert(command, transaction_id)
            elif action == 'create_table':
                result = self.handle_create_table(command)
            elif action == 'update':
                result = self.handle_update(command, transaction_id)
            elif action == 'delete':
                result = self.handle_delete(command, transaction_id)
            else:
                raise Exception(f"Unsupported action: {action}")

            self.tx_manager.commit(transaction_id)
            logging.info("Query executed successfully.")
            return result

        except Exception as e:
            self.tx_manager.rollback(transaction_id)
            logging.error(f"Error executing query: {e}")
            return f"Error: {e}"

    def handle_create_table(self, command):
        table_name = command['table']
        columns = command['columns']  # List of column definitions
        self.storage.create_table(table_name, columns)
        column_names = [col['name'] for col in columns]
        logging.info(f"Table '{table_name}' created with columns {column_names}.")
        return f"Table '{table_name}' created with columns {column_names}."

    def handle_insert(self, command, transaction_id):
        table_name = command['table']
        columns = command['columns']
        values = command['values']
        table_schema = self.storage.get_table_columns(table_name)

        # Ensure columns match
        if set(columns) != set(table_schema.keys()):
            raise Exception("Column names do not match table schema.")

        # Enforce data types and constraints
        row = {}
        for col, val in zip(columns, values):
            col_def = table_schema[col]
            typed_value = self._convert_value(val, col_def['type'])
            row[col] = typed_value

        # Enforce primary key uniqueness
        primary_keys = [col for col, defn in table_schema.items() if defn.get('primary_key')]
        if primary_keys:
            existing_data = self.storage.read_table(table_name)
            for existing_row in existing_data:
                if all(existing_row[pk] == row[pk] for pk in primary_keys):
                    raise Exception("Primary key constraint violation.")

        # Proceed with insertion
        if not self.lock_manager.acquire_lock(table_name, transaction_id):
            raise Exception("Table is locked by another transaction.")

        data = self.storage.read_table(table_name)
        data.append(row)
        self.storage.write_table(table_name, data)
        self.lock_manager.release_lock(table_name, transaction_id)
        self.tx_manager.log_operation(transaction_id, ('insert', table_name, row))
        logging.info(f"Row inserted into '{table_name}': {row}")
        return "Insert successful."

    def handle_select(self, command):
        table_name = command['table']
        data = self.storage.read_table(table_name)
        table_schema = self.storage.get_table_columns(table_name)
        condition = command.get('condition')

        if condition:
            data = [row for row in data if self._evaluate_condition(row, condition, table_schema)]

        fields = command['fields']
        if '*' in fields:
            result = data
        else:
            result = [{k: v for k, v in row.items() if k in fields} for row in data]

        logging.info(f"SELECT query result from '{table_name}': {result}")
        return result

    def handle_update(self, command, transaction_id):
        table_name = command['table']
        assignments = command['assignments']
        condition = command.get('condition')

        if not self.lock_manager.acquire_lock(table_name, transaction_id):
            raise Exception("Table is locked by another transaction.")

        data = self.storage.read_table(table_name)
        table_schema = self.storage.get_table_columns(table_name)
        updated_rows = 0

        for row in data:
            if self._evaluate_condition(row, condition, table_schema):
                for assign in assignments:
                    column = assign['column']
                    value = assign['value']

                    if column not in table_schema:
                        raise Exception(f"Column '{column}' does not exist in table '{table_name}'.")

                    col_def = table_schema[column]
                    typed_value = self._convert_value(value, col_def['type'])
                    row[column] = typed_value

                updated_rows += 1

        self.storage.write_table(table_name, data)
        self.lock_manager.release_lock(table_name, transaction_id)
        self.tx_manager.log_operation(transaction_id, ('update', table_name, assignments))
        logging.info(f"{updated_rows} rows updated in '{table_name}'.")
        return f"{updated_rows} rows updated."

    def handle_delete(self, command, transaction_id):
        table_name = command['table']
        condition = command.get('condition')

        if not self.lock_manager.acquire_lock(table_name, transaction_id):
            raise Exception("Table is locked by another transaction.")

        data = self.storage.read_table(table_name)
        table_schema = self.storage.get_table_columns(table_name)
        original_data = deepcopy(data)

        data = [row for row in data if not self._evaluate_condition(row, condition, table_schema)]
        deleted_rows = len(original_data) - len(data)

        self.storage.write_table(table_name, data)
        self.lock_manager.release_lock(table_name, transaction_id)
        self.tx_manager.log_operation(transaction_id, ('delete', table_name, condition))
        logging.info(f"{deleted_rows} rows deleted from '{table_name}'.")
        return f"{deleted_rows} rows deleted."

    def _evaluate_condition(self, row, condition, table_schema):
        if not condition:
            return True  # No condition means all rows match

        column, operator, value = condition

        if column not in table_schema:
            return False

        col_def = table_schema[column]
        data_type = col_def['type']
        row_value = row.get(column)

        if row_value is None:
            return False

        # Convert value to appropriate type
        try:
            typed_value = self._convert_value(value, data_type)
        except ValueError:
            return False

        # Perform comparison
        if operator == '=':
            return row_value == typed_value
        elif operator == '!=':
            return row_value != typed_value
        elif operator == '<':
            return row_value < typed_value
        elif operator == '>':
            return row_value > typed_value
        elif operator == '<=':
            return row_value <= typed_value
        elif operator == '>=':
            return row_value >= typed_value
        else:
            return False

    def _convert_value(self, value, data_type):
        value = value.strip("'")

        if data_type.upper() in ['INT', 'INTEGER']:
            return int(value)
        elif data_type.upper() in ['FLOAT', 'DOUBLE']:
            return float(value)
        elif data_type.upper() in ['BOOL', 'BOOLEAN']:
            return value.lower() in ('true', '1')
        else:  # Default to string types
            return value

def main():
    dbms = SimpleDBMS()
    print("Welcome to the Custom DBMS CLI. Type 'exit' to quit.")
    while True:
        query = input("dbms> ")
        if query.lower() == 'exit':
            break
        result = dbms.execute(query)
        print(result)

if __name__ == "__main__":
    main()