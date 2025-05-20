import os
import pickle

class StorageEngine:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.schema_file = os.path.join(self.data_dir, 'schema.pkl')
        self.schema = self.load_schema()

    def load_schema(self):
        """Loads the schema from disk."""
        if os.path.exists(self.schema_file):
            with open(self.schema_file, 'rb') as f:
                return pickle.load(f)
        else:
            return {}

    def save_schema(self):
        """Saves the schema to disk."""
        with open(self.schema_file, 'wb') as f:
            pickle.dump(self.schema, f)

    def create_table(self, table_name, columns):
        """Creates a new table with the given columns and constraints."""
        if table_name in self.schema:
            raise Exception(f"Table '{table_name}' already exists.")

        # Store columns with data types and constraints
        column_defs = {}
        for col in columns:
            column_name = col['name']
            data_type = col['type'].upper()
            primary_key = col.get('primary_key', False)
            column_defs[column_name] = {
                'type': data_type,
                'primary_key': primary_key
            }

        self.schema[table_name] = column_defs
        self.save_schema()

        # Initialize the table data file
        self.write_table(table_name, [])

    def get_table_columns(self, table_name):
        """Retrieves the schema of a table."""
        if table_name not in self.schema:
            raise Exception(f"Table '{table_name}' does not exist.")
        return self.schema[table_name]

    def read_table(self, table_name):
        """Reads table data from disk."""
        self.ensure_table_exists(table_name)
        table_file = os.path.join(self.data_dir, f"{table_name}.tbl")
        if os.path.exists(table_file):
            with open(table_file, 'rb') as f:
                return pickle.load(f)
        else:
            return []

    def write_table(self, table_name, data):
        """Writes table data to disk."""
        self.ensure_table_exists(table_name)
        table_file = os.path.join(self.data_dir, f"{table_name}.tbl")
        with open(table_file, 'wb') as f:
            pickle.dump(data, f)

    def ensure_table_exists(self, table_name):
        """Checks if the table exists in the schema."""
        if table_name not in self.schema:
            raise Exception(f"Table '{table_name}' does not exist.")

    def delete_table(self, table_name):
        """Deletes a table and its data."""
        self.ensure_table_exists(table_name)
        # Remove table file
        table_file = os.path.join(self.data_dir, f"{table_name}.tbl")
        if os.path.exists(table_file):
            os.remove(table_file)
        # Remove table schema
        del self.schema[table_name]
        self.save_schema()

    def table_exists(self, table_name):
        """Checks if a table exists."""
        return table_name in self.schema

    def drop_table(self, table_name):
        """Drops a table (alias for delete_table)."""
        self.delete_table(table_name)

    def clear_table_data(self, table_name):
        """Clears all data from a table without deleting its schema."""
        self.ensure_table_exists(table_name)
        self.write_table(table_name, [])

# For testing purposes
if __name__ == "__main__":
    storage = StorageEngine()

    # Define table columns with data types and constraints
    columns = [
        {'name': 'id', 'type': 'INT', 'primary_key': True},
        {'name': 'name', 'type': 'VARCHAR'},
        {'name': 'age', 'type': 'INT'}
    ]

    # Create a table
    storage.create_table('users', columns)

    # Retrieve table schema
    schema = storage.get_table_columns('users')
    print(f"Schema for 'users': {schema}")

    # Write data to the table
    data = [
        {'id': 1, 'name': 'Alice', 'age': 30},
        {'id': 2, 'name': 'Bob', 'age': 25}
    ]
    storage.write_table('users', data)

    # Read data from the table
    users = storage.read_table('users')
    print(f"Data in 'users' table: {users}")

    # Clear table data
    storage.clear_table_data('users')
    users = storage.read_table('users')
    print(f"Data in 'users' table after clearing: {users}")

    # Drop the table
    storage.drop_table('users')
    print(f"Table 'users' exists after dropping: {storage.table_exists('users')}")