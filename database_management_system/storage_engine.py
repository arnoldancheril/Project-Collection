# storage_engine.py

import os
import pickle

class StorageEngine:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def read_table(self, table_name):
        try:
            with open(f"{self.data_dir}/{table_name}.tbl", 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return []

    def write_table(self, table_name, data):
        with open(f"{self.data_dir}/{table_name}.tbl", 'wb') as f:
            pickle.dump(data, f)
