# index.py

class BTreeIndex:
    def __init__(self):
        self.index = {}

    def build_index(self, table_data, column):
        self.index = {}
        for row in table_data:
            key = row.get(column)
            if key in self.index:
                self.index[key].append(row)
            else:
                self.index[key] = [row]

    def query(self, value):
        return self.index.get(value, [])
