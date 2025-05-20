class LockManager:
    def __init__(self):
        self.locks = {}

    def acquire_lock(self, table_name, transaction_id):
        if table_name in self.locks:
            return False  # Lock already held
        else:
            self.locks[table_name] = transaction_id
            return True

    def release_lock(self, table_name, transaction_id):
        if self.locks.get(table_name) == transaction_id:
            del self.locks[table_name]
