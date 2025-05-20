# transaction_manager.py

class TransactionManager:
    def __init__(self):
        self.active_transactions = {}

    def begin_transaction(self, transaction_id):
        self.active_transactions[transaction_id] = {'logs': []}

    def log_operation(self, transaction_id, operation):
        self.active_transactions[transaction_id]['logs'].append(operation)

    def commit(self, transaction_id):
        del self.active_transactions[transaction_id]

    def rollback(self, transaction_id):
        # Reverse operations (simplified)
        del self.active_transactions[transaction_id]
