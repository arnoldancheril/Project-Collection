from pyparsing import (
    Word, alphas, alphanums, Keyword, Forward, Group, Suppress, delimitedList,
    oneOf, Literal, QuotedString, Optional, ZeroOrMore, nums, ParserElement
)

# Enable packrat parsing for better performance
ParserElement.enablePackrat()

class SQLParser:
    def __init__(self):
        self._build_parser()

    def _build_parser(self):
        # Define SQL tokens
        identifier = Word(alphas, alphanums + "_$").setName("identifier")
        columnName = identifier.copy()
        tableName = identifier.copy()
        integer = Word(nums)
        value = (
            QuotedString("'", escChar='\\') |
            QuotedString('"', escChar='\\') |
            Word(alphanums + "_$@.:-")
        )

        datatype = oneOf("INT INTEGER VARCHAR CHAR TEXT FLOAT DOUBLE BOOL BOOLEAN")

        # Define SQL keywords
        SELECT, FROM, WHERE, INSERT, INTO, VALUES, CREATE, TABLE, UPDATE, SET, DELETE, PRIMARY, KEY = map(
            lambda s: Keyword(s, caseless=True),
            "SELECT FROM WHERE INSERT INTO VALUES CREATE TABLE UPDATE SET DELETE PRIMARY KEY".split()
        )

        # Comparison operators
        compOp = oneOf("= != < > >= <= eq ne lt le gt ge", caseless=True)

        # Define expression components
        columnRval = value | integer
        condition = Group(
            columnName('column') +
            compOp('operator') +
            columnRval('value')
        )

        # WHERE clause
        whereClause = Optional(Suppress(WHERE) + condition('where'))

        # Define SELECT statement
        selectStmt = (
            SELECT +
            ('*' | Group(delimitedList(columnName))('columns')) +
            FROM +
            tableName('table') +
            whereClause
        )

        # Define INSERT statement
        insertStmt = (
            INSERT + INTO +
            tableName('table') +
            Group(Suppress('(') + delimitedList(columnName) + Suppress(')'))('columns') +
            VALUES +
            Group(Suppress('(') + delimitedList(value) + Suppress(')'))('values')
        )

        # Define column definitions for CREATE TABLE
        columnDef = Group(
            columnName('name') +
            datatype('type') +
            Optional(PRIMARY + KEY, default=False)('primary_key')
        )

        # Define CREATE TABLE statement
        createStmt = (
            CREATE + TABLE +
            tableName('table') +
            Group(Suppress('(') + delimitedList(columnDef) + Suppress(')'))('columns')
        )

        # Define assignment for UPDATE
        assignment = Group(
            columnName('column') +
            Suppress('=') +
            value('value')
        )

        # Define UPDATE statement
        updateStmt = (
            UPDATE +
            tableName('table') +
            SET +
            Group(delimitedList(assignment))('assignments') +
            whereClause
        )

        # Define DELETE statement
        deleteStmt = (
            DELETE + FROM +
            tableName('table') +
            whereClause
        )

        # Define the full SQL statement parser
        self.parser = (
            selectStmt('select') |
            insertStmt('insert') |
            createStmt('create') |
            updateStmt('update') |
            deleteStmt('delete')
        ) + Suppress(';')

    def parse(self, query):
        try:
            parsed = self.parser.parseString(query, parseAll=True)
            return self._parsed_to_command(parsed)
        except Exception as e:
            raise SyntaxError(f"Invalid SQL syntax: {e}")

    def _parsed_to_command(self, parsed):
        if 'select' in parsed:
            select = parsed['select']
            fields = select.columns.asList() if 'columns' in select and select.columns != '*' else ['*']
            condition = select.where.asList() if 'where' in select else None
            return {
                'action': 'select',
                'fields': fields,
                'table': select.table,
                'condition': condition
            }
        elif 'insert' in parsed:
            insert = parsed['insert']
            return {
                'action': 'insert',
                'table': insert.table,
                'columns': insert.columns.asList(),
                'values': insert.values.asList()
            }
        elif 'create' in parsed:
            create = parsed['create']
            columns = []
            for col in create.columns:
                columns.append({
                    'name': col.name,
                    'type': col.type,
                    'primary_key': bool(col.primary_key)
                })
            return {
                'action': 'create_table',
                'table': create.table,
                'columns': columns
            }
        elif 'update' in parsed:
            update = parsed['update']
            assignments = []
            for assign in update.assignments:
                assignments.append({
                    'column': assign.column,
                    'value': assign.value
                })
            condition = update.where.asList() if 'where' in update else None
            return {
                'action': 'update',
                'table': update.table,
                'assignments': assignments,
                'condition': condition
            }
        elif 'delete' in parsed:
            delete = parsed['delete']
            condition = delete.where.asList() if 'where' in delete else None
            return {
                'action': 'delete',
                'table': delete.table,
                'condition': condition
            }
        else:
            raise ValueError("Unsupported SQL command.")

# For testing purposes
if __name__ == "__main__":
    parser = SQLParser()

    queries = [
        "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR, age INT);",
        "INSERT INTO users (id, name, age) VALUES ('1', 'Alice', '30');",
        "SELECT * FROM users;",
        "UPDATE users SET age='31' WHERE id='1';",
        "DELETE FROM users WHERE id='2';"
    ]

    for query in queries:
        print(f"Query: {query}")
        command = parser.parse(query)
        print(f"Parsed Command: {command}\n")
