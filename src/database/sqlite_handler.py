
import sqlite3
import pandas as pd
from pathlib import Path

class SQLiteHandler:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
    
    def connect(self):
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
    
    def create_tables(self, schema_file):
        try:
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            cursor = self.connection.cursor()
            cursor.executescript(schema_sql)
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Table creation failed: {e}")
            return False
    
    def insert_dataframe(self, df, table_name, if_exists='replace'):
        try:
            df.to_sql(table_name, self.connection, if_exists=if_exists, index=False)
            return True
        except Exception as e:
            print(f"Data insertion failed: {e}")
            return False
    
    def execute_query(self, query, params=None, return_df=True):
        try:
            if return_df:
                df = pd.read_sql_query(query, self.connection, params=params)
                return df
            else:
                cursor = self.connection.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                self.connection.commit()
                return cursor.rowcount
        except Exception as e:
            print(f"Query execution failed: {e}")
            return None
    
    def close(self):
        if self.connection:
            self.connection.close()
