import os
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "support_tickets.db")
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "support_tickets.csv")

# Disallowed SQL keywords for safe read-only execution
DISALLOWED_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE", "CREATE", "EXEC", "ATTACH", "DETACH"]

class DatabaseManager:
    """Manages SQLite database ingestion, indexing, and safe read-only queries."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH, csv_path: str = DEFAULT_CSV_PATH):
        self.db_path = db_path
        self.csv_path = csv_path
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self, force_reload: bool = False) -> int:
        """Initializes database and loads tickets from CSV if table does not exist or force_reload is True."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tickets'")
            table_exists = cursor.fetchone() is not None
            
            if table_exists and not force_reload:
                cursor.execute("SELECT COUNT(*) FROM tickets")
                count = cursor.fetchone()[0]
                if count > 0:
                    return count

            # Create table schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP NOT NULL,
                    category TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_time_hrs REAL,
                    resolution_time_hrs REAL,
                    agent_id TEXT NOT NULL,
                    customer_rating INTEGER,
                    issue_summary TEXT NOT NULL
                )
            """)

            # Read CSV
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"CSV dataset not found at {self.csv_path}")

            df = pd.read_csv(self.csv_path)
            # Standardize column types
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['response_time_hrs'] = pd.to_numeric(df['response_time_hrs'], errors='coerce')
            df['resolution_time_hrs'] = pd.to_numeric(df['resolution_time_hrs'], errors='coerce')
            df['customer_rating'] = pd.to_numeric(df['customer_rating'], errors='coerce')

            # Clean existing records if force_reload
            cursor.execute("DELETE FROM tickets")

            # Insert batch
            records = []
            for _, row in df.iterrows():
                records.append((
                    str(row['ticket_id']),
                    row['created_at'].strftime('%Y-%m-%d %H:%M:%S'),
                    str(row['category']),
                    str(row['priority']),
                    str(row['status']),
                    float(row['response_time_hrs']) if pd.notnull(row['response_time_hrs']) else None,
                    float(row['resolution_time_hrs']) if pd.notnull(row['resolution_time_hrs']) else None,
                    str(row['agent_id']),
                    int(row['customer_rating']) if pd.notnull(row['customer_rating']) else None,
                    str(row['issue_summary'])
                ))

            cursor.executemany("""
                INSERT OR REPLACE INTO tickets (
                    ticket_id, created_at, category, priority, status,
                    response_time_hrs, resolution_time_hrs, agent_id, customer_rating, issue_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)

            # Create performance indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON tickets(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON tickets(priority);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_id ON tickets(agent_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON tickets(category);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON tickets(created_at);")

            conn.commit()
            return len(records)

    def execute_safe_query(self, sql_query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Executes a strictly validated read-only SQL query."""
        clean_query = sql_query.strip().rstrip(';')
        
        # Security validation: reject multiple statements or destructive verbs
        if ";" in clean_query:
            raise ValueError("Multiple SQL statements are not permitted for security reasons.")

        tokens = [t.strip().upper() for t in clean_query.split()]
        if not tokens or tokens[0] not in ["SELECT", "WITH"]:
            raise ValueError(f"Only SELECT or WITH queries are permitted. Query starts with '{tokens[0] if tokens else ''}'.")

        for kw in DISALLOWED_KEYWORDS:
            if kw in tokens:
                raise ValueError(f"Security error: Query contains prohibited keyword '{kw}'.")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(clean_query, params)
            else:
                cursor.execute(clean_query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_dataframe(self) -> pd.DataFrame:
        """Retrieves entire tickets table as pandas DataFrame."""
        with self.get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM tickets", conn)
            df['created_at'] = pd.to_datetime(df['created_at'])
            return df

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculates global operational KPIs."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_tickets,
                    SUM(CASE WHEN status = 'Open' THEN 1 ELSE 0 END) as open_tickets,
                    SUM(CASE WHEN status = 'Resolved' THEN 1 ELSE 0 END) as resolved_tickets,
                    SUM(CASE WHEN status = 'Escalated' THEN 1 ELSE 0 END) as escalated_tickets,
                    ROUND(AVG(response_time_hrs), 2) as avg_response_hrs,
                    ROUND(AVG(resolution_time_hrs), 2) as avg_resolution_hrs,
                    ROUND(AVG(customer_rating), 2) as avg_csat,
                    MIN(created_at) as earliest_date,
                    MAX(created_at) as latest_date
                FROM tickets
            """)
            row = dict(cursor.fetchone())
            return row

# Global instance for easy import
db_manager = DatabaseManager()
