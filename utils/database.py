import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
from config import Config
import os

def init_database():
    """Initialize the SQLite database with required tables"""
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
    
    conn = sqlite3.connect(Config.DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_id TEXT UNIQUE,
            po_id TEXT,
            vendor TEXT,
            description TEXT,
            date TEXT,
            amount REAL,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gl_id TEXT UNIQUE,
            invoice_id TEXT,
            vendor TEXT,
            account_code TEXT,
            date TEXT,
            amount REAL,
            debit_credit TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bank_statements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT UNIQUE,
            date TEXT,
            description TEXT,
            amount REAL,
            balance REAL,
            reference TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_table TEXT,
            source_id INTEGER,
            target_table TEXT,
            target_id INTEGER,
            match_type TEXT,
            confidence REAL,
            reasoning TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exceptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exception_id TEXT UNIQUE,
            type TEXT,
            description TEXT,
            amount REAL,
            priority_score REAL,
            status TEXT,
            assigned_to TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT,
            pattern_data TEXT,
            accuracy_impact REAL,
            usage_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def get_connection():
    """Get database connection"""
    return sqlite3.connect(Config.DATABASE_PATH)

def save_dataframe(df: pd.DataFrame, table_name: str, if_exists: str = 'replace') -> bool:
    """Save DataFrame to database table"""
    try:
        conn = get_connection()
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving to {table_name}: {str(e)}")
        return False

def load_dataframe(table_name: str, where_clause: str = None) -> pd.DataFrame:
    """Load DataFrame from database table"""
    try:
        conn = get_connection()
        query = f"SELECT * FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error loading from {table_name}: {str(e)}")
        return pd.DataFrame()
