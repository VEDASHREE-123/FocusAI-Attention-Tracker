import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = 'attention_data.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS session_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            status TEXT,
            score INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def insert_log(status, score):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO session_logs (timestamp, status, score) VALUES (?, ?, ?)',
              (datetime.now(), status, score))
    conn.commit()
    conn.close()

def fetch_logs():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query('SELECT * FROM session_logs', conn)
    conn.close()
    return df

def clear_logs():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM session_logs')
    conn.commit()
    conn.close()
