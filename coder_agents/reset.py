
import sqlite3

from .utils import db_path, get_session_id, log, log_node

def reset_agent(state):
    """
    Graph node that resets the session by dropping and recreating the
    programmer_messages SQLite table, then passes the state through unchanged.
    """
    
    print("\n\n---RESETTING---")
    log_node("RESET")
    log(f"  session: {get_session_id()}")
    log(f"  database: {db_path()}")
    # Step 1: Create (or connect to) the database
    conn = sqlite3.connect(db_path())
    cursor = conn.cursor()

    # Step 2: Create a table to store messages (if it doesn't exist)
    cursor.execute("DROP TABLE IF EXISTS programmer_messages")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS programmer_messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prefix TEXT,
        install TEXT,
        imports TEXT,
        code TEXT,
        timestamp TEXT
    )
    ''')

    conn.close()
    return state