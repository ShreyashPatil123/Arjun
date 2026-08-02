import sqlite3
import os

app_data = os.path.expanduser(r"~\AppData\Roaming\com.sarathi.app")
db_path = os.path.join(app_data, "sarathi_memory.db")

print("Checking SQLite DB at:", db_path)
if not os.path.exists(db_path):
    print("DB file does NOT exist!")
else:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()
    print("Tables:", tables)

    for table in ["user_profile", "memory_nodes", "projects"]:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            cnt = cur.fetchone()[0]
            print(f"Table '{table}' count: {cnt}")
            if cnt > 0:
                cur.execute(f"SELECT * FROM {table} LIMIT 5")
                print(f"Sample rows from {table}:", cur.fetchall())
        except Exception as e:
            print(f"Error checking {table}:", e)
