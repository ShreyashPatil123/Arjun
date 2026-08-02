import sqlite3
import os

db_path = os.path.expanduser(r"~\AppData\Roaming\com.sarathi.app\sarathi.db")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("PRAGMA table_info(memory_nodes);")
columns = cur.fetchall()
print("memory_nodes columns:", columns)

cur.execute("PRAGMA table_info(user_profile);")
print("user_profile columns:", cur.fetchall())
conn.close()
