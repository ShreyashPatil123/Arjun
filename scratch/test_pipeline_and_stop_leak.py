import sqlite3

db_path = r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app\sarathi.db"
print("Inspecting DB at:", db_path)

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()
print("Tables:", tables)

for (tname,) in tables:
    cur.execute(f"SELECT COUNT(*) FROM {tname}")
    cnt = cur.fetchone()[0]
    print(f"Table '{tname}': {cnt} rows")
    if cnt > 0:
        cur.execute(f"SELECT * FROM {tname} LIMIT 5")
        print(f"   Sample rows from '{tname}':", cur.fetchall())
