import sqlite3, os
db = os.path.expanduser(r"~\AppData\Roaming\com.sarathi.app\sarathi.db")
conn = sqlite3.connect(db)
conn.execute("UPDATE user_profile SET value='Shreyash Patil' WHERE key='name'")
conn.commit()
print("Fixed name in user_profile")
for r in conn.execute("SELECT * FROM user_profile").fetchall():
    print(f"  {r}")
conn.close()
