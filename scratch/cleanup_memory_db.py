"""Clean up bulk stress test garbage from sarathi.db memory_nodes table."""
import sqlite3
import os

db_path = os.path.expanduser(r"~\AppData\Roaming\com.sarathi.app\sarathi.db")
print(f"Database: {db_path}")
print(f"Exists: {os.path.exists(db_path)}")

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Count before
cur.execute("SELECT COUNT(*) FROM memory_nodes")
total_before = cur.fetchone()[0]
print(f"\nTotal memory_nodes BEFORE cleanup: {total_before}")

# Count bulk stress test entries
cur.execute("SELECT COUNT(*) FROM memory_nodes WHERE id LIKE 'mem_bulk_%'")
bulk_count = cur.fetchone()[0]
print(f"Bulk stress test entries (id LIKE 'mem_bulk_%'): {bulk_count}")

# Count non-bulk entries
cur.execute("SELECT COUNT(*) FROM memory_nodes WHERE id NOT LIKE 'mem_bulk_%'")
real_count = cur.fetchone()[0]
print(f"Real memory entries: {real_count}")

# Show real entries
cur.execute("SELECT id, memory_type, content, importance_score FROM memory_nodes WHERE id NOT LIKE 'mem_bulk_%' LIMIT 20")
rows = cur.fetchall()
print(f"\nReal memory entries ({len(rows)}):")
for r in rows:
    print(f"  - id='{r[0]}', type='{r[1]}', content='{r[2][:80]}', importance={r[3]}")

# Show user_profile entries
cur.execute("SELECT key, value, category FROM user_profile")
profiles = cur.fetchall()
print(f"\nUser Profile ({len(profiles)} entries):")
for p in profiles:
    print(f"  - key='{p[0]}', value='{p[1]}', category='{p[2]}'")

# Delete bulk stress test entries
cur.execute("DELETE FROM memory_nodes WHERE id LIKE 'mem_bulk_%'")
deleted = cur.rowcount
conn.commit()
print(f"\n[CLEANUP] Deleted {deleted} bulk stress test entries")

# Count after
cur.execute("SELECT COUNT(*) FROM memory_nodes")
total_after = cur.fetchone()[0]
print(f"Total memory_nodes AFTER cleanup: {total_after}")

conn.close()
print("\n[DONE] Database cleanup complete.")
