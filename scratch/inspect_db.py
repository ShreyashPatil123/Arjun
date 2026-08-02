import os
import glob

appdata_roaming = os.path.expanduser(r"~\AppData\Roaming")
appdata_local = os.path.expanduser(r"~\AppData\Local")

print("Searching for sarathi_memory.db across AppData...")
for root, dirs, files in os.walk(appdata_roaming):
    for f in files:
        if "memory" in f.lower() or "sarathi" in f.lower():
            print("Found file:", os.path.join(root, f))
            
for root, dirs, files in os.walk(appdata_local):
    for f in files:
        if "sarathi_memory" in f.lower():
            print("Found file:", os.path.join(root, f))
