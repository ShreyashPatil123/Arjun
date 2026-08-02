import subprocess
import sys
import os

print("--- Testing Python Sidecar Startup ---")
sidecar_script = os.path.abspath("sidecars/memory_engine_sidecar/main.py")
print(f"Target Script: {sidecar_script}")

try:
    proc = subprocess.Popen(
        [sys.executable, sidecar_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print("Process spawned successfully!")
    out, err = proc.communicate(input='{"jsonrpc": "2.0", "id": 1, "method": "health_check", "params": {}}\n', timeout=5)
    print("STDOUT:", out)
    print("STDERR:", err)
except Exception as e:
    print("EXECUTION FAILED:", e)
