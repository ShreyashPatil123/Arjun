"""
Sarathi Memory Engine Sidecar Server
Listens for JSON-RPC 2.0 frames over stdin/stdout.
Processes requests with zero network sockets or firewall prompts.
"""

import sys
import os

# Ensure sidecar directory is on sys.path for absolute imports
sidecar_dir = os.path.dirname(os.path.abspath(__file__))
if sidecar_dir not in sys.path:
    sys.path.insert(0, sidecar_dir)

import json
import traceback
from router import MemoryActionRouter

def main():
    router = MemoryActionRouter()
    # Signal ready over stdout
    sys.stdout.flush()

    for line in sys.stdin:
        if not line:
            break
        line_str = line.strip()
        if not line_str:
            continue

        try:
            req = json.loads(line_str)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            try:
                result = router.dispatch(method, params)
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": result
                }
            except Exception as ex:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32603,
                        "message": str(ex),
                        "data": traceback.format_exc()
                    }
                }

        except Exception as json_err:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {json_err}"
                }
            }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
