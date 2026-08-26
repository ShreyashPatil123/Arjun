"""ARJUN Document Sidecar.

Reads JSON-RPC 2.0 frames from stdin and writes them to stdout, one per line.

Stdio rather than a socket, deliberately and for the same reason the memory
sidecar does it: a listening port is network surface, however local. A process
speaking over a pipe has none at all, cannot be reached by anything on the
machine, and never triggers a Windows firewall prompt. On a product whose core
claim is that nothing leaves the machine, "there is no socket" is a much easier
thing to prove than "the socket is bound to loopback".
"""

import json
import os
import sys
import traceback

# The sidecar is launched with this directory as its root, so its own modules
# are importable regardless of where the parent process was started from.
_SIDECAR_DIR = os.path.dirname(os.path.abspath(__file__))
if _SIDECAR_DIR not in sys.path:
    sys.path.insert(0, _SIDECAR_DIR)

from router import DocumentRouter  # noqa: E402


def _error(request_id, code, message, data=None):
    body = {"code": code, "message": message}
    if data is not None:
        body["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": body}


def main() -> None:
    router = DocumentRouter()

    # Announced once at startup so the Rust side knows what it is talking to
    # before it sends any work — including whether it is on the fallback engine.
    sys.stderr.write(f"[document-sidecar] {json.dumps(router.status())}\n")
    sys.stderr.flush()
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            method = request.get("method")
            params = request.get("params") or {}

            try:
                result = router.dispatch(method, params)
                response = {"jsonrpc": "2.0", "id": request_id, "result": result}
            except ValueError as bad_request:
                # The caller asked for something impossible — a missing file, an
                # unknown method. Its own fault, and recoverable, so it is an
                # invalid-params error rather than an internal one.
                response = _error(request_id, -32602, str(bad_request))
            except Exception as failure:  # noqa: BLE001
                response = _error(
                    request_id, -32603, str(failure), traceback.format_exc()
                )

        except json.JSONDecodeError as parse_error:
            response = _error(None, -32700, f"Parse error: {parse_error}")

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
