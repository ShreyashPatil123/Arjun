"""ARJUN intent sidecar.

Reads JSON-RPC 2.0 frames on stdin and answers on stdout, one JSON object per
line -- the transport `graph_sidecar` and `memory_engine_sidecar` already use.
No socket, so nothing to bind, no port to collide with and nothing another
process on the machine could reach. Laya ships an HTTP server (`laya-serve`),
and it was not used: it binds 0.0.0.0 unless told otherwise, and the Rust side
would need an HTTP client of its own that `scripts/check-egress.mjs` would then
have to exempt.

Methods:

    intent.ping      -> {"ok": true}
    intent.load      -> loads the installed checkpoints; returns intent.status
    intent.status    -> device, threads, memory, question fingerprint
    intent.classify  {"prompt": str, "checkpoint": "english"|"multilingual"|null}

Run it directly for a smoke test:

    ARJUN_LAYA_DIR=... python main.py
    {"jsonrpc":"2.0","id":1,"method":"intent.load","params":{}}
"""

import os

# Before anything can import huggingface_hub or transformers. Laya downloads a
# checkpoint whenever the path it is handed does not exist; on an air-gapped
# workbench that has to be an error, and these are the switches that make it one.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

sidecar_dir = os.path.dirname(os.path.abspath(__file__))
if sidecar_dir not in sys.path:
    sys.path.insert(0, sidecar_dir)

from engine import IntentEngine  # noqa: E402


def main() -> None:
    # The protocol owns stdout. Laya prints a warning to stdout when CUDA is
    # requested and absent (`laya/agent.py`), and one stray line there would be
    # read by the parent as a malformed response. Everything that is not a
    # response goes to stderr, which the parent inherits.
    protocol = sys.stdout
    sys.stdout = sys.stderr
    # The parent writes UTF-8. Python otherwise decodes stdin with the locale's
    # encoding -- cp1252 on many Windows machines -- which turns a Devanagari
    # prompt into mojibake before Laya ever sees it. Responses are written with
    # `ensure_ascii`, so they are valid on any console encoding.
    sys.stdin.reconfigure(encoding="utf-8", errors="strict")

    engine = IntentEngine()
    methods = {
        "intent.ping": lambda params: {"ok": True},
        "intent.load": lambda params: engine.load(),
        "intent.status": lambda params: engine.status(),
        "intent.classify": lambda params: engine.classify(
            str(params["prompt"]), params.get("checkpoint")
        ),
    }

    for line in sys.stdin:
        line_str = line.strip()
        if not line_str:
            continue
        try:
            req = json.loads(line_str)
            req_id = req.get("id")
            method = methods.get(req.get("method"))
            try:
                if method is None:
                    raise ValueError("unknown method %r" % req.get("method"))
                response = {"jsonrpc": "2.0", "id": req_id, "result": method(req.get("params") or {})}
            except Exception as ex:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(ex), "data": traceback.format_exc()},
                }
        except Exception as json_err:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "parse error: %s" % json_err},
            }
        protocol.write(json.dumps(response) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
