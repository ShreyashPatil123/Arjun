"""Intent sidecar tests that run with no weights and, mostly, no torch.

What can be checked without the model is checked here: the question file, the
refusals `load` makes before it touches torch, the JSON-RPC surface as the
parent process sees it, and the head-budget arithmetic against a stand-in
tokenizer. What cannot -- Laya's probabilities on real prompts -- is measured by
`capability::intent_eval::measure` on the Rust side, against the real weights,
and nowhere else.

Written against `unittest`, like the graph and document sidecars next door, so
an air-gapped install can run its own suite without pytest.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import engine  # noqa: E402

LABELS = ["coding", "mathematics", "reasoning", "tool-calling", "research", "general"]
HAS_TORCH_AND_LAYA = all(importlib.util.find_spec(m) for m in ("torch", "laya"))


class QuestionTest(unittest.TestCase):
    def test_the_question_names_the_six_intents(self):
        question = engine.load_question()
        spec = question["questions"][question["question_id"]]
        self.assertEqual(spec["type"], "choice")
        self.assertEqual(sorted(spec["criteria"]), sorted(LABELS))

    def test_the_fingerprint_is_the_file_bytes(self):
        with open(engine.QUESTION_PATH, "rb") as handle:
            expected = hashlib.sha256(handle.read()).hexdigest()
        self.assertEqual(engine.load_question()["fingerprint"], expected)


class LoadRefusalTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("ARJUN_LAYA_DIR")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("ARJUN_LAYA_DIR", None)
        else:
            os.environ["ARJUN_LAYA_DIR"] = self._saved

    def test_a_missing_directory_is_refused_by_name(self):
        os.environ["ARJUN_LAYA_DIR"] = os.path.join(tempfile.gettempdir(), "arjun-no-such-laya")
        with self.assertRaises(FileNotFoundError) as caught:
            engine.IntentEngine().load()
        self.assertIn("not installed", str(caught.exception))

    def test_a_directory_without_a_checkpoint_is_refused(self):
        with tempfile.TemporaryDirectory() as empty:
            os.environ["ARJUN_LAYA_DIR"] = empty
            with self.assertRaises(FileNotFoundError) as caught:
                engine.IntentEngine().load()
            self.assertIn("rl_agent_config.json", str(caught.exception))

    def test_classify_before_load_is_an_error(self):
        with self.assertRaises(RuntimeError):
            engine.IntentEngine().classify("hello")


class ProtocolTest(unittest.TestCase):
    """The sidecar as the Rust parent sees it: lines in, lines out."""

    def run_sidecar(self, *frames):
        with tempfile.TemporaryDirectory() as empty:
            env = dict(os.environ, ARJUN_LAYA_DIR=empty)
            completed = subprocess.run(
                [sys.executable, os.path.join(HERE, "main.py")],
                input="\n".join(frames) + "\n",
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
        return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]

    def test_every_line_on_stdout_is_a_response(self):
        replies = self.run_sidecar(
            '{"jsonrpc":"2.0","id":1,"method":"intent.ping","params":{}}',
            "not json",
            '{"jsonrpc":"2.0","id":2,"method":"intent.load","params":{}}',
            '{"jsonrpc":"2.0","id":3,"method":"intent.nope","params":{}}',
            '{"jsonrpc":"2.0","id":4,"method":"intent.status","params":{}}',
        )
        self.assertEqual([r["id"] for r in replies], [1, None, 2, 3, 4])
        self.assertEqual(replies[0]["result"], {"ok": True})
        self.assertEqual(replies[1]["error"]["code"], -32700)
        self.assertIn("rl_agent_config.json", replies[2]["error"]["message"])
        self.assertIn("unknown method", replies[3]["error"]["message"])
        status = replies[4]["result"]
        self.assertFalse(status["loaded"])
        self.assertEqual(status["device"], "cpu", "CPU unless an operator opts in")
        self.assertEqual(status["questionFingerprint"], engine.load_question()["fingerprint"])

    def test_hindi_survives_a_legacy_console_encoding(self):
        # cp1252 is what stdin decodes as on many Windows machines. The sidecar
        # must still read the parent's UTF-8, and answer in ASCII-safe JSON.
        prompt = "इस रिपोर्ट का सारांश दो"
        frame = json.dumps({"jsonrpc": "2.0", "id": 1, "method": prompt, "params": {}}, ensure_ascii=False)
        with tempfile.TemporaryDirectory() as empty:
            env = dict(os.environ, ARJUN_LAYA_DIR=empty, PYTHONIOENCODING="cp1252", PYTHONUTF8="0")
            completed = subprocess.run(
                [sys.executable, os.path.join(HERE, "main.py")],
                input=(frame + "\n").encode("utf-8"),
                capture_output=True,
                env=env,
                timeout=60,
            )
        raw = completed.stdout.decode("ascii")  # would raise on non-ASCII output
        reply = json.loads(raw)
        self.assertIn(prompt, reply["error"]["message"], "the prompt arrived intact")

    def test_the_hub_is_switched_off_before_anything_can_import_it(self):
        with open(os.path.join(HERE, "main.py"), encoding="utf-8") as handle:
            source = handle.read()
        offline = source.index('os.environ["HF_HUB_OFFLINE"] = "1"')
        self.assertLess(offline, source.index("from engine import"))
        self.assertIn('os.environ["TRANSFORMERS_OFFLINE"] = "1"', source)


class _WordTokenizer:
    """One token per whitespace-separated word. A stand-in, not ModernBERT."""

    def __call__(self, text, add_special_tokens=False, **_):
        return {"input_ids": list(range(len(text.split())))}


class _Agent:
    def __init__(self, head_max_len):
        self.tok = _WordTokenizer()
        self.cfg = {"head_max_len": head_max_len}


@unittest.skipUnless(HAS_TORCH_AND_LAYA, "needs torch and laya: laya.common imports torch")
class BudgetTest(unittest.TestCase):
    """The mirror of `laya.common.build_sequence`'s truncation rule."""

    def test_the_question_fits_a_roomy_budget(self):
        engine.IntentEngine()._check_budget("english", _Agent(head_max_len=192))

    def test_a_budget_the_options_overflow_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            engine.IntentEngine()._check_budget("english", _Agent(head_max_len=60))
        self.assertIn("would truncate", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
