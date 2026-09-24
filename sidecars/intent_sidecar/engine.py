"""Laya `choice` over ARJUN's six intents, from weights already on this machine.

The sidecar answers one question -- what kind of work is this turn asking for --
and nothing else. It never names a model: which installed LLM answers is decided
by ARJUN's own registry and router, against role, clearance and VRAM, exactly as
before. Laya only replaces the keyword counting that fed that router.

Three decisions here are deliberate and each was checked against Laya 0.3.20's
source rather than its README:

* **Local weights only.** `laya.Agent` calls `huggingface_hub.snapshot_download`
  whenever the path it is given does not exist. `main.py` sets the Hub's offline
  switches before anything imports it, and this module refuses to build a Router
  unless the checkpoint directory is already on disk, so a missing model is an
  error the parent can read -- never a download.
* **CPU by default.** The two checkpoints are 421M and 322M parameters: about
  1.7 GB and 1.3 GB of weights at fp32, half that at bf16, before activations.
  That is arithmetic from the parameter counts -- Laya publishes no VRAM figure
  -- and it is `status()` that reports what a GPU run actually allocated. On the
  8 GB cards ARJUN is sized for, a gigabyte or more is the margin that decides
  whether the chat model fits entirely in VRAM or runs partly on the CPU. `graph_sidecar` keeps
  REBEL on the CPU for the same reason. `ARJUN_LAYA_DEVICE=cuda` opts in.
* **No silent truncation of the option descriptions.** On the English checkpoint
  the six descriptions share a 192-token head budget with the instruction, and
  `laya.common.build_sequence` cuts every option down without a word when they
  overflow it. A description the model never read is a routing rule that does
  not exist, so `load` measures the budget with the checkpoint's own tokenizer
  and refuses to start rather than run with cut descriptions.
"""

import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

QUESTION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intent_question.json")

# The two checkpoints Laya's Router chooses between automatically. The bundle
# repo keeps the English one at its root and the multilingual one in a
# subfolder; `typed-decisions` is fine-tuned on four unrelated workflows and is
# never selected here.
CHECKPOINTS = {
    "english": None,
    "multilingual": "multilingual",
}


def load_question() -> Dict[str, Any]:
    """The intent question and a fingerprint of its exact bytes.

    The fingerprint is what a calibration file is fitted against. Changing one
    word of a description changes the probabilities Laya returns, so a gate fitted
    on the old wording must stop applying the moment the wording changes.
    """
    with open(QUESTION_PATH, "rb") as handle:
        raw = handle.read()
    spec = json.loads(raw.decode("utf-8"))
    return {
        "version": spec["version"],
        "fingerprint": hashlib.sha256(raw).hexdigest(),
        "question_id": spec["questionId"],
        "questions": {
            spec["questionId"]: {
                "type": "choice",
                "instructions": spec["instructions"],
                "criteria": spec["criteria"],
            }
        },
    }


def _rss_mb() -> Optional[float]:
    """Current resident memory of this process, where the OS will say."""
    try:
        with open("/proc/self/status", "r", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    return None


def _peak_rss_mb() -> Optional[float]:
    try:
        import resource  # POSIX only
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return round(peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0, 1)


class IntentEngine:
    """One Laya Router, built once, answering one question per turn."""

    def __init__(self) -> None:
        self.model_dir = os.environ.get("ARJUN_LAYA_DIR", "")
        self.device = (os.environ.get("ARJUN_LAYA_DEVICE") or "cpu").strip().lower()
        self.threads = self._thread_count()
        self.question = load_question()
        self.router = None
        self.installed: List[str] = []
        self.load_ms: Optional[float] = None

    @staticmethod
    def _thread_count() -> int:
        raw = os.environ.get("ARJUN_LAYA_THREADS", "").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        # Laya's BENCHMARKS.md, laptop CPU: the best intra-op setting was the
        # physical core count, and SMT siblings contended. os.cpu_count() counts
        # logical CPUs, so half of it approximates the physical cores.
        return max(1, min(8, (os.cpu_count() or 2) // 2))

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> Dict[str, Any]:
        if self.router is not None:
            return self.status()
        if not self.model_dir or not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                "The Laya intent model is not installed. Expected the convaiinnovations/laya "
                "bundle at %r (set ARJUN_LAYA_DIR)." % self.model_dir
            )

        self.installed = [
            name
            for name, sub in CHECKPOINTS.items()
            if os.path.isfile(os.path.join(self.model_dir, sub or "", "rl_agent_config.json"))
        ]
        if "english" not in self.installed:
            raise FileNotFoundError(
                "%r holds no Laya checkpoint (rl_agent_config.json is missing at its root)."
                % self.model_dir
            )

        started = time.perf_counter()
        import torch

        # Laya's BENCHMARKS.md: one forward pass per call leaves inter-op
        # parallelism nothing to overlap, and torch's default of several
        # inter-op threads made a three-question call 12x slower on a laptop.
        torch.set_num_threads(self.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Only settable before the first parallel region; already set is fine.
            pass

        from laya import Router

        models = {name: (self.model_dir, CHECKPOINTS[name]) for name in self.installed}
        router = Router(models=models, device=self.device, max_loaded=len(models))
        router.preload(self.installed)
        for name in self.installed:
            self._check_budget(name, router.load(name))
        # One throwaway forward pass per checkpoint before reporting ready. The
        # first pass pays for allocator warm-up and lazy initialisation, and it
        # would otherwise land on somebody's first turn -- where the parent's
        # deadline would turn it into a keyword fallback for no reason.
        for name in self.installed:
            router.predict({"request": "hello"}, self.question["questions"], model=name)
        self.router = router
        self.load_ms = round((time.perf_counter() - started) * 1000.0, 1)
        return self.status()

    def _check_budget(self, name: str, agent: Any) -> None:
        """Refuse to run with option descriptions the checkpoint would cut short.

        Mirrors `laya.common.build_sequence`: each option is `[MASK] label: text`,
        capped at 48 tokens, and when the options leave fewer than 16 tokens of
        the head budget -- or fewer than the instruction needs -- every option and
        then the instruction are truncated in place.
        """
        from laya.common import render_options

        spec = self.question["questions"][self.question["question_id"]]
        internal = {"t": "choice", "ins": spec["instructions"], "crit": spec["criteria"]}
        tok = agent.tok
        head_max_len = int(agent.cfg.get("head_max_len", 192))
        head = tok("choice question: %s" % spec["instructions"], add_special_tokens=False)["input_ids"]
        options = []
        for text in render_options(internal):
            ids = tok(" " + text, add_special_tokens=False)["input_ids"]
            if len(ids) > 48:
                raise ValueError(
                    "the %s checkpoint would cut the option %r at 48 tokens (it needs %d); "
                    "shorten it in intent_question.json" % (name, text.split(":", 1)[0], len(ids))
                )
            options.append(1 + len(ids))  # the [MASK] marker
        room = head_max_len - sum(options)
        if room < max(16, len(head)):
            raise ValueError(
                "the %s checkpoint has a %d-token head budget; the six options use %d and the "
                "instruction %d, so Laya would truncate them. Shorten intent_question.json."
                % (name, head_max_len, sum(options), len(head))
            )

    def status(self) -> Dict[str, Any]:
        cuda_mb = cuda_peak_mb = None
        if self.router is not None and self.device.startswith("cuda"):
            import torch

            if torch.cuda.is_available():
                cuda_mb = round(torch.cuda.memory_allocated() / 2**20, 1)
                cuda_peak_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        try:
            from laya import __version__ as laya_version
        except ImportError:
            laya_version = None
        return {
            "loaded": self.router is not None,
            "installed": self.installed,
            "device": self.device,
            "threads": self.threads,
            "loadMs": self.load_ms,
            "layaVersion": laya_version,
            "questionVersion": self.question["version"],
            "questionFingerprint": self.question["fingerprint"],
            "rssMb": _rss_mb(),
            "peakRssMb": _peak_rss_mb(),
            "cudaAllocatedMb": cuda_mb,
            "cudaPeakMb": cuda_peak_mb,
        }

    # -- the one question ------------------------------------------------------

    def classify(self, prompt: str, checkpoint: Optional[str] = None) -> Dict[str, Any]:
        if self.router is None:
            raise RuntimeError("intent.classify before intent.load")
        if checkpoint is not None and checkpoint not in self.installed:
            raise FileNotFoundError(
                "the %s Laya checkpoint is not installed, so this prompt cannot be read by it"
                % checkpoint
            )

        qid = self.question["question_id"]
        state = {"request": prompt}
        questions = self.question["questions"]
        started = time.perf_counter()
        # With no checkpoint named, Laya's own script and language detection
        # chooses; ARJUN names one only where that detection is known to be
        # wrong -- romanized Hindi, which it reads as English. The decision is
        # taken before the forward pass so that a checkpoint which is not
        # installed is refused here, by name, rather than reached for.
        if checkpoint is None:
            decision = self.router.route(state, questions)
            used, reason = decision["model"], decision["reason"]
        else:
            used, reason = checkpoint, "requested by ARJUN's language detection"
        if used not in self.installed:
            raise FileNotFoundError(
                "Laya routed this prompt to the %s checkpoint, which is not installed" % used
            )
        result = self.router.predict(state, questions, model=used)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        answer = result["answers"][qid]
        return {
            "choice": answer["choice"],
            "probabilities": answer["probabilities"],
            # max(p) after temperature scaling: the quantity Laya's calibration
            # fits and its ECE figures measure. `confidence` on a choice answer is
            # normalised entropy, which Laya's own common.py says must not be
            # compared against the same threshold.
            "answerConfidence": answer.get("answer_confidence"),
            "entropyConfidence": answer.get("confidence"),
            "checkpoint": used,
            "checkpointReason": reason,
            "inputTokens": (result.get("usage") or {}).get("input_tokens"),
            "latencyMs": round(elapsed_ms, 2),
        }
