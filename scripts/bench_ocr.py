#!/usr/bin/env python3
"""Unlimited-OCR measurement harness: quantisation tier x slider stop.

What this measures
------------------
Eight cells - two installed weight files (Q6_K, Q4_K_M) crossed with the four
slider stops - against pages whose text is known. For each cell:

  * wall-clock milliseconds per page and real tokens/second
  * peak VRAM, when nvidia-smi is present
  * character error rate against ground truth
  * whether the run LOOPED
  * how many boxes landed outside the page

The last two are the point of the exercise, not extras.

Why looping gets its own column
-------------------------------
Baidu's reference implementation runs a no-repeat-ngram logit processor
(`no_repeat_ngram_size=35`, window 128) that llama.cpp does not have.
Independent testing found Q4_K_M loops forever on some prompts without it.
ARJUN substitutes llama.cpp's DRY sampler, which is an approximation of that
processor and not a reproduction of it. Whether the approximation holds is
exactly what this harness exists to find out, so a looped run is recorded as a
looped run rather than averaged into a latency figure that looks fine.

Why boxes-off-page gets its own column
--------------------------------------
The model's coordinates are either normalised 0-999 or input pixels, and the
two are only distinguishable empirically. Under the wrong assumption boxes
land outside the page. Counting them turns a silent rendering bug into a
number on a sheet.

No fallbacks
------------
If the binary is missing, this exits non-zero and writes nothing. It does not
substitute a plausible constant for a measurement it did not take. A previous
version of the sibling `bench.py` returned a hardcoded 38 tok/s when its
binding failed to import, and that constant was published as a measured
benchmark. Measure, or fail loudly.

Usage
-----
    python scripts/bench_ocr.py \\
        --models "%LOCALAPPDATA%/com.arjun.workbench/models/OCR/Unlimited-OCR" \\
        --cli    "path/to/llama-mtmd-cli.exe" \\
        --pages  scratch/phase0/pages \\
        --truth  scratch/phase0/page_text.json \\
        --out    docs/sih/ocr-benchmarks.csv
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECTOR = "mmproj-Unlimited-OCR-F16.gguf"

# (detent, weights file, --image-max-tokens, -n). Mirrors
# ai_engine::ocr_profile; if the two ever disagree, this sheet is describing a
# configuration the product does not ship.
CELLS = [
    ("fastest", "Unlimited-OCR-Q4_K_M.gguf", 100, 2048),
    ("fast", "Unlimited-OCR-Q4_K_M.gguf", 256, 4096),
    ("detailed", "Unlimited-OCR-Q6_K.gguf", 256, 8192),
    ("maximum", "Unlimited-OCR-Q6_K.gguf", 400, 16384),
]

PROMPT = "<|grounding|>Convert the document to markdown."

# Baidu's own post-processing, used so the error rate is comparable to the
# numbers they publish rather than to a stricter measure of our own devising.
DET_RE = re.compile(r"<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)", re.DOTALL)
BOX_RE = re.compile(r"<\|det\|>[^<\[]*\[([^\]]*)\]\s*<\|/det\|>")


def remove_det(raw: str) -> str:
    """Strip detection markers, keeping the text in reading order."""
    blocks, cur = [], None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = DET_RE.match(line)
        if m:
            category, content = m.group(1).strip(), m.group(2).strip()
            if category == "image":
                continue
            if cur is not None:
                blocks.append(cur)
            cur = [content] if content else []
            continue
        if cur is None:
            cur = []
        cur.append(line)
    if cur is not None:
        blocks.append(cur)
    return "\n\n".join("\n".join(b) for b in blocks).strip()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def character_error_rate(truth: str, got: str) -> float:
    """Edit distance over the length of the truth, as a percentage."""
    if not truth:
        return 0.0
    return 100.0 * levenshtein(truth, got) / len(truth)


def looks_looped(text: str, decode_cap: int, tokens: int, n: int = 35) -> bool:
    """A run that filled its budget while repeating an n-gram.

    Both halves matter. Repetition alone is normal in a table; hitting the
    decode cap alone can just mean a dense page. Together they are the
    signature of the failure the missing sampler causes.
    """
    if tokens < decode_cap:
        return False
    words = text.split()
    if len(words) < n * 2:
        return False
    seen = set()
    for i in range(len(words) - n + 1):
        gram = " ".join(words[i : i + n])
        if gram in seen:
            return True
        seen.add(gram)
    return False


def boxes_off_page(raw: str, width: int, height: int) -> int:
    """Emitted boxes that cannot be on a page of this size, read as pixels."""
    off = 0
    for match in BOX_RE.finditer(raw):
        try:
            nums = [int(p.strip()) for p in match.group(1).split(",")]
        except ValueError:
            continue
        if len(nums) != 4:
            continue
        x1, y1, x2, y2 = nums
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 > x2 or y1 > y2:
            off += 1
    return off


def nvidia_smi_vram_mib():
    """Peak VRAM, or None when nvidia-smi is not present.

    None, not zero: a machine without the tool has an unknown figure, and
    zero would read as a measurement.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        return int(out.decode("utf-8").strip().splitlines()[0])
    except Exception:
        return None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_cell(cli: Path, weights: Path, projector: Path, image: Path,
             image_tokens: int, decode: int):
    """One page through one configuration. Returns (raw output, seconds)."""
    cmd = [
        str(cli),
        "-m", str(weights),
        "--mmproj", str(projector),
        "--image", str(image),
        "-p", PROMPT,
        "--temp", "0",
        "-n", str(decode),
        "-c", "8192",
        "--image-max-tokens", str(image_tokens),
        "--dry-multiplier", "0.8",
        "--dry-allowed-length", "35",
        "--dry-penalty-last-n", "128",
    ]
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    elapsed = time.time() - started
    if proc.returncode != 0:
        raise RuntimeError(
            "llama-mtmd-cli exited %d for %s on %s. stderr tail:\n%s"
            % (proc.returncode, weights.name, image.name, proc.stderr[-2000:])
        )
    return proc.stdout, elapsed


def main() -> int:
    p = argparse.ArgumentParser(prog="bench_ocr")
    p.add_argument("--models", required=True, help="Directory holding the GGUFs and the projector")
    p.add_argument("--cli", required=True, help="Path to llama-mtmd-cli")
    p.add_argument("--pages", required=True, help="Directory of page images")
    p.add_argument("--truth", required=True, help="JSON: {page_file: expected_text}")
    p.add_argument("--out", default="docs/sih/ocr-benchmarks.csv")
    args = p.parse_args()

    models, cli = Path(args.models), Path(args.cli)
    pages, truth_path, out_path = Path(args.pages), Path(args.truth), Path(args.out)

    # Every precondition is checked before anything runs, so a missing file
    # cannot produce a half-populated sheet that looks like a complete one.
    if not cli.exists():
        sys.stderr.write(
            "[bench-ocr] llama-mtmd-cli not found at %s.\n"
            "[bench-ocr] Nothing is written: an unmeasured benchmark is not a benchmark.\n"
            % cli
        )
        return 2
    projector = models / PROJECTOR
    needed = set(c[1] for c in CELLS) | {PROJECTOR}
    missing = [f for f in sorted(needed) if not (models / f).exists()]
    if missing:
        sys.stderr.write("[bench-ocr] missing weight files in %s: %s\n" % (models, missing))
        return 2
    if not truth_path.exists():
        sys.stderr.write(
            "[bench-ocr] no ground truth at %s.\n"
            "[bench-ocr] Accuracy cannot be scored against nothing, so no row is written.\n"
            % truth_path
        )
        return 2

    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    images = sorted(q for q in pages.glob("*.png"))
    if not images:
        sys.stderr.write("[bench-ocr] no page images in %s\n" % pages)
        return 2

    digests = {f: sha256_of(models / f) for f in set(c[1] for c in CELLS)}

    rows, failures = [], 0
    for detent, weights_file, image_tokens, decode in CELLS:
        for image in images:
            expected = truth.get(image.name)
            if expected is None:
                sys.stderr.write("[bench-ocr] no ground truth for %s; skipped\n" % image.name)
                continue
            try:
                raw, elapsed = run_cell(
                    cli, models / weights_file, projector, image, image_tokens, decode
                )
            except RuntimeError as error:
                # Recorded, not swallowed: a configuration that cannot run is
                # a result about that configuration.
                sys.stderr.write("[bench-ocr] %s\n" % error)
                failures += 1
                continue

            text = remove_det(raw)
            tokens = len(raw.split())
            rows.append({
                "tier": "Q6_K" if "Q6_K" in weights_file else "Q4_K_M",
                "detent": detent,
                "page": image.name,
                "vision_tokens": image_tokens,
                "decode_tokens": decode,
                "wall_ms": int(elapsed * 1000),
                "tokens_per_second": round(tokens / elapsed, 2) if elapsed > 0 else 0.0,
                "vram_peak_mib": nvidia_smi_vram_mib(),
                "cer_pct": round(character_error_rate(expected, text), 2),
                "looped": looks_looped(text, decode, tokens),
                "regions": len(BOX_RE.findall(raw)),
                "boxes_off_page": boxes_off_page(raw, 1000, 1400),
                "model_sha256": digests[weights_file],
            })
            print("[bench-ocr] %-9s %-24s %7dms  cer=%6.2f%%  looped=%s"
                  % (detent, image.name, rows[-1]["wall_ms"],
                     rows[-1]["cer_pct"], rows[-1]["looped"]))

    if not rows:
        sys.stderr.write("[bench-ocr] every cell failed; nothing measured, nothing written.\n")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    looped = sum(1 for r in rows if r["looped"])
    print("\n[bench-ocr] %d cells written to %s" % (len(rows), out_path))
    print("[bench-ocr] %d looped, %d failed to run" % (looped, failures))
    if looped:
        print("[bench-ocr] NOTE: looping means the DRY substitute is not holding "
              "for that configuration. Do not quote its latency as a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
