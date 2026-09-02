#!/usr/bin/env python3
"""Arjun model certification runner.

Currently a stub that refuses. It measures nothing, so it certifies nothing.

## Why this file no longer produces a report

The previous version advertised a "16-Point Automated Model Certification Test
Suite" and wrote `certification.json`, `certification_report.md` and
`certification_report.html`. It did not load a model. It never opened
`model_path` at all. Every module's score came from one line:

    score = 95.0 if "leakage" in mod or "token" in mod or "chat" in mod else 92.5

so every package averaged 93.0 and every package came out "Certified" --
including package ids naming nothing that exists. The `profile_hash` was a
SHA-256 of the package id string and the `signature` an MD5 of the same, so
both could be produced without a model, a GPU, or a single inference call.

Those generated files were committed to the repository and presented as
evidence of model quality.

A certification runner that cannot measure must not certify. Until the modules
below are implemented against a loaded model, this script exits non-zero.
"""

import argparse
import sys

# The intended checks. A list of names, not an implementation.
TEST_MODULES = [
    "instruction_following",
    "reasoning_quality",
    "hallucination_rate",
    "coding_ability",
    "mathematical_reasoning",
    "json_reliability",
    "tool_calling_accuracy",
    "memory_engine_compatibility",
    "context_window_retention",
    "response_stability",
    "chat_template_correctness",
    "bos_eos_stop_token_compliance",
    "reasoning_tag_leakage_filter",
    "streaming_parser_stability",
    "runtime_process_stability",
    "restart_state_persistence",
]


def run_certification(package_id, model_path, output_dir="."):
    """Refuses, and says exactly what would have to exist for it not to."""
    sys.stderr.write(
        "certification_runner: no benchmark implementation exists.\n"
        "\n"
        f"  requested package : {package_id}\n"
        f"  requested model   : {model_path}\n"
        f"  modules intended  : {len(TEST_MODULES)}\n"
        f"  modules implemented: 0\n"
        "\n"
        "This script previously emitted constant scores without loading the\n"
        "model, and wrote them to certification.json as though they were\n"
        "measurements. That output has been removed from the repository.\n"
        "\n"
        "To restore certification: implement the modules in TEST_MODULES so\n"
        "each one loads the GGUF at --model-path, runs a defined prompt set,\n"
        "and returns a score derived from the model's actual responses. Record\n"
        "the prompt set and the grading rule alongside the score, so a reader\n"
        "can disagree with the method rather than only with the number.\n"
        "\n"
        "No artifact was written.\n"
    )
    return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arjun model certification runner (unimplemented; refuses)"
    )
    parser.add_argument("--package", default=None, help="Target package id")
    parser.add_argument("--model-path", default=None, help="Path to local GGUF file")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()
    sys.exit(run_certification(args.package, args.model_path, args.outdir))
