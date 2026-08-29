#!/usr/bin/env python3
"""Voice sidecar for ARJUN.

Two commands, both with a `--mode` flag that toggles between a stub
(zero dependencies, always works) and a real path (requires local
Whisper and Piper model files).

Stub mode
---------
The stub is the default. The sidecar reads a `--text` (for speak) or
audio bytes on stdin (for transcribe) and returns a deterministic
placeholder. The point of the stub is that the front-end wiring is
testable end-to-end on a machine without a working microphone or
without the model weights downloaded.

Real mode
---------
Set `--mode real` and the sidecar switches to:

  - transcribe: `whisper.cpp` with `ggml-tiny.en.bin` (39 MB, English)
  - speak:      `piper` with `en_US-lessac-medium.onnx` (15 MB)

Both model files are expected in `--model-dir`, which the Tauri
command passes as `<app_data_dir>/voice/`. The sidecar is honest
about the failure mode: a missing model file in real mode causes a
clean JSON error, not a stack trace, so the front-end can show the
operator a "drop the model here" message.

Honest scope
------------
- This is a push-to-talk bridge, not an always-on wake-word
  detector. The continuous-audio path is a multi-day integration
  and is deliberately not in scope for the SIH demo.
- The audio format expected on stdin is 16 kHz mono PCM s16le.
  The front-end should downsample from the browser MediaRecorder
  output before posting. (See voice_bridge.format_audio in the
  Rust module for the format contract.)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REQUIRED_WHISPER_FILE = "ggml-tiny.en.bin"
REQUIRED_PIPER_FILE = "en_US-lessac-medium.onnx"


def has_whisper(model_dir: Path) -> bool:
    return (model_dir / REQUIRED_WHISPER_FILE).exists()


def has_piper(model_dir: Path) -> bool:
    return (model_dir / REQUIRED_PIPER_FILE).exists()


def stub_transcribe(audio_bytes: bytes) -> dict:
    """Return a deterministic placeholder transcript.

    The placeholder carries enough information that a reviewer can
    see the bridge is wired correctly: it reports the audio length
    in milliseconds, the model id, and a short placeholder string.
    """
    audio_ms = len(audio_bytes) // 32  # 16 kHz, 16-bit, mono = 32 B/ms
    return {
        "text": "[stub transcript — drop ggml-tiny.en.bin in the voice dir for real STT]",
        "stub": True,
        "real": False,
        "confidence": None,
        "audioMs": int(audio_ms),
        "modelId": "stub",
    }


def real_transcribe(model_dir: Path) -> dict:
    """Load whisper.cpp and transcribe.

    Imported lazily so the stub path does not require the user to
    install the heavier dependencies. The error path is JSON, not
    a Python traceback, so the front-end can show a useful message.
    """
    if not has_whisper(model_dir):
        return {
            "error": (
                "Whisper model not found. Place ggml-tiny.en.bin in "
                f"{model_dir} to enable real transcription."
            )
        }
    try:
        # Imported lazily so the stub does not require whisper-cpp.
        from pywhispercpp.model import Model  # type: ignore
    except Exception as e:
        return {"error": f"pywhispercpp not available: {e}"}
    try:
        audio_bytes = sys.stdin.buffer.read()
        audio_ms = len(audio_bytes) // 32
        # pywhispercpp expects a numpy array of float32 at 16 kHz.
        # The front-end is responsible for converting from the
        # browser's MediaRecorder output; the bridge only validates
        # that the byte count is sane.
        if audio_ms <= 0:
            return {
                "text": "",
                "stub": False,
                "real": True,
                "confidence": None,
                "audioMs": 0,
                "modelId": "whisper-tiny",
            }
        import numpy as np  # type: ignore
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        model = Model(str(model_dir / REQUIRED_WHISPER_FILE), n_threads=4)
        t0 = time.time()
        result = model.transcribe(audio)
        text = " ".join(seg.text for seg in result) if result else ""
        return {
            "text": text,
            "stub": False,
            "real": True,
            "confidence": None,
            "audioMs": int(audio_ms),
            "modelId": "whisper-tiny",
        }
    except Exception as e:
        return {"error": f"transcription failed: {e}"}


def stub_speak(text: str, out: Path) -> None:
    """Write a tiny silent WAV file so the front-end <audio> element
    has something to play. The placeholder proves the round-trip works.
    """
    sample_rate = 16000
    duration_s = 0.5
    n_samples = int(sample_rate * duration_s)
    with open(out, "wb") as f:
        # RIFF/WAVE header
        f.write(b"RIFF")
        f.write((36 + n_samples * 2).to_bytes(4, "little"))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write((16).to_bytes(4, "little"))
        f.write((1).to_bytes(2, "little"))  # PCM
        f.write((1).to_bytes(2, "little"))  # mono
        f.write(sample_rate.to_bytes(4, "little"))
        f.write((sample_rate * 2).to_bytes(4, "little"))
        f.write((2).to_bytes(2, "little"))
        f.write((16).to_bytes(2, "little"))
        f.write(b"data")
        f.write((n_samples * 2).to_bytes(4, "little"))
        f.write(b"\x00\x00" * n_samples)


def real_speak(model_dir: Path, out: Path, text: str) -> None:
    if not has_piper(model_dir):
        raise FileNotFoundError(
            f"Piper voice not found. Place en_US-lessac-medium.onnx in {model_dir}."
        )
    try:
        from piper import PiperVoice  # type: ignore
    except Exception as e:
        raise RuntimeError(f"piper-tts not available: {e}")
    voice = PiperVoice.load(str(model_dir / REQUIRED_PIPER_FILE))
    with open(out, "wb") as f:
        voice.synthesize(text, f)


def main() -> int:
    p = argparse.ArgumentParser(prog="voice_bridge")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe")
    t.add_argument("--model-dir", required=True)
    t.add_argument("--mode", choices=("stub", "real"), default="stub")
    t.add_argument("--stdin", action="store_true",
                   help="Read audio from stdin (always true in practice)")

    s = sub.add_parser("speak")
    s.add_argument("--model-dir", required=True)
    s.add_argument("--mode", choices=("stub", "real"), default="stub")
    s.add_argument("--out", required=True)
    s.add_argument("--text", required=True)
    s.add_argument("--language", default="en_US")
    s.add_argument("--speed", type=float, default=1.0)

    args = p.parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "transcribe":
        if args.mode == "stub":
            audio = sys.stdin.buffer.read() if args.stdin else b""
            result = stub_transcribe(audio)
        else:
            result = real_transcribe(model_dir)
        sys.stdout.write(json.dumps(result))
        sys.stdout.write("\n")
        return 0

    if args.cmd == "speak":
        out = Path(args.out)
        try:
            if args.mode == "stub":
                stub_speak(args.text, out)
            else:
                real_speak(model_dir, out, args.text)
        except FileNotFoundError as e:
            sys.stderr.write(str(e) + "\n")
            return 2
        except Exception as e:
            sys.stderr.write(f"speak failed: {e}\n")
            return 1
        sys.stdout.write(json.dumps({"ok": True, "out": str(out)}))
        sys.stdout.write("\n")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
