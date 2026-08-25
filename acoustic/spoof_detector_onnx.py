"""
spoof_detector_onnx.py
-----------------------
Drop-in replacement for SpoofDetector / RollingSpoofScorer (spoof_detector.py)
that runs on ONNX Runtime instead of PyTorch, for use by the Render web
service specifically.

Why this exists: `import torch` alone uses ~490MB RSS on this environment --
almost the entire 512MB free-tier RAM budget, before FastAPI, uvicorn, or a
single request is handled. `import onnxruntime` uses ~40MB. That gap was
causing the deployed instance to get OOM-killed and restart in a loop under
normal use (visible in Render logs as repeated "Started server process" /
"[startup] loading AASIST-L" cycles a couple minutes apart), which in turn
wiped in-memory session state and produced the cascading 502/503/404 errors
seen in the browser console.

This module is numerically verified against the original PyTorch model
(see acoustic/export_onnx.py and the verification run that produced this
file) -- outputs match to floating-point noise (~1e-7) across random test
inputs, since it's the exact same weights, just a different runtime.

Requires aasist_model/aasist_l.onnx to exist (produced by export_onnx.py,
which itself needs torch -- run that once in a build stage that has torch
installed, then only ship this file + the .onnx artifact + onnxruntime in
the final image; see webdemo/Dockerfile's multi-stage build).

Public interface intentionally mirrors spoof_detector.py's SpoofDetector /
RollingSpoofScorer exactly, so callers (e.g. webdemo/app.py) don't need to
change anything except the import line.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

_THIS_DIR = Path(__file__).parent
_MODEL_DIR = _THIS_DIR / "aasist_model"

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 64600  # ~4.04s - fixed by the model architecture, not tunable


def _pad_or_tile(x: np.ndarray, max_len: int = WINDOW_SAMPLES) -> np.ndarray:
    """Same preprocessing as the original AASIST eval pipeline: truncate if
    long enough, otherwise tile (repeat) the audio to fill the window."""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


def _softmax(logits: np.ndarray) -> np.ndarray:
    ex = np.exp(logits - logits.max())
    return ex / ex.sum()


class SpoofDetector:
    """One-shot / file-based inference, same interface as the torch version."""

    def __init__(self, onnx_path: Optional[Path] = None):
        onnx_path = onnx_path or (_MODEL_DIR / "aasist_l.onnx")
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"{onnx_path} not found. Run acoustic/export_onnx.py once "
                f"(in an environment with torch installed) to produce it."
            )
        # Single-threaded: this is a tiny model (85k params), and capping
        # thread pool size keeps memory/CPU contention low on a constrained
        # instance instead of onnxruntime spinning up threads per core.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name

    def predict_array(self, samples: np.ndarray) -> dict:
        """samples: 1-D float32 array, 16kHz mono, range roughly [-1, 1]."""
        x = _pad_or_tile(samples.astype(np.float32))[None, :]  # [1, 64600]

        start = time.time()
        _, logits = self.session.run(None, {self._input_name: x})
        elapsed_ms = (time.time() - start) * 1000

        probs = _softmax(logits[0])
        spoof_prob = float(probs[0])
        bonafide_prob = float(probs[1])

        return {
            "spoof_score": round(spoof_prob, 4),
            "bonafide_score": round(bonafide_prob, 4),
            "inference_ms": round(elapsed_ms, 1),
        }

    def predict_file(self, wav_path: str) -> dict:
        import soundfile as sf
        samples, sr = sf.read(wav_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"Expected {SAMPLE_RATE}Hz audio, got {sr}Hz. Resample first "
                f"(e.g. with soundfile+resampy, or ffmpeg -ar 16000)."
            )
        result = self.predict_array(samples)
        result["source_file"] = wav_path
        return result


class RollingSpoofScorer:
    """Identical behavior to the torch version's RollingSpoofScorer."""

    def __init__(self, detector: Optional[SpoofDetector] = None):
        self.detector = detector or SpoofDetector()
        self._buffer: deque = deque(maxlen=WINDOW_SAMPLES)

    def push(self, samples: np.ndarray) -> None:
        self._buffer.extend(samples.tolist())

    def score(self) -> dict:
        if not self._buffer:
            return {"spoof_score": 0.0, "bonafide_score": 1.0, "inference_ms": 0.0,
                     "note": "no audio buffered yet"}
        arr = np.array(self._buffer, dtype=np.float32)
        return self.detector.predict_array(arr)
