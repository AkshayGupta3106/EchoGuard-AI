"""ONNX Runtime AASIST-L inference for mono 16 kHz waveform.

Loads the self-contained assets/acoustic/aasist_l.onnx export.
SpoofDetector provides file/array inference; RollingSpoofScorer retains the
latest 64,600 samples. Inputs are truncated or tiled to the model window.
This backend is used by the web service without importing PyTorch.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from echoguard.paths import ACOUSTIC_MODEL_DIR

_MODEL_DIR = ACOUSTIC_MODEL_DIR

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 64600  # ~4.04s - fixed by the model architecture, not tunable


def _pad_or_tile(x: np.ndarray, max_len: int = WINDOW_SAMPLES) -> np.ndarray:
    """AASIST window preprocessing: truncate if
    long enough, otherwise tile (repeat) the audio to fill the window."""
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Audio must be a nonempty, finite, one-dimensional mono waveform")
    if max_len <= 0:
        raise ValueError("Audio window length must be positive")
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


def _softmax(logits: np.ndarray) -> np.ndarray:
    ex = np.exp(logits - logits.max())
    return ex / ex.sum()


class SpoofDetector:
    """AASIST-L array/file inference through ONNX Runtime."""

    def __init__(self, onnx_path: Optional[Path] = None):
        onnx_path = onnx_path or (_MODEL_DIR / "aasist_l.onnx")
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"{onnx_path} not found. Run tools/models/export_aasist.py once "
                f"(in an environment with torch installed) to produce it."
            )
        # Limit thread-pool contention on constrained hosts.
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
            "spoof_margin": float(logits[0][0] - logits[0][1]),
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
    """Keep the latest model window and score it on demand."""

    def __init__(self, detector: Optional[SpoofDetector] = None):
        self.detector = detector or SpoofDetector()
        self._buffer: deque = deque(maxlen=WINDOW_SAMPLES)

    def push(self, samples: np.ndarray) -> None:
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("Audio chunks must be finite, one-dimensional mono waveforms")
        self._buffer.extend(samples.tolist())

    def score(self) -> dict:
        if not self._buffer:
            return {"spoof_score": 0.0, "bonafide_score": 1.0, "inference_ms": 0.0,
                     "note": "no audio buffered yet"}
        arr = np.array(self._buffer, dtype=np.float32)
        return self.detector.predict_array(arr)
