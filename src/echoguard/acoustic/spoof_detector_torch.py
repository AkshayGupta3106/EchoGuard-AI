"""PyTorch AASIST-L file/array and rolling waveform inference (MIT model license).

Uses mono 16 kHz raw audio, tiled/truncated to 64,600 samples (~4.04 seconds).
Logit index 0 is spoof, index 1 bonafide; spoof_score is a probability in [0, 1].
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from echoguard.paths import ACOUSTIC_MODEL_DIR

_MODEL_DIR = ACOUSTIC_MODEL_DIR

# Make the vendored models/AASIST.py importable.
sys.path.insert(0, str(_MODEL_DIR))

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


class SpoofDetector:
    """One-shot array and file inference using the pretrained checkpoint."""

    def __init__(self,
                 weights_path: Optional[Path] = None,
                 config_path: Optional[Path] = None,
                 device: str = "cpu"):
        from models.AASIST import Model  # vendored copy, see assets/acoustic/models/

        weights_path = weights_path or (_MODEL_DIR / "models" / "AASIST-L.pth")
        config_path = config_path or (_MODEL_DIR / "AASIST-L.conf")

        with open(config_path) as f:
            config = json.load(f)

        self.device = torch.device(device)
        self.model = Model(config["model_config"])
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_array(self, samples: np.ndarray) -> dict:
        """samples: 1-D float32 array, 16kHz mono, range roughly [-1, 1]."""
        x = _pad_or_tile(samples.astype(np.float32))
        x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)  # [1, 64600]

        start = time.time()
        _, logits = self.model(x_t)
        elapsed_ms = (time.time() - start) * 1000

        probs = F.softmax(logits, dim=1)[0]
        spoof_prob = float(probs[0])
        bonafide_prob = float(probs[1])

        return {
            "spoof_score": round(spoof_prob, 4),
            "bonafide_score": round(bonafide_prob, 4),
            "spoof_margin": float(logits[0, 0] - logits[0, 1]),
            "inference_ms": round(elapsed_ms, 1),
        }

    def predict_file(self, wav_path: str) -> dict:
        import soundfile as sf
        samples, sr = sf.read(wav_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)  # downmix to mono
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"Expected {SAMPLE_RATE}Hz audio, got {sr}Hz. Resample first "
                f"(e.g. with soundfile+resampy, or ffmpeg -ar 16000)."
            )
        result = self.predict_array(samples)
        result["source_file"] = wav_path
        return result


class RollingSpoofScorer:
    """
    Retains the last WINDOW_SAMPLES supplied by the caller and re-scores
    on demand. This class does not schedule inference or apply VAD gating.
    """

    def __init__(self, detector: Optional[SpoofDetector] = None):
        self.detector = detector or SpoofDetector()
        self._buffer: deque = deque(maxlen=WINDOW_SAMPLES)

    def push(self, samples: np.ndarray) -> None:
        """Append a finite mono 16 kHz waveform chunk."""
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("Audio chunks must be finite, one-dimensional mono waveforms")
        self._buffer.extend(samples.tolist())

    def score(self) -> dict:
        """Score buffered audio on demand, tiling short input to the model window."""
        if not self._buffer:
            return {"spoof_score": 0.0, "bonafide_score": 1.0, "inference_ms": 0.0,
                     "note": "no audio buffered yet"}
        arr = np.array(self._buffer, dtype=np.float32)
        return self.detector.predict_array(arr)


# Inference smoke check: python -m echoguard.acoustic.spoof_detector_torch
if __name__ == "__main__":
    print("Loading AASIST-L (this is the actual pretrained checkpoint, "
          "not a stub)...")
    detector = SpoofDetector()
    n_params = sum(p.numel() for p in detector.model.parameters())
    print(f"Loaded. Parameter count: {n_params:,} "
          f"(paper reports 85,306 for AASIST-L - should match)")

    # Test 1: pure silence
    silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)  # 2s of silence
    print("\n--- Test: silence ---")
    print(detector.predict_array(silence))

    # Test 2: white noise (not real spoofed speech, just checking the
    # pipeline runs end-to-end on non-trivial input)
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(SAMPLE_RATE * 2) * 0.05).astype(np.float32)
    print("\n--- Test: low-amplitude noise ---")
    print(detector.predict_array(noise))

    # Test 3: rolling scorer, fed in small chunks like a live stream would
    print("\n--- Test: RollingSpoofScorer fed in 0.5s chunks ---")
    scorer = RollingSpoofScorer(detector=detector)
    chunk = (rng.standard_normal(SAMPLE_RATE // 2) * 0.05).astype(np.float32)
    for i in range(4):
        scorer.push(chunk)
        print(f"after {0.5*(i+1)}s of audio: {scorer.score()}")

    print("\nNOTE: silence/noise scores above are pipeline sanity checks "
          "only - they say nothing about real accuracy. The test that "
          "actually matters is running predict_file() against your real "
          "demo clone clip once you have it recorded.")
