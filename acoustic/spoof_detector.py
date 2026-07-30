"""
spoof_detector.py
------------------
Owner: Person A (acoustic stream)

Wraps the official pretrained AASIST-L checkpoint (clovaai/aasist, MIT
license) to produce spoof_score() - a float in [0, 1], higher = more likely
an AI-generated / cloned voice.

IMPORTANT correction vs. the original rough plan: AASIST-L does NOT take a
mel-spectrogram. It takes raw 16kHz waveform directly (it has its own
SincConv front-end baked in) - there is no separate spectrogram-extraction
step to write. If you already started building a librosa mel-spectrogram
pipeline, you can drop that step entirely for this model.

Two ways to use this:
  1. SpoofDetector.predict_file(wav_path)   - one-shot, for testing against
     your demo clone clip (the most important Day-1 test in the whole plan).
  2. RollingSpoofScorer.push(samples)       - for the live pipeline: feed it
     small audio chunks continuously (post-VAD), it maintains the rolling
     ~4-second window AASIST-L expects internally and re-scores on demand.

Model details (confirmed by loading the actual checkpoint):
  - 85,306 parameters total.
  - Expects exactly 64,600 samples (~4.04s at 16kHz) per inference call.
    Shorter audio is tiled (repeated) to fill the window, same as the
    original authors' eval-time preprocessing - not a shortcut we invented.
  - Output: 2 logits, index 0 = spoof, index 1 = bonafide (confirmed against
    the label convention used in the original repo's data loader and scoring
    code: label 1 = bonafide, and the repo scores using the bonafide-class
    logit directly).
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

_THIS_DIR = Path(__file__).parent
_MODEL_DIR = _THIS_DIR / "aasist_model"

# Make the vendored models/AASIST.py importable.
sys.path.insert(0, str(_MODEL_DIR))

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


class SpoofDetector:
    """One-shot / file-based inference. Use this for the Day-1 test against
    your actual demo clone clip before wiring up the live rolling version."""

    def __init__(self,
                 weights_path: Optional[Path] = None,
                 config_path: Optional[Path] = None,
                 device: str = "cpu"):
        from models.AASIST import Model  # vendored copy, see aasist_model/models/

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
    Live-pipeline version. Feed it small chunks continuously (post-VAD);
    it keeps a rolling buffer of the last WINDOW_SAMPLES and re-scores
    whenever score() is called - matches the "updated every 1-2s" cadence
    from the architecture, driven by however often the caller invokes
    score(), not by this class itself.
    """

    def __init__(self, detector: Optional[SpoofDetector] = None):
        self.detector = detector or SpoofDetector()
        self._buffer: deque = deque(maxlen=WINDOW_SAMPLES)

    def push(self, samples: np.ndarray) -> None:
        """samples: 1-D float32 array, 16kHz mono chunk from the mic/VAD."""
        self._buffer.extend(samples.tolist())

    def score(self) -> dict:
        """Call this every 1-2 seconds from your main loop. If less than
        ~4s of audio has been collected yet (e.g. right after the VAD gate
        opens), the buffer is tiled to fill the window - same behavior as
        the file-based path, just applied to whatever's been captured so far."""
        if not self._buffer:
            return {"spoof_score": 0.0, "bonafide_score": 1.0, "inference_ms": 0.0,
                     "note": "no audio buffered yet"}
        arr = np.array(self._buffer, dtype=np.float32)
        return self.detector.predict_array(arr)


# ---------------------------------------------------------------------------
# Self-test - loads the REAL pretrained checkpoint and runs REAL inference.
# Run directly: python3 spoof_detector.py
# ---------------------------------------------------------------------------
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
