"""
vad_gate.py
------------------
Owner: Person A (acoustic stream)

Wraps the official Silero VAD ONNX model (snakers4/silero-vad, MIT license)
to gate audio before it reaches AASIST-L and the semantic stream (Person B's
Zipformer). This is the "NEW" component from the architecture update - it
cuts wasted compute on silence/hold-music by only letting speech segments
through to the two heavier models downstream.

Confirmed I/O contract by loading the actual model (not documentation -
the model file itself):
  inputs:  "input" [1, N] float32 audio chunk
           "state"  [2, 1, 128] float32 - recurrent state, carried between
                     calls (this is the detail people most often get wrong -
                     forgetting to feed the updated state back in makes the
                     VAD much less accurate)
           "sr"     scalar int64 sample rate (16000)
  outputs: "output"  [1, 1] float32 speech probability
           "stateN"  [2, 1, 128] float32 - new state, feed into next call

Standard chunk size for this model is 512 samples (32ms) at 16kHz - feed it
consistently at that size for best results, matching how it was trained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

_THIS_DIR = Path(__file__).parent
_MODEL_PATH = _THIS_DIR / "aasist_model" / "silero_vad.onnx"

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # 32ms at 16kHz - the size this model expects


class VadGate:
    def __init__(self, model_path: Path = _MODEL_PATH, threshold: float = 0.5):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def reset(self):
        """Call this at the start of each new call/session - stale recurrent
        state from a previous call will skew the first few predictions."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, chunk: np.ndarray) -> dict:
        """chunk: 1-D float32 array, ideally exactly CHUNK_SAMPLES long.
        Returns the speech probability and a boolean gate decision."""
        if chunk.shape[0] != CHUNK_SAMPLES:
            # Pad or trim rather than erroring - live audio callbacks don't
            # always hand you exactly 512 samples.
            if chunk.shape[0] < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - chunk.shape[0]))
            else:
                chunk = chunk[:CHUNK_SAMPLES]

        x = chunk.astype(np.float32)[np.newaxis, :]
        prob, new_state = self.session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._state = new_state  # carry state forward - see docstring above

        p = float(prob[0][0])
        return {"speech_prob": round(p, 4), "is_speech": p >= self.threshold}


# ---------------------------------------------------------------------------
# Self-test - loads the REAL Silero VAD model.
# Run directly: python3 vad_gate.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading Silero VAD (real ONNX model, not a stub)...")
    vad = VadGate()

    rng = np.random.default_rng(0)
    silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    noise = (rng.standard_normal(CHUNK_SAMPLES) * 0.3).astype(np.float32)

    print("\n--- Test: silence chunk ---")
    print(vad.is_speech(silence))

    print("\n--- Test: loud noise chunk (not real speech, just non-trivial signal) ---")
    print(vad.is_speech(noise))

    print("\nNOTE: like the AASIST-L test, this only confirms the pipeline runs "
          "end-to-end with the real model - test against actual recorded "
          "speech/silence on Day 1, not synthetic noise.")
