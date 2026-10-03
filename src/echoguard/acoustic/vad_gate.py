"""Silero ONNX speech probability with recurrent state and 512-sample chunks.

Uses mono 16 kHz audio, pads/trims input, and carries state [2, 1, 128] between
calls. No inter-chunk audio context is prepended; callers decide whether to gate
audio on the returned flag. Reset state between sessions. Model license: MIT.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from echoguard.paths import ACOUSTIC_MODEL_DIR

_MODEL_PATH = ACOUSTIC_MODEL_DIR / "silero_vad.onnx"

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # 32ms at 16kHz - the size this model expects


class VadGate:
    def __init__(self, model_path: Path = _MODEL_PATH, threshold: float = 0.5):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def reset(self):
        """Clear recurrent state between independent audio sessions."""
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
        self._state = new_state

        p = float(prob[0][0])
        return {"speech_prob": round(p, 4), "is_speech": p >= self.threshold}


# Inference smoke check: python -m echoguard.acoustic.vad_gate
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

    print("\nNOTE: this smoke check exercises VAD inference on synthetic inputs; "
          "use consented speech/silence recordings to evaluate speech coverage.")
