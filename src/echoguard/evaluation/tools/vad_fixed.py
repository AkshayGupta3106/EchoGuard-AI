"""
vad_fixed.py -- Silero VAD wrapper with the 64-sample context the model needs.

Evaluation helper for the Silero input/state/sr contract. With use_context=True,
prepend the last 64 samples to each 512-sample chunk (576 samples at 16 kHz).
The context and recurrent state are reset between recordings. use_context=False
supports measuring context-free behavior. This helper is not the runtime VAD.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
CHUNK = 512
CONTEXT = 64


class VadGateFixed:
    def __init__(self, model_path, threshold: float = 0.5, use_context: bool = True):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.use_context = use_context
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros(CONTEXT, dtype=np.float32)

    def prob(self, chunk: np.ndarray) -> float:
        if chunk.shape[0] < CHUNK:
            chunk = np.pad(chunk, (0, CHUNK - chunk.shape[0]))
        chunk = chunk[:CHUNK].astype(np.float32)
        x = np.concatenate([self._ctx, chunk]) if self.use_context else chunk
        p, self._state = self.session.run(None, {"input": x[None, :], "state": self._state, "sr": self._sr})
        self._ctx = chunk[-CONTEXT:]
        return float(p[0][0])

    def speech_rate(self, samples: np.ndarray):
        """(fraction of chunks >= threshold, max prob, n_chunks) for a whole clip."""
        self.reset()
        probs = [self.prob(samples[i:i + CHUNK]) for i in range(0, len(samples) - CHUNK + 1, CHUNK)]
        if not probs:
            return 0.0, 0.0, 0
        a = np.array(probs)
        return float((a >= self.threshold).mean()), float(a.max()), len(a)
