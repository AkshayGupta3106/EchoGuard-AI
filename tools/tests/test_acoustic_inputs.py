"""Invalid input is rejected without changing valid AASIST window preprocessing."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(params=["spoof_detector_onnx", "spoof_detector_torch"])
def backend(request):
    if request.param.endswith("torch"):
        pytest.importorskip("torch")
    return importlib.import_module("echoguard.acoustic." + request.param)


@pytest.mark.parametrize("samples", [np.zeros(0), np.zeros((2, 2)), np.array([np.nan]), np.array([np.inf])])
def test_invalid_waveforms_raise_clear_error(backend, samples):
    with pytest.raises(ValueError, match="waveform"):
        backend._pad_or_tile(samples)


@pytest.mark.parametrize("length", [1, 100, 16000, 64600, 70000])
def test_valid_window_bytes_unchanged(backend, length):
    samples = np.arange(length, dtype=np.float32)
    expected = samples[:64600] if length >= 64600 else np.tile(samples, int(64600 / length) + 1)[:64600]
    assert backend._pad_or_tile(samples).tobytes() == expected.tobytes()


def test_invalid_chunk_does_not_poison_rolling_buffer(backend):
    scorer = backend.RollingSpoofScorer(detector=SimpleNamespace())
    scorer.push(np.zeros(3))
    for bad in (np.zeros((2, 2)), np.array([np.nan])):
        with pytest.raises(ValueError, match="chunks"):
            scorer.push(bad)
    scorer.push(np.zeros(0))
    assert len(scorer._buffer) == 3
