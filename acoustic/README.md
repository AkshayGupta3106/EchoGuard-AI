# Acoustic stream — Person A

## Files

| File | Status | What it is |
|---|---|---|
| `vad_gate.py` | **Tested, working** — real Silero VAD ONNX model | Speech/silence gate |
| `spoof_detector.py` | **Tested, working** — real pretrained AASIST-L checkpoint | `predict_file()` / `RollingSpoofScorer` |
| `export_onnx.py` | **Tested, working** | Exports AASIST-L to ONNX, then quantizes to INT8, writing directly into the Android app's assets folder (`app/app/src/main/assets/models/aasist_l.onnx`) — no manual copy/rename needed |
| `aasist_model/` | Real files, not placeholders | Vendored `AASIST.py`, `AASIST-L.pth` (original checkpoint), `AASIST-L.conf`, `aasist_l.onnx` (fp32 intermediate export, kept locally for debugging only — not bundled), `silero_vad.onnx`, both licenses |
| `android/VadGate.kt` | Integration skeleton | ONNX Runtime Mobile port of `vad_gate.py` |
| `android/SpoofDetector.kt` | Integration skeleton | ONNX Runtime Mobile port of `spoof_detector.py` |

Unlike Person B's `.kt` skeletons, everything here was tested against the **real pretrained weights** — not stubs. Specifically verified in this environment:
- AASIST-L loads and runs — parameter count matches the paper exactly (85,306).
- The ONNX export's output matches the original PyTorch model's output on identical input (checked side by side).
- Silero VAD correctly scores silence/noise as non-speech.

What's *not* verified here: real speech (vs. synthetic noise/silence), real Android/ONNX Runtime Mobile execution, and real on-device latency — that's Day 1's job, on an actual phone.

## Day 1 checklist

1. `python3 vad_gate.py` and `python3 spoof_detector.py` — confirm both run (they should, out of the box, no downloads needed — the real weights are already bundled in `aasist_model/`).
2. **The most important test in the whole plan:** get your demo voice-clone clip, then run:
   ```python
   from spoof_detector import SpoofDetector
   d = SpoofDetector()
   print(d.predict_file("your_demo_clip.wav"))  # must be 16kHz mono
   ```
   If `spoof_score` isn't clearly high on your actual demo clip, you need to know that now, not on stage.
3. On Android: `python acoustic/export_onnx.py` (produces the quantized `aasist_l.onnx` asset automatically), wire up `VadGate.kt` and `SpoofDetector.kt`, and time `SpoofDetector.score()` on a real phone. On this dev sandbox's CPU, PyTorch inference took ~750-1000ms per 4-second window — that's your rough ballpark to beat or match on-device; profile for real rather than assuming.

## Contract with the fusion engine

`SpoofDetector.score() -> {spoofScore, bonafideScore, inferenceMs}` (Kotlin) / `RollingSpoofScorer.score() -> dict` (Python) — feed `spoof_score` into the fusion engine every 1-2 seconds, same cadence as Person B's `scam_score()`.

`VadGate.isSpeech(chunk) -> {speechProb, isSpeech}` — gate both this stream and Person B's Zipformer input on `isSpeech`; only push audio through when true.

## One correction to the original plan

AASIST-L takes **raw 16kHz waveform directly**, not a mel-spectrogram — it has its own SincConv front-end built in. If you'd started on a librosa spectrogram-extraction step for this model, drop it; it's not needed and won't match what the model expects.

## Note on the ONNX export

`export_onnx.py` produces two files: a full-precision fp32 export
(`aasist_model/aasist_l.onnx`, kept locally as an intermediate/debugging
artifact only) and an INT8-quantized version written directly to
`app/app/src/main/assets/models/aasist_l.onnx` — the one the Android app
actually loads. The quantized model is a single self-contained file with
**no external `.onnx.data` companion** (unlike some large fp32 exports)
- `SpoofDetector.kt` was updated to only expect the one file.

The fp32→int8 quantization step needs `onnxruntime.quantization`'s
`quant_pre_process` shape-inference preprocessing before quantizing, and
the export itself must use the legacy TorchScript-based exporter
(`torch.onnx.export(..., dynamo=False)`) rather than PyTorch's newer
default dynamo exporter — the newer exporter writes conflicting shape
metadata for this model's two differently-shaped outputs (`embedding`
[1,160] vs `logits` [1,2]) that crashes the quantizer. Both are handled
already inside `export_onnx.py`; mentioned here in case you ever need to
debug a similar issue when re-exporting.

## Licensing

`aasist_model/LICENSE_AASIST` (MIT) and `LICENSE_SILERO_VAD` (MIT) are both included — keep them if you redistribute the app, standard practice for MIT-licensed model weights.