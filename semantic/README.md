# Semantic stream — Person B

## Files

| File | Status | What it is |
|---|---|---|
| `scam_classifier.py` | **Tested, working** | Rule layer + MiniLM zero-shot scam scoring (Python side — used by `fusion/` demos and the optional online-mode backend). Run it: `python3 scam_classifier.py` |
| `export_minilm_onnx.py` | **Tested, working** | Exports MiniLM to ONNX, then quantizes to INT8, writing directly into the Android app's assets folder (`app/app/src/main/assets/models/minilm.onnx` + `vocab.txt`) — no manual copy/rename needed |
| `export_exemplar_embeddings.py` | Precomputes the fixed scam/benign exemplar sentence embeddings as a bundled JSON asset — these never change at runtime, so there's no on-device cost for them |
| `minilm_model/` | Local intermediate output only | Holds the fp32 export (`minilm.onnx`, kept for debugging, not bundled) — the actual Android asset is written straight to `app/app/src/main/assets/models/` by `export_minilm_onnx.py`, not copied from here |
| `android/ScamClassifier.kt` | Kotlin port, on-device | Rule layer (direct port of `RULE_CATEGORIES` above) + MiniLM ONNX inference + WordPiece tokenizer. This is what actually runs live in the app — see `app/app/src/main/java/com/echoguard/semantic/ScamClassifier.kt` for the buildable copy |
| `android/KrokoLiveTranscriber.kt` / `android/IndicConformerLiveTranscriber.kt` | Kotlin, on-device | sherpa-onnx streaming ASR — English (Kroko-128L) and Hindi/Hinglish (IndicConformer) live transcription paths |

## Setup

```bash
python export_minilm_onnx.py
```

Needs `torch` + `transformers` (already in `requirements.txt`) and internet
access on first run to pull `sentence-transformers/all-MiniLM-L6-v2` from
Hugging Face. Writes the quantized model straight into the Android assets
folder under its final name — nothing to copy or rename afterward.

The quantized model is a single self-contained `.onnx` file — no external
`.onnx.data` companion, unlike some large fp32 exports. `ScamClassifier.kt`
(both here and the buildable copy under `app/`) expects exactly this.

## Day 1 checklist

1. `python3 scam_classifier.py` — confirm it runs (works even without internet; MiniLM falls back to rules-only if the model can't download, so don't panic if you see that fallback message — just means the semantic layer isn't active yet, rules still work).
2. On your actual dev machine (with internet), run `python3 export_minilm_onnx.py` once — confirm the int8 `minilm.onnx` + `vocab.txt` land in `app/app/src/main/assets/models/`.
3. On an actual Android phone: confirm `ScamClassifier.kt` logs `Initializing ScamClassifier...` (from `PipelineRunner`) without a `SemanticScorer` load-failure warning right after — that confirms MiniLM loaded, not just the rule layer.
4. Feed it real mic audio via `KrokoLiveTranscriber`/`IndicConformerLiveTranscriber`, log the per-frame latency — this is the important number from the whole plan; confirm it's fast enough before building anything else on top.

## Contract with the fusion engine

`ScamClassifier.scam_score(transcript) -> ScamScoreResult` (Python) /
`ScamClassifier.scamScore(transcript) -> ScamScoreResult` (Kotlin)
- `.score` — float 0–1
- `.explain()` — human-readable string of which signals fired, for the fraud timeline UI
- `.ruleHits` / `.semanticResult` — structured breakdown if the UI wants to render them separately

Feed it the running transcript from the live STT output every 1-2 seconds — same cadence as Person A's `spoof_score()`.

## Tuning notes

- `RULE_CATEGORIES` in `scam_classifier.py` (and its Kotlin mirror in `ScamClassifier.kt`) are hand-written starting points, now covering both English and Hindi/Hinglish phrasing. Add real phrases as you test — no retraining needed, embeddings are computed fresh each run. Keep both files in sync if you edit either.
- The `SemanticScorer` RAM guard (`availMb < 100` in `ScamClassifier.kt`) falls back to rules-only on low-memory devices. Quantizing MiniLM down to ~22 MB (from ~90 MB fp32) makes this threshold much easier to clear than before, but it's still worth checking on a genuinely low-end test device.
- The semantic-score margin mapping (`margin / 0.22f` in `SemanticScorer.score`) is a tuned-but-not-exhaustively-verified constant. Once you have a few real test calls, check whether scam calls consistently score above the exemplar margin and adjust if needed.