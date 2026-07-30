# Semantic stream — Person B

## Files

| File | Status | What it is |
|---|---|---|
| `scam_classifier.py` | **Tested, working** | Rule layer + MiniLM zero-shot scam scoring. Run it: `python3 scam_classifier.py` |
| `android/ZipformerLiveTranscriber.kt` | Integration skeleton | sherpa-onnx streaming Zipformer, live path. **Day 1 priority.** |
| `android/WhisperOfflinePass.kt` | Integration skeleton | whisper.cpp batch pass, offline path only, runs after the call ends |

The `.kt` files follow the official sherpa-onnx and whisper.cpp Android integration patterns but haven't been compiled here (no Android build environment in this sandbox) — treat them as a correct starting skeleton, not drop-in tested code. Wire them up early on Day 1 specifically so any build/API mismatches surface immediately.

## Day 1 checklist

1. `python3 scam_classifier.py` — confirm it runs (works even without internet; MiniLM falls back to rules-only if the model can't download, so don't panic if you see that fallback message — just means the semantic layer isn't active yet, rules still work).
2. On your actual dev machine (with internet), let `sentence-transformers` download `all-MiniLM-L6-v2` once — confirm `available: True` shows up in a test run, then the semantic layer kicks in.
3. On an actual Android phone: wire up `ZipformerLiveTranscriber`, feed it real mic audio, log the per-frame latency. This is the important number from the whole plan — confirm it's fast enough before building anything else on top.
4. Same phone: run `WhisperOfflinePass` once against a short test recording, confirm `realtimeFactor > 1.0`.

## Contract with the fusion engine

`ScamClassifier.scam_score(transcript) -> ScamScoreResult`
- `.score` — float 0–1
- `.explain()` — human-readable string of which signals fired, for the fraud timeline UI
- `.rule_hits` / `.semantic_hit` — structured breakdown if the UI wants to render them separately

Feed it the running transcript from Zipformer's live output every 1-2 seconds — same cadence as Person A's `spoof_score()`.

## Tuning notes

- `SCAM_EXEMPLARS` / `RULE_CATEGORIES` in `scam_classifier.py` are hand-written starting points. Add real phrases as you test — no retraining needed, embeddings are computed fresh each run.
- The semantic-score margin mapping (`margin / 0.5` in `SemanticScorer.score`) is an untuned guess. Once you have a few real test calls, check whether scam calls consistently score above ~0.4-0.5 and adjust the constant.
