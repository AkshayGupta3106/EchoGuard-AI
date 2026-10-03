# Evaluation

[Project overview](../README.md) · [Setup and usage](guide.md)

## Run evaluations

Prepare the runtime/test environment using the [setup guide](guide.md#python-environment).
Datasets are in `data/datasets/`; local audio is in
`data/audio/{synthetic,bonafide,bonafide_phone}/`.

| Command | Measures |
|---|---|
| `python tools/evaluation/run_eval.py --help` | Python text, acoustic, and VAD behavior |
| `python tools/evaluation/run_eval_e2e.py --help` | Web HTTP supplied-text sessions and optional audio false alarms |
| `python tools/evaluation/device_eval.py --help` | Android native readiness and supplied-text scoring |
| `python tools/evaluation/audio_replay.py --serial DEVICE_SERIAL` | Packaged Android ASR on existing WAV files |
| `python tools/evaluation/audio_diagnostics.py` | Acoustic scores and context-free/context-aware VAD coverage |

Examples, from the repository root:

```powershell
python tools/evaluation/run_eval.py --repo . --config both --no-acoustic
python tools/evaluation/run_eval_e2e.py --url http://127.0.0.1:7860 --dataset data/datasets/test_heldout.json data/datasets/test_heldout2.json
adb devices -l
python tools/evaluation/device_eval.py --serial DEVICE_SERIAL --dataset data/datasets/test_heldout.json data/datasets/test_heldout2.json
```

Python MiniLM evaluation needs the full model environment. Acoustic evaluation
accepts `--audio-dir`, repeatable `--bonafide-dir`, and `--vad-model`; `--sweep`
selects a threshold on DEV only. Device evaluation builds/installs debug APKs
with `adb install -r` and preserves app data. ASR replay requires both APKs
installed; `--limit 1` runs a small pilot.

Supplied-text tests do not measure microphone/ASR accuracy or the complete
asynchronous audio-plus-text pipeline. Native readiness uses generated silence,
not speech accuracy. DEV is in-sample; use held-out results for generalization.
Do not send private recordings to public endpoints without consent.

## Reports

Each run writes a unique folder under
`artifacts/evaluation/{python,webdemo,android-device}/`. Python runs store
`eval_results.json`, `RESULTS.md`, per-call reports, and `audio_diagnostics.json`
where applicable. Web/device scoring runs store per-dataset JSON; the ASR replay
run stores `summary.json` with `asr-en.json`/`asr-hi.json`. Existing reports are
not overwritten. Failed runs return a nonzero status and retain partial
diagnostics; check completion, errors, and actual coverage before interpreting
metrics. Dataset basenames must be unique within a run. Undefined metrics use
JSON null.

Reports, transcripts, and recordings remain local and are excluded from Git.

## Measured results

### Python benchmark highlights

Pooled held-out text-only fusion F1 is **92.3% English / 85.7% Hindi**, with 36
conversations per language. Single-threaded AASIST-L acoustic inference uses
4.04-second windows: median RTF **0.092** (~11× real-time throughput), with
**~527 ms P99** as the median of five per-run P99 values. These are separate
text/acoustic measurements, not full microphone-pipeline performance.
The saved Python report is `RESULTS.md` with matching `eval_results.json` and
per-call reports in the local Python evaluation directory.

The MiniLM INT8 export is ~75% smaller than its FP32 model plus external weights.
Bundled Android ONNX models total ~358 MiB (~376 MB), excluding the staged VAD;
this is model storage, not APK size or peak memory.

### Web and Android supplied-text scoring

| Dataset | Web HTTP F1 | Android F1 |
|---|---:|---:|
| DEV, 100 conversations | 1.0000 | 0.9000 |
| Held-out 1, 40 conversations | 0.8571 | 0.6111 |
| Held-out 2, 40 conversations | 0.9500 | 0.7222 |

Platform rules, tokenization, and fusion state differ. Web requests completed
without errors; the configuration canary is heuristic, not proof of model availability.

### Android ASR replay

All 127 existing clips were replayed: all 9 public and 100 synthetic clips
produced text, as did 17/18 private clips using the English model.
The nine referenced public English clips yielded **3.97% WER / 3.08% CER** under
Unicode/punctuation normalization. This small clean-English set does not establish
Hindi or noisy-recording accuracy. Private clips have no verified references.

Final recognized-text semantic F1 on synthetic DEV clips was **1.0000 English**
and **0.6486 Hindi**, with 13/25 Hindi scam clips missed. These results exclude
acoustic fusion and real-time scheduling; synthetic script agreement is not
independently verified WER.

### Acoustic and VAD diagnostics

All 127 clips reached acoustic-only WARN, including all nine public human-speech
fixtures. Fusion permits acoustic-only risk up to 0.45, above WARN at 0.35;
this is a false-alarm finding on a small fixture set, not a population estimate.
Synthetic voice scores do not establish scam intent.

Context-free runtime VAD detected speech in 0% of complete chunks on public/private
clips. Context-aware diagnostic mean per-clip coverage was **90.70% public** and
**59.23% private**. Coverage is not annotated frame accuracy. Android forwards audio
downstream regardless of VAD; diagnostic context handling is separate from runtime.

## Verification and limitations

| Target | Verified scope |
|---|---|
| Python/API | Syntax/imports, input limits, retries, report safety, and repository hygiene |
| Browser | Safe rendering, history persistence/scrolling, themes, and session cleanup; not recognition accuracy |
| Android | Debug/test builds, lint, native readiness/lifecycle, and isolated history/theme checks on Samsung SM-M356B (API 36) |
| Docker | Health, analysis, sessions, and ONNX inference smoke checks |

Important limits: Hindi recognition/semantic coverage is insufficient as sole fraud
protection. Native inference can delay Android shutdown; demonstration WAV validation
varies, and model downloads are not revision/hash pinned. Wall-clock timing tests
can fail under competing build load. Model fallbacks and low risk alone do not prove
readiness. No complete asynchronous-pipeline accuracy, OEM-wide reliability, or
production load benchmark is established. See the guide for
[web session limits](guide.md#http-api) and [recording preparation](guide.md#development-and-tools).

## Public audio sources and licenses

Nine clean English human-speech fixtures from two speakers are used locally under
`data/audio/bonafide/`. They are not a representative noisy-speech benchmark.

**LJSpeech (8 clips):** Keith Ito's public-domain LJSpeech-1.1 audiobook narration
from LibriVox, obtained from [coqui-ai/TTS fixtures](https://github.com/coqui-ai/TTS/tree/dev/tests/data/ljspeech).
Audio was resampled from 22,050 Hz to 16 kHz mono.

| file | transcript |
|---|---|
| ljspeech_LJ001-0001.wav | Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition |
| ljspeech_LJ001-0002.wav | in being comparatively modern. |
| ljspeech_LJ001-0003.wav | For although the Chinese took impressions from wood blocks engraved in relief for centuries before the woodcutters of the Netherlands, by a similar process |
| ljspeech_LJ001-0004.wav | produced the block books, which were the immediate predecessors of the true printed book, |
| ljspeech_LJ001-0005.wav | the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing. |
| ljspeech_LJ001-0006.wav | And it is worth mention in passing that, as an example of fine typography, |
| ljspeech_LJ001-0007.wav | the earliest book printed with movable types, the Gutenberg, or "forty-two line Bible" of about 1455, |
| ljspeech_LJ001-0008.wav | has never been surpassed. |

**JFK (1 clip):** public-domain US government recording from the inaugural
address, obtained from [whisper.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/samples/jfk.wav).

| file | transcript |
|---|---|
| jfk_whisper_cpp_sample.wav | And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country. |
