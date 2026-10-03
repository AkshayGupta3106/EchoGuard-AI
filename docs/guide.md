# Setup and usage

[Project overview](../README.md) · [Benchmarks and evaluation](evaluation.md)

Run commands from the repository root unless shown otherwise. Use Python 3.11;
Android builds require Java 17 and the Android SDK.

## Python environment

```powershell
git clone https://github.com/Chirag514/EchoGuard-AI.git
cd EchoGuard-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

| Task | Install |
|---|---|
| Lightweight web | `python -m pip install -r webdemo/requirements.txt` |
| Full local inference/model tools | `python -m pip install -r tools/requirements/models.txt` |
| Evaluation/test helpers | `python -m pip install -r tools/requirements/evaluation.txt` |

Install the checkout with `python -m pip install --no-deps -e .` (included in
model/evaluation requirements). Evaluation helpers also need the runtime
dependencies for the backends being tested.

## Web demo

```powershell
cd webdemo
uvicorn app:app --host 127.0.0.1 --port 7860
```

Or build Docker from the repository root:

```powershell
docker build -f webdemo/Dockerfile -t echoguard-web .
docker run --rm -p 127.0.0.1:7860:7860 echoguard-web
```

The image requires `assets/acoustic/aasist_l.onnx`. It uses ONNX acoustic
inference and rules-only text scoring, without PyTorch or sentence-transformers.
The full local environment can also use MiniLM. Deployment uses `PORT` (default
7860) and must run a single worker for process-local sessions.
Open **http://localhost:7860** after starting the server.

- **Microphone:** select English or Hindi, start, speak, then stop. Browser
  recognition requires support, microphone permission, and localhost or HTTPS.
- **Paste transcript:** analyze text, optionally with an audio recording.
- **History:** view complete transcripts, delete entries, or clear history.
  The latest 25 sessions are saved in this browser, up to 50,000 characters each.
  Longer transcripts are not saved; storage failures show a warning.
- **Theme:** switch between light and dark; the preference persists in this browser.

Audio/text are sent to the server. History is browser-local, not synchronized
with Android or other devices. Unconfirmed queued/interim text is marked in
history; Stop does not submit remaining queued turns. Remove history on shared devices.

## Android

Open `app/` in Android Studio, or build:

```powershell
.\app\gradlew.bat -p app --no-daemon assembleDebug assembleDebugAndroidTest
```

APKs are under `app/app/build/outputs/apk/{debug,androidTest/debug}/`.
Gradle downloads sherpa-onnx 1.13.4 if needed and uses ONNX Runtime 1.27.0.
The required `assets/acoustic/silero_vad.onnx` is staged into build assets.

Select English or Hindi, then **Start microphone**. Only microphone permission
is requested. Stop or leaving the app ends capture and saves available session
output. **History → View full transcript** opens selectable, scrollable text;
entries containing only a preview offer **View saved snippet**.
Inference and history are local; Android backup is disabled. Initialization
errors and unavailable-model warnings appear in the UI.

## Models and detection

| Asset | Location |
|---|---|
| AASIST-L model, checkpoint, config, and licenses | `assets/acoustic/` (web ONNX: `aasist_l.onnx`) |
| Silero VAD | `assets/acoustic/silero_vad.onnx` |
| Android AASIST/MiniLM/vocabulary/embeddings | `app/app/src/main/assets/models/` |
| English Kroko ASR | `app/app/src/main/assets/kroko-128l/` |
| Hindi IndicConformer ASR | `app/app/src/main/assets/indicconformer-hi/` |
| Python semantic assets | `assets/semantic/` |

Large Android model binaries and downloads stay local and excluded from Git;
the shared web AASIST-L export, checkpoint, and Silero VAD are published.
Download missing ASR assets with `tools/models/download_kroko.py` and
`tools/models/download_indicconformer.py`. Other model tools are
`export_aasist.py`, `export_minilm.py`, and `export_exemplars.py` in that folder;
run exports only when deliberately replacing models. The exemplar exporter
does not copy embeddings into Android automatically. Check model/tokenizer
compatibility before replacing packaged assets.

AASIST-L accepts mono 16 kHz audio in 64,600-sample windows; short input is repeated
and longer input truncated. Fusion uses `scam + spoof × (1 − scam) × 0.45`, with
WARN at 0.35 and BLOCK at 0.65. Android retains peak risk until reset;
intentional joke/prank handling can reset it. Platform rules/tokenizers differ.
Android scores every 45 audio chunks and forwards audio downstream regardless
of the VAD flag.

## HTTP API

Routes: `/api/health`, `/api/analyze`, `/api/live/start`, `/api/live/turn`,
`/api/live/audio-chunk`, and `/api/live/end`.

Live sessions expire after 20 idle minutes. Updates are serialized per session;
`turn_id` supports idempotent text retries and `revision` orders results. Browser
requests have a 30-second timeout and an ordered, bounded transcript retry queue;
audio chunks are best-effort. There is no server-side ASR.

Limits: 16 sessions, 50,000 transcript characters, 1,000 turns, 5,000 updates,
10 MiB requests, and 8 MiB audio uploads. Decoded audio must be nonempty/finite,
mono/stereo, at most 60 seconds and 96 kHz. Authentication and distributed quotas
are not implemented.

## Project structure

| Folder | Contents |
|---|---|
| `app/` | Android app and device tests |
| `src/echoguard/` | Python detection, fusion, HTTP API, and tooling implementations |
| `webdemo/` | Browser UI, server entry point, and Docker deployment |
| `assets/` | Shared models, vocabulary, embeddings, and licenses |
| `tools/` | Development, model, demonstration, and evaluation commands |
| `data/` | Conversation datasets and ignored local audio |
| `artifacts/` | Ignored local evaluation reports |

## Development and tools

```powershell
python tools/check_source.py --layout
python -m pytest -q
node --test tools/tests/frontend_regressions.cjs
.\app\gradlew.bat -p app lintDebug
```

Optional installed analyzers: `python tools/check_source.py --analysis`.
Opt-in container tests: set `ECHOGUARD_DOCKER_IMAGE` to a built image, then run
`python -m pytest tools/tests/test_docker_smoke.py -q`.

Demonstrations live in `tools/examples/`; inspect models with
`python tools/inspect_onnx.py model.onnx`. Prepare consented recordings with
`python tools/evaluation/prepare_bonafide.py --src "D:/recordings"`; use a fresh
destination because preparation overwrites outputs. Anonymous filenames do not
anonymize voices or speech. Keep recordings and signing credentials local.
