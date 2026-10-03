# EchoGuard-AI

**Agentic AI Voice Scam Detection System**

> **Detect the voice. Understand the intent. Stop the scam.**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-46E3B7?style=for-the-badge)](https://echoguard-ai-dmaj.onrender.com/)

[Setup & usage](docs/guide.md) · [Evaluation & limitations](docs/evaluation.md)

EchoGuard-AI combines **voice-spoof detection** and **scam-intent analysis** with
an agentic supervisor that tracks conversation risk and explains **MONITOR**,
**WARN**, and **BLOCK** recommendations. Android inference runs on-device;
the web demo sends audio/text to the server for analysis.

**Stack:** Python · Kotlin/Jetpack Compose · PyTorch · ONNX Runtime · sherpa-onnx · FastAPI · Docker

## 🧠 How it works

```text
                      Microphone audio
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
         Speech-to-Text              Voice Authenticity
    Kroko-128L / IndicConformer           AASIST-L
               │                           │
               ▼                           │
       Scam Intent Analysis                │
          MiniLM + Rules                    │
               │                           │
               └─────────────┬─────────────┘
                             ▼
                       Fusion Engine
                             ▼
                      Supervisor Agent
                             ▼
                   Risk + Recommendation
```

**Decision loop:** `Observe → Reason → Explain → Recommend → Act`

- **Acoustic authenticity:** scores possible AI-generated or spoofed voices.
- **Semantic intent:** identifies OTP phishing, impersonation, financial requests,
  secrecy, and urgency—even when the speaker uses a genuine voice.

The diagram shows the Android pipeline. The web demo uses browser speech
recognition; its lightweight deployment uses rules-only text scoring, while
the full local Python environment can also use MiniLM.

## 📱 Android & 🌐 web demo

| Feature | Android | Web demo |
|---|---|---|
| English/Hindi speech analysis | Kroko-128L / IndicConformer | Browser speech recognition |
| Voice authenticity + scam intent | On-device ONNX inference and rules | Server-side inference and rules |
| Input | Microphone | Microphone, pasted text, optional audio upload |
| Session history | Local full transcripts, view/delete/clear | Browser-saved full transcripts, view/delete/clear |
| Appearance | Light, dark, or system theme | Light/dark theme with saved preference |

Android uses INT8 model exports and pre-quantized ASR; the web demo is deployed
via **Docker/FastAPI on Render**.

## 📊 Benchmark highlights

| Metric | Result | Scope |
|---|---|---|
| English / Hindi F1 | **92.3% / 85.7%** | Pooled held-out Python supplied-text fusion |
| Real-time factor (RTF) | **0.092 (~11× real-time throughput)** | AASIST-L acoustic inference |
| P99 inference latency | **~527 ms** | Acoustic inference, median per-run P99 |
| MiniLM export-size reduction | **~75%** | FP32 to self-contained ONNX INT8 |
| Android model footprint | **~358 MiB (~376 MB)** | Bundled ONNX models, excluding staged VAD; not APK size or peak RAM |

These are separate text/acoustic benchmarks, not complete microphone-pipeline
accuracy or end-to-end latency. [Methodology, device results, and limitations](docs/evaluation.md#measured-results).

## 🚀 Get started

| Goal | Guide |
|---|---|
| Run the web demo locally | [Python or Docker setup](docs/guide.md#web-demo) |
| Build and use Android | [Android setup](docs/guide.md#android) |
| Find source, models, and development commands | [Project structure](docs/guide.md#project-structure) |
| Reproduce and interpret benchmarks | [Evaluation commands](docs/evaluation.md#run-evaluations) |

Recordings, generated reports, build outputs, and secrets are excluded from Git.
Recommendations are risk signals, not proof of fraud or automatic blocking.
