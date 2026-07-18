# EchoGuard-AI

**Agentic AI Voice Scam Detection System**
*AI Arena 3.0 Hackathon Project — AI Voice • Agentic AI • Multimodal Sensing*

Built by [Akshay Gupta](https://github.com/AkshayGupta3106) & Chirag Bhutra

---

## What is EchoGuard-AI

EchoGuard-AI listens to an ongoing phone call and continuously judges whether it's a scam, combining two independent signals — whether the caller's voice is AI-generated, and whether the conversation itself sounds like a fraud script. It explains its reasoning at every step and recommends an action (warn, hang up, block) before sensitive information like an OTP gets shared.

It runs in two modes:

- **Offline mode** — everything runs on-device. Audio never leaves the phone.
- **Online mode** — same on-device signal extraction, with an optional cloud step for richer explanations and threat-intel lookups. Only fused signals and transcript text are ever sent out, never raw audio. If connectivity drops mid-call, it falls back to offline mode automatically.

## Why

Traditional spam blockers work off caller-ID and phone number databases. They can't tell you if the voice on the line is cloned, or if the conversation itself is a manipulation script. EchoGuard-AI looks at both, fuses the evidence, and reasons about it like an agent instead of just returning a number.

## How It Works

| Stage | What it does | Model / Tool |
|---|---|---|
| Speech-to-text | Streams live speech to text on-device, low latency | sherpa-onnx (streaming Zipformer) |
| Voice authenticity | Scores whether the caller's voice is AI-generated | CNN over Mel-spectrograms (librosa + PyTorch) |
| Scam intent | Scores conversational scam intent, with memory across turns (OTP requests, urgency, impersonation claims) | MiniLM (all-MiniLM-L6-v2) + Logistic Regression |
| Fusion | Calibrates and combines both scores into one fraud-risk number | Calibrated fusion model |
| Supervisor Agent | Reasons over the fused signal and decides what to do | Offline rule engine (on-device) / LLM reasoning agent (online) |

The Supervisor Agent follows a five-step loop on every update: **Observe → Reason → Explain → Recommend → Act.**

## Tech Stack

- **Mobile:** Kotlin, Jetpack Compose
- **Speech Recognition:** sherpa-onnx (streaming Zipformer), ONNX Runtime
- **Voice Analysis:** librosa, PyTorch, CNN
- **NLP:** sentence-transformers (all-MiniLM-L6-v2), scikit-learn
- **Backend (online mode only):** FastAPI
- **ML:** PyTorch, scikit-learn

## Project Structure

```
EchoGuard-AI/
├── android-app/
├── backend/
│   ├── asr/          # sherpa-onnx streaming integration
│   ├── audio_cnn/
│   ├── scam_intent/
│   ├── fusion/
│   ├── agent/
│   └── api/
├── datasets/
├── models/
├── documentation/
└── README.md
```

## Datasets

- **Voice deepfake detection:** ASVspoof 2019, ASVspoof 2021 (codec-augmented for real call audio)
- **Scam detection:** SMS Spam Collection, scam call transcripts, banking scam dialogues, synthetic scam conversations

## Demo Flow

1. A call comes in.
2. EchoGuard-AI captures the audio locally.
3. sherpa-onnx transcribes it in real time.
4. The CNN scores voice authenticity.
5. MiniLM + the conversation-state tracker score scam intent.
6. The fusion engine computes an overall fraud-risk score.
7. The Supervisor Agent explains its reasoning.
8. The user gets a real-time warning and a recommended action.

## Future Scope

- WhatsApp call monitoring
- Video call deepfake detection
- Multilingual scam detection
- Enterprise fraud monitoring
- Elderly protection mode
- Federated learning across devices
- Integration with caller-ID / threat-intel services

## Team

- **Akshay Gupta** — [@AkshayGupta3106](https://github.com/AkshayGupta3106) — ML pipeline: ASR, voice authenticity, scam intent, fusion, Supervisor Agent
- **Chirag Bhutra** — Android app, on-device model integration, backend, UI, demo

## License

Not yet decided — add a `LICENSE` file before making this public if you want reuse terms to be explicit.
