# EchoGuard-AI

**Agentic AI Voice Scam Detection System**

> **Detect the voice. Understand the intent. Stop the scam.**

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Render-46E3B7)](https://echoguard-ai-dmaj.onrender.com/)

EchoGuard-AI is a **privacy-first voice scam detection system** that combines **acoustic spoof detection** and **semantic scam-intent analysis**, using an agentic supervisor to continuously assess conversations and escalate fraud risk.

## 🧠 How It Works

```text
                    Phone Conversation
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
        Speech-to-Text             Voice Authenticity
     Kroko-128L / IndicConformer        AASIST-L
             │                           │
             ▼                           ▼
      Semantic Intent              Acoustic Score
       MiniLM + Rules                    │
             │                           │
             └─────────────┬─────────────┘
                           ▼
                    Fusion Engine
                           │
                           ▼
                  Supervisor Agent
                           │
                           ▼
                     Risk / Action
```

**Decision loop:** `Observe → Reason → Explain → Recommend → Act`

### Dual-Modal Detection

- 🎙️ **Acoustic Authenticity** — detects AI-generated / spoofed voices with `AASIST-L`
- 🧠 **Semantic Intent** — detects scam tactics such as OTP phishing, impersonation, financial requests, and forced urgency

Combining both signals helps detect both **human scammers with genuine voices** and **AI-generated voice scams**.

## 🔬 Detection Pipeline

| Component | Technology |
|---|---|
| Speech-to-Text | `Kroko-128L` · `IndicConformer` · ONNX Runtime |
| Voice Activity Detection | `Silero VAD` |
| Voice Spoof Detection | `AASIST-L` |
| Scam Intent | `all-MiniLM-L6-v2` + semantic search |
| Context Detection | Hindi/English rule engine |
| Signal Fusion | Custom `FusionEngine` |
| Decision Layer | `SupervisorAgent` |

## 📱 Android

The main application is built with **Kotlin, Jetpack Compose, Coroutines, ONNX Runtime, and sherpa-onnx**.

The Android pipeline is designed for **on-device inference**, keeping the core voice-analysis pipeline local and minimizing latency and cloud dependency.

## 🌐 Web Demo

The `webdemo/` directory provides a browser-based demonstration with:

- Live microphone analysis
- Transcript analysis
- WAV audio analysis
- English and Hindi support
- Semantic scam scoring
- Acoustic spoof scoring
- Continuous fraud-risk updates
- Supervisor-agent reasoning

The deployed web demo runs inference server-side; the Android application targets on-device inference.

## 📂 Project Structure

```text
EchoGuard-AI/
├── app/                         # Android application
├── acoustic/                    # VAD + AASIST-L spoof detection
├── semantic/                    # Scam intent detection
├── fusion/                      # Fusion engine + supervisor
├── backend/                     # FastAPI online mode
├── webdemo/                     # Browser-based demonstration
├── download_indicconformer.py   # Hindi STT setup
├── download_kroko.py            # English STT setup
└── requirements.txt
```

## 🚀 Setup

Heavy ML models and Android binary dependencies are excluded from Git.

```bash
git clone https://github.com/Chirag514/EchoGuard-AI.git
cd EchoGuard-AI

pip install -r requirements.txt
python download_indicconformer.py
python download_kroko.py
```

Open `app/` in Android Studio, sync Gradle, and run on an Android device.

### Python Pipeline

```bash
cd acoustic && python vad_gate.py && python spoof_detector.py
cd ../semantic && python scam_classifier.py
cd ../fusion && python demo_pipeline.py
```

## 🔮 Future Scope

- VoIP and WhatsApp scam detection
- Pan-India multilingual support
- Federated learning for scam intelligence
- Elderly protection mode
- Adaptive detection of emerging scam patterns

---

**EchoGuard-AI — Detect the voice. Understand the intent. Stop the scam.**
