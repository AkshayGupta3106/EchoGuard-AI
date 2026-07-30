# EchoGuard-AI

**Agentic AI Voice Scam Detection System**  
*AI Arena 3.0 Hackathon Project — AI Voice • Agentic AI • Multimodal Sensing*

Built by [Chirag Bhutra](https://github.com/Chirag514) & [Akshay Gupta](https://github.com/AkshayGupta3106) 

---

## 🛡️ What is EchoGuard-AI?

EchoGuard-AI is an intelligent, privacy-first agent that listens to ongoing phone calls and continuously judges whether it is a scam. It goes beyond simple caller-ID checks by combining two independent, state-of-the-art signals:
1. **Acoustic Authenticity:** Is the caller's voice AI-generated or spoofed?
2. **Semantic Intent:** Does the conversation script match known fraud tactics (e.g., OTP phishing, forced urgency)?

It explains its reasoning at every step and recommends immediate action (Warn, Hang Up, Block) before sensitive information can be compromised.

---

## 💡 New Insights & Justifications

To achieve the 25-mark standard for **Innovation & Novelty**, we rejected traditional Truecaller-style databases. Caller-IDs are easily spoofed, and databases cannot protect users from novel zero-day AI voice clones. 

**Our Core Insights & Architectural Justifications:**
- **Why dual-modal fusion?** A human scammer has a real voice but a malicious script. An AI voice clone has a synthetic voice but might be reading a benign script. Relying on just acoustic or just semantic models creates massive blind spots. Fusing both creates a mathematically robust fraud-risk score.
- **Why ONNX Runtime on Edge?** Sending live phone call audio to a cloud server violates user privacy and introduces latency. We justified compressing our acoustic (`AASIST-L`) and transcription (`Kroko-128L`) models into 8-bit quantized ONNX formats so the entire pipeline can run **100% offline** on a mid-range Android device.
- **Agentic Decision Making:** Instead of just flashing a red warning, our **Supervisor Agent** holds conversational memory. It knows if the caller previously asked for money and is *now* asking for an OTP, dynamically escalating the threat level.

---

## 🧠 Process & Architecture

| Stage | Action | Underlying Technology |
|---|---|---|
| **Speech-to-Text** | Streams live speech to text on-device (English & Hindi) | `Kroko-128L` (English) & `IndicConformer` (Hindi) via ONNX Runtime |
| **Voice Authenticity** | Scores whether the caller's voice is AI-generated | `AASIST-L` (ONNX) Acoustic Spoof Detection |
| **Scam Intent** | Scores conversational intent, tracking context across turns | `all-MiniLM-L6-v2` + Semantic Embedding Search |
| **Context Overrides** | Catches edge cases (e.g., detecting "मज़ाक कर रहा" / "just kidding") | Custom Cross-Lingual (Hindi/English) Regex Engine |
| **Fusion Engine** | Calibrates both Acoustic and Semantic scores into one fraud-risk metric | Custom Python Fusion Engine & `FusionEngine.kt` |
| **Supervisor Agent** | Reasons over the fused signal and decides what to do | Local Pipeline State Manager & Reasoner |

The **Supervisor Agent** acts as the core "brain", following a continuous loop on every audio chunk:  
**Observe → Reason → Explain → Recommend → Act.**

---

## 💻 Tech Stack & Developer Details

- **Mobile App:** Native Android, Kotlin, Jetpack Compose, Kotlin Coroutines
- **Speech Recognition:** ONNX Runtime, `sherpa-onnx` local JNI bindings
- **Acoustic AI:** `AASIST-L` (Spoof Detection), `Silero VAD` (Voice Activity Detection)
- **Semantic AI:** `all-MiniLM-L6-v2`
- **Backend (Online mode):** Python, FastAPI

### 📂 Project Structure for Developers

```text
EchoGuard-AI/
├── app/               # Full Android Studio Project (Jetpack Compose UI)
├── acoustic/          # Python acoustic models, VAD gating & AASIST-L scripts
├── backend/           # FastAPI backend for Online Mode API
├── fusion/            # Fusion Engine logic (merging Acoustic + Semantic scores)
├── semantic/          # MiniLM embeddings, hotword generation, and Scam Classifier
├── download_indicconformer.py
└── requirements.txt
```
*(Note: To comply with GitHub's LFS limits, heavy `.onnx` models are excluded from this repository and are packaged directly within our final `.zip` build artifact).*

---

## 🚀 Future Scope (Scale & Real-World Problem Solving)

Our vision for EchoGuard-AI extends far beyond standard cellular calls to solve real-world problems at scale:

1. **VoIP & WhatsApp Integration:** Utilizing Android Accessibility Services to monitor encrypted internet calls (WhatsApp, Telegram) in real-time.
2. **Pan-India Multilingual Expansion:** Extending our Devanagari regex and semantic embedding models to natively support all 22 official Indian languages.
3. **Federated Learning on the Edge:** Allowing edge devices to share newly discovered scam phonetic signatures with a central server *without* ever uploading raw audio, continuously improving the global model while perfectly preserving privacy.
4. **Elderly Protection Mode:** A high-sensitivity UI mode designed for vulnerable demographics that can automatically intercept calls or silently alert trusted family members when high fraud risk is detected.

---

## 📄 License

This project was built for the **AI Arena 3.0 Hackathon**. All rights reserved by the authors.
