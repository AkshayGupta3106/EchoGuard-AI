# EchoGuard-AI

**Agentic AI Voice Scam Detection System**  

---

## 🛡️ What is EchoGuard-AI?

EchoGuard-AI is an intelligent, privacy-first agent that listens to ongoing phone calls and continuously judges whether it is a scam. It goes beyond simple caller-ID checks by combining two independent, state-of-the-art signals:
1. **Acoustic Authenticity:** Is the caller's voice AI-generated or spoofed?
2. **Semantic Intent:** Does the conversation script match known fraud tactics (e.g., OTP phishing, forced urgency)?

It explains its reasoning at every step and recommends immediate action (Warn, Hang Up, Block) before sensitive information can be compromised.

---

## 💡 New Insights & Justifications

To achieve the **Innovation & Novelty**, we rejected traditional Truecaller-style databases. Caller-IDs are easily spoofed, and databases cannot protect users from novel zero-day AI voice clones. 

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
├── download_kroko.py
└── requirements.txt
```
### 🛠️ Getting Started for Developers

Because this project relies on heavy ML models and binary libraries (`.onnx`, `.aar`) that exceed GitHub's file size limits, they are excluded from this repository via `.gitignore`. 

To set up the repository after cloning, **you must run both setup scripts:**
```bash
pip install -r requirements.txt
python download_indicconformer.py   # Hindi STT model + sherpa-onnx AAR library
python download_kroko.py            # English STT model (Kroko-128L)
```
These scripts automatically download the required `sherpa-onnx` AAR library and both STT models, placing them in the correct Android asset directories. Skip either script and the corresponding language's live transcription will silently fail at runtime (the app still builds and runs — it just won't produce a transcript). After running both, open the `app/` folder in Android Studio, sync, and run!

---

## 🧪 Testing

There are two layers to test: the Python pipeline (fast, no phone needed) and the Android app itself (requires a device/emulator). Test the Python side first — if a stage is broken there, it'll be broken on-device too, and it's much faster to debug on a laptop.

### 1. Per-module sanity checks

Each stream can be tested in isolation, no downloads or setup beyond `requirements.txt`:

```bash
cd acoustic && python3 vad_gate.py && python3 spoof_detector.py
```
Confirms Silero VAD and AASIST-L (spoof detection) both load and run against the bundled test audio, using the real pretrained weights already vendored in `acoustic/aasist_model/`.

```bash
cd semantic && python3 scam_classifier.py
```
Confirms the rule-layer scam detection runs. If you see a fallback message about MiniLM not downloading, that's expected offline — the rule layer still works; run again with internet access once to cache `all-MiniLM-L6-v2` and enable the semantic layer.

**The most important acoustic test** — run your own demo voice-clone clip through the spoof detector before trusting it on stage:
```python
from spoof_detector import SpoofDetector
d = SpoofDetector()
print(d.predict_file("your_demo_clip.wav"))  # must be 16kHz mono
```

### 2. End-to-end fusion test (simulated calls)

```bash
cd fusion && python3 demo_pipeline.py
```
Wires real `SpoofDetector` + real `ScamClassifier` through `FusionEngine` → `SupervisorAgent` on two scripted example calls (one benign, one escalating bank-impersonation/OTP scam) and prints the resulting fraud timeline. Expected: the benign call stays low-risk throughout; the scam call climbs from `warn` to `block` as more signals fire. This is the proof that the acoustic and semantic modules agree on a contract (`StreamSignal`) without knowing about each other.

### 3. Test against your own audio

```bash
cd fusion && python3 custom_demo.py --audio path/to/call.wav --text "full transcript of what is said"
```
Feeds a real 16kHz mono `.wav` file through the acoustic path while revealing your supplied transcript gradually (simulating live STT), so you can sanity-check fusion output against a real recording without needing the Android app or a working STT model.

If you'd rather not type out the transcript yourself, transcribe automatically first (uses Whisper, slower on CPU but no live STT dependency):
```bash
cd fusion && python3 full_audio_demo.py --audio path/to/call.wav
```

### 4. Backend (online mode, optional)

```bash
cd backend && uvicorn online_mode_api:app --reload
```
Starts the optional online-mode FastAPI backend on `localhost:8000`. Only fused signals + transcript text are ever sent here — never raw audio — and the app is designed to fall back to fully offline mode automatically if this is unreachable.

### 5. Testing the Android app

1. Make sure both `download_indicconformer.py` and `download_kroko.py` have been run (see Getting Started above) — without them the app builds fine but transcription will silently do nothing.
2. Open `app/` in Android Studio, sync Gradle, and run on a physical device (a real mic/speaker path matters here — an emulator's virtual audio won't give you a meaningful test).
3. Grant the `RECORD_AUDIO`, `READ_PHONE_STATE`, and notification permissions when prompted.
4. **Put the call on speakerphone.** This app deliberately captures ambient audio via the ordinary microphone rather than tapping the telephony subsystem directly (see `app/README.md` for why) — without speakerphone, it will only pick up your own side of the conversation.
5. Start a call (or use "Start Live Protection Demo" if present in the build) and watch the fraud timeline / risk banner update as speech is transcribed and scored.
6. If transcription doesn't appear: check `adb logcat` filtered to your package name for a `KrokoLiveTranscriber` or `IndicConformerLiveTranscriber` error — both throw a specific message naming exactly which model asset is missing, rather than failing silently, so this is the fastest way to tell a missing-model problem from an actual bug.

---

## 🚀 Future Scope (Scale & Real-World Problem Solving)

Our vision for EchoGuard-AI extends far beyond standard cellular calls to solve real-world problems at scale:

1. **VoIP & WhatsApp Integration:** Utilizing Android Accessibility Services to monitor encrypted internet calls (WhatsApp, Telegram) in real-time.
2. **Pan-India Multilingual Expansion:** Extending our Devanagari regex and semantic embedding models to natively support all 22 official Indian languages.
3. **Federated Learning on the Edge:** Allowing edge devices to share newly discovered scam phonetic signatures with a central server *without* ever uploading raw audio, continuously improving the global model while perfectly preserving privacy.
4. **Elderly Protection Mode:** A high-sensitivity UI mode designed for vulnerable demographics that can automatically intercept calls or silently alert trusted family members when high fraud risk is detected.
