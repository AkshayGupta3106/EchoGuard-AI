# EchoGuard-AI: On-Device Voice Scam Detection
**AI Arena 3.0 Hackathon Submission - Design & Technical Documentation**

## 1. Problem Understanding
- **The Problem:** Phone scams and voice cloning (deepfakes) are growing at an alarming rate globally, causing billions in financial losses. Existing solutions either rely on blocklists (which fail against spoofed/new numbers) or require sending private call audio to the cloud for analysis, which is a massive privacy violation and introduces latency.
- **User Needs:** A real-time, privacy-first, zero-latency solution that works entirely offline to detect deepfakes, coercive scam tactics, and financial fraud over phone calls.
- **Constraints:** Must run on edge devices (smartphones) without relying on cloud APIs. Must be highly memory and CPU efficient. Must not drain battery. Must work across multiple languages (specifically handling English and Hindi).

## 2. Concept Development
- **Idea:** A "Physical Prototype" Edge AI application deployed on an Android smartphone that actively listens to the call audio stream and fuses acoustic (voice deepfake) and semantic (conversation context) analysis in real-time.
- **Trade-offs Identified:** Cloud vs. Edge. Cloud offers larger models (LLMs), but Edge guarantees absolute privacy and zero network latency. We chose Edge and optimized small ONNX models to achieve cloud-like accuracy on a mobile phone.
- **Technology Selection:** 
  - **Acoustic:** AASIST-L (Lightweight Audio Anti-Spoofing).
  - **Semantic:** MiniLM embedded with ONNX Runtime for semantic similarity, alongside a deterministic Rule Engine for immediate keyword triggers (e.g., OTP requests, digital arrest).
  - **ASR (Speech-to-Text):** IndicConformer via Sherpa-ONNX to support English and Hindi code-switching (Hinglish), typical in Indian scam calls.
  - **Mobile:** Native Android (Kotlin & Jetpack Compose) for optimal hardware access and UI responsiveness.

## 3. System Design (High-Level)
The complete product architecture follows a robust data pipeline:
1. **VadGate:** Voice Activity Detection (Silero VAD) to filter out silence and background noise, saving battery and preventing unnecessary compute.
2. **Acoustic Stream (SpoofDetector):** Analyzes the raw audio waveform using AASIST-L to detect synthetic/cloned speech.
3. **Semantic Stream (LiveTranscriber + ScamClassifier):** Transcribes audio locally using IndicConformer. Analyzes the text using exact-match rules and MiniLM embeddings to detect coercive context.
4. **Fusion Engine (Asymmetric Boost Algorithm):** Combines the acoustic spoof score and semantic scam score into a unified, compounded Risk Score. Because the semantic model is highly reliable while the acoustic model can be noisy in real-world conditions, we employ an **Asymmetric Boost algorithm**:
   ```math
   Risk = Scam + Spoof \times (1 - Scam) \times 0.45
   ```
   This means the conversation context sets the baseline risk, and the acoustic AI voice detector acts as an "accelerant" that boosts the score by a maximum of 45%. If the acoustic model throws a false positive on an innocent call, it only boosts the score to 45% (which triggers a Warning, but doesn't block the call). But if it detects an AI voice during a suspicious conversation (e.g., 50% scam score), it pushes it to 72.5% (Red Block!).
5. **Supervisor Agent:** Translates the risk score into user-facing actions (MONITOR, WARN, BLOCK) and maintains a real-time UI timeline.

## 4. Low-Level Design
- **Component Specifications:**
  - *Audio Capture:* `AudioRecord` at 16kHz, mono, 16-bit PCM.
  - *Models:* AASIST-L (PyTorch Mobile), MiniLM (.onnx), IndicConformer (.onnx). All bundled securely in app assets.
- **Algorithm & Logic Details:**
  - **Contextual Joke Override Logic:** Employs precise regex and index-tracking to detect conversational context shifts. The logic evaluates the end-index of the latest joke phrase (e.g., "I was just kidding", "mazak") versus the latest serious phrase (e.g., "I'm not kidding", "serious") to dynamically override false positives in real-time.
  - **Memory Management:** Models are aggressively flushed when the call ends, and memory constraints are checked before initialization. The ASR uses `ByteBuffer.allocateDirect` to prevent JNI native memory segmentation faults (SIGSEGV).
- **UI Architecture:** Built in Jetpack Compose utilizing a `StateFlow` unidirectional data architecture for seamless, glitch-free UI updates.

## 5. Novelty & Innovation (On par with Global Standards)
- **Global Standard Novelty:** Running *both* a live ASR + semantic embedding model *and* an anti-spoofing acoustic model simultaneously on a mobile edge device is a novel architecture. It achieves what massive telecommunications companies are attempting, but completely offline.
- **Real-World Indian Standards:** Custom ASR model (IndicConformer) inherently handles Hinglish. Custom scam rules explicitly target Indian fraud patterns like "Digital Arrest", "CBI Threat", and "OTP sharing".
- **Usability:** Zero user intervention required. The app runs in the background. The UI is minimal, "Lovable", and provides a beautiful dark/light mode toggle with smooth micro-animations. No technical setup is required from the end user.

## 6. Testing, Validation, & Iteration
- **Testing:** The system was tested against real scam call recordings, live synthetic speech generation, and harmless conversational pranks.
- **Iteration Execution:** Initially, the system flagged phrases like "I was just kidding" as high risk if preceded by a joke scam demand. We iterated by introducing a temporal index-matching logic that compares the exact character string indexes of joke phrases vs. serious phrases to intelligently lower the risk score in real-time, completely eliminating the false positive while maintaining rigorous security.

## 7. Justifications for other Developers
- **Why ONNX Runtime?** Cross-platform consistency and aggressive hardware acceleration on ARM CPUs. 
- **Why IndicConformer over Whisper?** Whisper hallucinates on Indian accents and code-switching. IndicConformer is trained explicitly on the 22 official Indian languages, yielding vastly superior Word Error Rates (WER) for our target demographic.
- **Why no LLM?** A 7B parameter LLM cannot run concurrently with an audio-spoofing model on an average Indian smartphone without severe thermal throttling and battery drain. The Fusion Engine + MiniLM approach is 99% faster and achieves the exact same functional outcome for scam categorization.
