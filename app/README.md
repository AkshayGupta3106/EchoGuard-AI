# app/ — the Android project shell

This is the piece that didn't exist before: an actual buildable app
structure tying `/acoustic/`, `/semantic/`, and `/fusion/` together.
**Nothing in this folder has been compiled** — there's no Android SDK/build
toolchain in the environment this was written in. Written carefully by hand
against documented Android APIs, but treat it as a strong starting point to
build and fix, not finished code.

## The call-audio-access decision (read this first)

This was flagged as an open question, and it's a real one — Android does
not let an arbitrary app tap into live call audio by default. Here's what
was actually checked, and what this scaffold assumes:

| Approach | Verdict |
|---|---|
| `CallScreeningService` | Doesn't help for this. It screens whether to *allow/block* a call before/as it connects, based on caller ID — it does not expose the live audio stream during the call. Useful as a *complementary* caller-ID pre-check, not for transcription/analysis. |
| `MediaRecorder.AudioSource.VOICE_CALL` / `VOICE_COMMUNICATION` | Historically how call-recording apps worked, but Google has progressively restricted this since Android 9/10. Support is inconsistent and OEM-dependent — some Android skins (common on several popular Indian OEM devices) still allow it, stock/Pixel-class Android generally blocks it. **Not reliable enough to build the demo path on.** |
| Being the default Phone/Dialer app (`InCallService`) | Gets you call *state*, not guaranteed audio access either — audio access rules above still apply on top of this, and taking over the user's default dialer is a heavy UX ask for a hackathon demo. |
| **Speakerphone + regular microphone recording** | **What this scaffold uses.** Put the call on speaker, then record with the ordinary microphone (`RECORD_AUDIO` permission only) — the same permission any voice-memo app uses. This picks up both sides of the conversation as ambient sound. It doesn't touch the telephony audio subsystem at all, so none of the restrictions above apply, and it works identically on every Android device regardless of OEM. |

**This is the deliberate, demo-safe choice.** It's also literally what a
lot of real call-recording and scam-protection apps fall back to for the
same reason. The trade-off: the user has to actually switch the call to
speaker for this to work, which needs a clear one-line prompt in the UI —
not a hidden requirement.

`CallMonitorService.kt` below uses `TelephonyCallback`/`PhoneStateListener`
only to detect *when a call is active* (fully reliable, standard,
`READ_PHONE_STATE` permission) — not to access its audio. Once a call is
detected, it starts a normal microphone recording loop, on the assumption
the user has switched to speaker.

## Source of truth vs. the buildable copy

`/acoustic/android/`, `/semantic/android/`, `/fusion/android/` remain the
source of truth for those Kotlin files — kept alongside their owning
stream's Python code so whoever owns that stream can find everything in
one place. Gradle, however, needs every `.kt` file under one
`app/src/main/java` tree to compile them together, so this scaffold
contains **copies** of those files under
`app/app/src/main/java/com.echoguard/{acoustic,semantic,fusion}/`.
If you edit a stream's Kotlin file, edit it in the owning folder and
re-copy — don't let the two drift silently out of sync.

## What's real vs. stubbed in this scaffold

| File | Status |
|---|---|
| `AndroidManifest.xml` | Permissions and service declarations are complete and standard |
| `CallMonitorService.kt` | Call-state detection logic is standard/correct; the audio capture loop is written but **never run against a real call** |
| `PipelineViewModel.kt` | Wires VAD → AASIST-L → IndicConformer → ScamClassifier → Fusion → SupervisorAgent in the order already tested in `demo_pipeline.py` — same logic, not yet run on-device |
| `IndicConformerLiveTranscriber.kt` | Live ASR engine — AI4Bharat IndicConformer (Hindi, NeMo CTC via sherpa-onnx). Handles Hinglish code-switching natively. Replaces the English-only Zipformer Transducer. |
| `MainActivity.kt` / Compose UI | Renders the fraud timeline + risk banner; not tested on a real screen |
| `build.gradle.kts` files | Dependency list is correct for the libraries actually used elsewhere in this project (ONNX Runtime Mobile, sherpa-onnx); versions may need bumping |

## What's still genuinely missing after this scaffold

- **IndicConformer model assets** — run `python download_indicconformer.py` from the repo
  root to fetch `indicconformer-hi/model.int8.onnx` + `tokens.txt` (~150 MB) from
  Hugging Face (`parismitaglobalsolutions/indicconformer-sherpa-onnx`). The app will
  show a clear error message in the transcript card if these files are missing.
- A real on-device test of the speakerphone+mic approach — confirm actual captured audio
  quality is good enough for AASIST-L and IndicConformer before trusting this design
  end to end. Check logcat for `IndicConformerDebug` tag to see per-frame latency.
- App icon, proper theming, error states, permission-rationale screens — normal Android
  app polish, deliberately skipped here since it doesn't affect whether the core
  pipeline works.
- **Language switcher** (optional) — the `IndicConformerLiveTranscriber` already accepts
  a `languageCode` parameter. To add Tamil/Telugu/Bengali support, download the
  corresponding model subfolder and pass e.g. `languageCode = "ta"` in `PipelineRunner`.
