import argparse
import soundfile as sf

from echoguard.fusion.fusion_engine import FusionEngine, StreamSignal
from echoguard.fusion.supervisor_agent import SupervisorAgent
from echoguard.acoustic.spoof_detector_torch import RollingSpoofScorer
from echoguard.semantic.scam_classifier import ScamClassifier

def run_custom_call(audio_path, full_transcript):
    print(f"\n{'='*60}\nCUSTOM DEMO CALL: {audio_path}\n{'='*60}")

    # Load audio
    audio, sr = sf.read(audio_path, dtype="float32")
    if sr != 16000:
        raise ValueError("Audio must be 16kHz!")

    score_audio_call(audio, full_transcript)


def score_audio_call(audio, full_transcript):
    """Rolling acoustic/text demonstration shared by both transcript modes."""
    spoof_scorer = RollingSpoofScorer()
    scam_classifier = ScamClassifier()
    fusion = FusionEngine()
    agent = SupervisorAgent()

    chunk_size = 16000  # 1 second of audio
    
    # We will reveal the transcript gradually to simulate live transcription
    words = full_transcript.split()
    words_per_second = max(1, len(words) // (len(audio) // chunk_size + 1))
    
    running_transcript = ""

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            break
            
        spoof_scorer.push(chunk)
        
        # Add a few words to the transcript for this second
        word_idx = (i // chunk_size) * words_per_second
        new_words = " ".join(words[word_idx:word_idx + words_per_second])
        running_transcript += " " + new_words
        
        # This demo waits four chunks before scoring; the backend can tile shorter audio.
        if i >= chunk_size * 3:
            spoof_result = spoof_scorer.score()
            scam_result = scam_classifier.scam_score(running_transcript)

            spoof_signal = StreamSignal(
                score=spoof_result["spoof_score"],
                explain="likely AI-generated voice" if spoof_result["spoof_score"] > 0.5 else "no spoof detected",
                raw=spoof_result,
            )
            scam_signal = StreamSignal(
                score=scam_result.score,
                explain=scam_result.explain(),
                raw=None,
            )

            fusion_result = fusion.combine(spoof_signal, scam_signal)
            entry = agent.update(fusion_result)

            if entry:
                print(f"[{entry.elapsed_str}] risk={entry.risk_score:.2f} "
                      f"action={entry.recommendation.value:8s} | {entry.explanation}")
            else:
                print(f"  (tick {i//chunk_size}s: no change)")

    print(f"\n--- Final fraud timeline ({len(agent.timeline)} entries) ---")
    for e in agent.timeline:
        print(e.to_ui_dict())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to 16kHz wav file")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text", help="Full transcript of what is said in the audio")
    mode.add_argument("--transcribe", action="store_true", help="Transcribe with the existing Whisper-tiny demo")
    args = parser.parse_args()
    if args.transcribe:
        run_full_pipeline(args.audio)
    else:
        run_custom_call(args.audio, args.text)


def run_full_pipeline(audio_path):
    # Optional transcription is only loaded for this explicit demo mode.
    from transformers import pipeline

    print(f"\n{'='*60}\nFULL AUDIO PIPELINE: {audio_path}\n{'='*60}")
    print("[1/3] Loading audio file...")
    audio, sr = sf.read(audio_path, dtype="float32")
    if sr != 16000:
        raise ValueError("Audio must be 16kHz!")
    print("[2/3] Transcribing audio with Whisper (this may take a few seconds on CPU)...")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", chunk_length_s=30)
    result = asr({"raw": audio, "sampling_rate": 16000})
    full_transcript = result["text"].strip()
    print(f"\n>> Transcribed Text: '{full_transcript}'\n")
    print("[3/3] Running Fusion Engine Analysis...\n")
    score_audio_call(audio, full_transcript)


if __name__ == "__main__":
    main()
