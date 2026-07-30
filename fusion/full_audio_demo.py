import sys
import argparse
import soundfile as sf
import numpy as np
from pathlib import Path
from transformers import pipeline

# Setup paths to import acoustic and semantic modules
_FUSION_DIR = Path(__file__).parent
sys.path.insert(0, str(_FUSION_DIR))
sys.path.insert(0, str(_FUSION_DIR.parent / "acoustic"))
sys.path.insert(0, str(_FUSION_DIR.parent / "semantic"))

from fusion_engine import FusionEngine, StreamSignal
from supervisor_agent import SupervisorAgent
from spoof_detector import RollingSpoofScorer
from scam_classifier import ScamClassifier

def run_full_pipeline(audio_path):
    print(f"\n{'='*60}\nFULL AUDIO PIPELINE: {audio_path}\n{'='*60}")
    
    # 1. Load Audio
    print("[1/3] Loading audio file...")
    audio, sr = sf.read(audio_path, dtype="float32")
    if sr != 16000:
        raise ValueError("Audio must be 16kHz!")
        
    # 2. Transcribe Audio using Whisper
    print("[2/3] Transcribing audio with Whisper (this may take a few seconds on CPU)...")
    # Using whisper-tiny to keep it fast on CPU
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", chunk_length_s=30)
    
    result = asr({"raw": audio, "sampling_rate": 16000})
    full_transcript = result["text"].strip()
    print(f"\n>> Transcribed Text: '{full_transcript}'\n")
    
    # 3. Run Fusion Engine
    print("[3/3] Running Fusion Engine Analysis...\n")
    spoof_scorer = RollingSpoofScorer()
    scam_classifier = ScamClassifier()
    fusion = FusionEngine()
    agent = SupervisorAgent()

    chunk_size = 16000  # 1 second of audio
    
    # Split the transcript so we can reveal it gradually to simulate a live call
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
        
        # We need at least 4 seconds of audio in the buffer before AASIST will output a score
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to 16kHz wav file")
    args = parser.parse_args()
    
    run_full_pipeline(args.audio)
