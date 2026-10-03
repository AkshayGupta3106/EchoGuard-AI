"""Rolling PyTorch acoustic inference on supplied 16 kHz WAV files."""
import soundfile as sf
import numpy as np
from echoguard.acoustic.spoof_detector_torch import SpoofDetector, RollingSpoofScorer

def test_full_audio(wav_path):
    print(f"\n--- Testing: {wav_path} ---")
    
    # Load the 16kHz audio
    audio, sr = sf.read(wav_path, dtype="float32")
    
    # Initialize the live rolling scorer
    scorer = RollingSpoofScorer(SpoofDetector())
    
    # We will feed the audio in 1-second chunks (16,000 samples)
    chunk_size = 16000 
    scores = []
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        
        # Skip the last chunk if it's too small
        if len(chunk) < chunk_size:
            break
            
        scorer.push(chunk)
        
        # This demo waits four chunks before scoring; the backend can tile shorter audio.
        if i >= chunk_size * 3:
            result = scorer.detector.predict_array(np.array(scorer._buffer))
            score = result['spoof_score']
            scores.append(score)
            
            # Print the score at this specific second
            current_sec = (i + chunk_size) / 16000
            print(f"Score at {int(current_sec)} seconds: {score}")

    print(f"-> AVERAGE Spoof Score: {round(sum(scores)/len(scores), 4) if scores else 'N/A'}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the existing rolling spoof demonstration on supplied WAV files")
    parser.add_argument("audio", nargs="+", help="Path(s) to 16kHz WAV files")
    args = parser.parse_args()
    for path in args.audio:
        test_full_audio(path)


if __name__ == "__main__":
    main()
