# import librosa
# import soundfile as sf
# from spoof_detector import SpoofDetector

# # file_path = "/data/b23_chirag_bhutra/EchoGuard_AI/acoustic/LingaSir_ReferenceVoice.wav"
# # new_path = "/data/b23_chirag_bhutra/EchoGuard_AI/acoustic/LingaSir_real_16k.wav"
# file_path = "/data/b23_chirag_bhutra/EchoGuard_AI/acoustic/LingaSir_OutputVoice_speed=1.4.wav"
# new_path = "/data/b23_chirag_bhutra/EchoGuard_AI/acoustic/LingaSir_fake_16k.wav"

# print("Resampling audio to 16kHz...")
# # librosa automatically resamples to the target sr when loading
# audio, _ = librosa.load(file_path, sr=16000)

# # Save the properly formatted 16kHz audio
# sf.write(new_path, audio, 16000)
# print("Saved 16kHz version!")

# # Run the spoof detector on the new file
# detector = SpoofDetector()
# result = detector.predict_file(new_path)
# print("\n--- Final Score ---")
# print(result)


import soundfile as sf
import numpy as np
from spoof_detector import SpoofDetector, RollingSpoofScorer

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
        
        # AASIST needs at least 4 seconds in its buffer to make a reliable prediction
        # So we only start scoring after we've pushed at least 4 chunks (4 seconds)
        if i >= chunk_size * 3:
            result = scorer.detector.predict_array(np.array(scorer._buffer))
            score = result['spoof_score']
            scores.append(score)
            
            # Print the score at this specific second
            current_sec = (i + chunk_size) / 16000
            print(f"Score at {int(current_sec)} seconds: {score}")

    print(f"-> AVERAGE Spoof Score: {round(sum(scores)/len(scores), 4) if scores else 'N/A'}")

# Run both files
test_full_audio("/data/b23_chirag_bhutra/EchoGuard_AI/acoustic/LingaSir_real_16k.wav")
test_full_audio("/data/b23_chirag_bhutra/EchoGuard_AI/acoustic/LingaSir_fake_16k.wav")