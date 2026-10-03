"""
demo_pipeline.py
------------------
Synthetic waveform and scripted transcript demonstration: PyTorch AASIST-L and
ScamClassifier feed FusionEngine and SupervisorAgent to print a fraud timeline.
MiniLM is used when its model can load; otherwise the classifier uses rules.
This is not a microphone, ASR, Android, or dataset accuracy evaluation.
"""

import numpy as np

from echoguard.fusion.fusion_engine import FusionEngine, StreamSignal
from echoguard.fusion.supervisor_agent import SupervisorAgent


def run_demo_call(transcript_turns, audio_chunks, label=""):
    """
    transcript_turns: list of strings, one per "tick" of the simulated call
    audio_chunks: list of 1-D float32 numpy arrays, one per tick (same length)
    """
    from echoguard.acoustic.spoof_detector_torch import RollingSpoofScorer
    from echoguard.semantic.scam_classifier import ScamClassifier

    print(f"\n{'='*60}\nDEMO CALL: {label}\n{'='*60}")

    spoof_scorer = RollingSpoofScorer()
    scam_classifier = ScamClassifier()
    fusion = FusionEngine()
    agent = SupervisorAgent()

    running_transcript = ""
    for i, (chunk, new_text) in enumerate(zip(audio_chunks, transcript_turns)):
        spoof_scorer.push(chunk)
        running_transcript += " " + new_text

        spoof_result = spoof_scorer.score()
        scam_result = scam_classifier.scam_score(running_transcript)

        spoof_signal = StreamSignal(
            score=spoof_result["spoof_score"],
            explain="likely AI-generated voice" if spoof_result["spoof_score"] > 0.5
                     else "no spoof detected",
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
            print(f"  (tick {i}: no change)")

    print(f"\n--- Final fraud timeline ({len(agent.timeline)} entries) ---")
    for e in agent.timeline:
        print(e.to_ui_dict())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Low-amplitude noise standing in for real audio in this sandbox -
    # swap in real recorded chunks (16kHz mono) when running this for real.
    chunk = lambda: (rng.standard_normal(16000) * 0.05).astype(np.float32)

    run_demo_call(
        label="benign call",
        transcript_turns=[
            "Hi, just calling to check how you're doing.",
            "Can we reschedule our meeting to next week?",
            "Great, talk soon, bye!",
        ],
        audio_chunks=[chunk(), chunk(), chunk()],
    )

    run_demo_call(
        label="escalating scam call",
        transcript_turns=[
            "Hello, this is calling from your bank's security department.",
            "We've noticed suspicious activity, your account will be blocked.",
            "Please share the OTP that was just sent to your phone, this is urgent.",
        ],
        audio_chunks=[chunk(), chunk(), chunk()],
    )
