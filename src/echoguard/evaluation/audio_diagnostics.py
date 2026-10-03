"""Audit existing WAV files without calibration or changes to protected scoring."""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from echoguard.acoustic.spoof_detector_onnx import SpoofDetector, WINDOW_SAMPLES
from echoguard.acoustic.vad_gate import VadGate
from echoguard.evaluation.tools.vad_fixed import VadGateFixed
from echoguard.evaluation.audio_replay import collect_clips
from echoguard.paths import ACOUSTIC_MODEL_DIR, new_evaluation_run


def main():
    output = new_evaluation_run("python")
    output.mkdir(parents=True, exist_ok=False)
    detector = SpoofDetector()
    runtime_vad = VadGate()
    reference_vad = VadGateFixed(ACOUSTIC_MODEL_DIR / "silero_vad.onnx")
    rows = []
    for clip in collect_clips():
        samples, rate = sf.read(clip["path"], dtype="float32")
        if rate != 16000 or samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
            raise ValueError("Invalid WAV: " + clip["id"])
        results = [detector.predict_array(samples[offset:offset + WINDOW_SAMPLES])
                   for offset in range(0, len(samples), WINDOW_SAMPLES)]
        probabilities = [result["spoof_score"] for result in results]
        row = {"id": clip["id"], "group": clip["group"], "language": clip["language"],
               "sha256": hashlib.sha256(Path(clip["path"]).read_bytes()).hexdigest(),
               "windows": len(results), "mean_spoof": float(np.mean(probabilities)),
               "spoof_above_half_fraction": float(np.mean(np.array(probabilities) > .5)),
               "acoustic_only_warn_window_fraction": float(np.mean(np.array(probabilities) * .45 >= .35)),
               "acoustic_only_peak_risk": max(probabilities) * .45,
               "raw_margins": [result["spoof_margin"] for result in results]}
        if clip["group"] != "synthetic":
            runtime_vad.reset(); reference_vad.reset()
            chunks = [samples[offset:offset + 512] for offset in range(0, len(samples) - 511, 512)]
            row["runtime_vad_speech_fraction"] = float(np.mean([runtime_vad.is_speech(chunk)["is_speech"] for chunk in chunks])) if chunks else None
            row["context_vad_speech_fraction"] = float(np.mean([reference_vad.prob(chunk) >= reference_vad.threshold for chunk in chunks])) if chunks else None
        rows.append(row)
    summary = {}
    for group in sorted({row["group"] for row in rows}):
        subset = [row for row in rows if row["group"] == group]
        summary[group] = {"n": len(subset), "windows": sum(row["windows"] for row in subset),
            "clips_reaching_acoustic_only_warn": sum(row["acoustic_only_peak_risk"] >= .35 for row in subset),
            "mean_spoof": float(np.average([row["mean_spoof"] for row in subset], weights=[row["windows"] for row in subset]))}
        if group != "synthetic":
            measured = [row for row in subset if row["runtime_vad_speech_fraction"] is not None]
            summary[group].update(vad_measured_clips=len(measured),
                mean_runtime_vad_fraction=float(np.mean([row["runtime_vad_speech_fraction"] for row in measured])) if measured else None,
                mean_context_vad_fraction=float(np.mean([row["context_vad_speech_fraction"] for row in measured])) if measured else None)
    with (output / "audio_diagnostics.json").open("x", encoding="utf-8") as report:
        json.dump({"scope": "python_existing_audio_acoustic_vad_diagnostics", "summary": summary, "rows": rows,
            "limitations": ["Windows not speech-selected; short final windows use unchanged tiling",
                "Private phone source/language labels not independently verified",
                "Context VAD is a diagnostic comparison only; runtime behavior unchanged",
                "Acoustic-only warning is protected policy, not proof of a scam"]}, report, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    print("Local results:", output)


if __name__ == "__main__":
    main()
