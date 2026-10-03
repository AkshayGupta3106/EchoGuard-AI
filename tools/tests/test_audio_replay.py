"""Replay metrics/provenance unit checks; no device or microphone required."""
import math
import json

import pytest

from echoguard.evaluation.audio_replay import tokens, edit_distance, add_error_rates, collect_clips, summarize


def test_normalization_retains_hindi_marks_and_latin_digits():
    assert tokens("नमस्ते, बैंक! OTP १२३") == ["नमस्ते", "बैंक", "otp", "१२३"]
    assert tokens(" HELLO—World. ") == ["hello", "world"]


@pytest.mark.parametrize("left,right,distance", [([], [], 0), (["a"], [], 1), ([], ["a"], 1),
    (["a", "b"], ["a", "x", "b"], 1), (["a", "b"], ["x", "y"], 2)])
def test_levenshtein(left, right, distance):
    assert edit_distance(left, right) == distance


def test_empty_recognition_is_full_reference_error_not_skipped():
    row = {"transcript": ""}
    add_error_rates(row, "Hello there")
    assert row["wer"] == 1 and row["cer"] == 1
    with pytest.raises(ValueError, match="Empty reference"):
        add_error_rates(row, "... ")


def test_references_are_verified_only_for_public_fixtures():
    clips = collect_clips()
    public = [clip for clip in clips if clip["group"] == "bonafide"]
    assert len(public) == 9 and all(clip["reference"] for clip in public)
    assert all(clip["reference"] is None for clip in clips if clip["group"] != "bonafide")
    assert {clip["language"] for clip in clips if clip["group"] == "synthetic"} == {"en", "hi"}


def test_summary_weights_reference_words_and_does_not_invent_phone_accuracy():
    rows = []
    for reference, text in [("hello there", "hello"), ("good morning to you", "good morning")]:
        row = {"group": "bonafide", "language": "en", "label": None, "transcript": text,
               "asr_ms": 50, "duration_seconds": 1, "scam_score": 0}
        add_error_rates(row, reference)
        rows.append(row)
    rows.append({"group": "bonafide_phone", "language": "en", "label": None, "transcript": "",
                 "asr_ms": 50, "duration_seconds": 1, "scam_score": 0})
    report = summarize(rows)
    assert report["bonafide"]["wer"] == .5
    assert "wer" not in report["bonafide_phone"]
    assert report["bonafide_phone"]["nonempty"] == 0


def test_single_class_summary_is_strict_json():
    row = {"group": "synthetic", "language": "en", "label": "benign", "transcript": "hello",
           "asr_ms": 50, "duration_seconds": 1, "scam_score": 0}
    json.dumps(summarize([row]), allow_nan=False)


def test_margin_diagnostic_does_not_change_probability_contract():
    from echoguard.acoustic.spoof_detector_onnx import SpoofDetector
    import numpy as np
    detector = SpoofDetector()
    result = detector.predict_array(np.zeros(64600, dtype=np.float32))
    assert result["spoof_score"] == pytest.approx(1 / (1 + math.exp(-result["spoof_margin"])), abs=.0001)


@pytest.mark.parametrize("length", [100, 1024])
def test_acoustic_audit_uses_context_probability_contract_and_strict_json(tmp_path, monkeypatch, length):
    import numpy as np
    import soundfile as sf
    from echoguard.evaluation import audio_diagnostics as audit
    path = tmp_path / "fixture.wav"
    sf.write(path, np.zeros(length, dtype=np.float32), 16000)
    output = tmp_path / "report"
    monkeypatch.setattr(audit, "new_evaluation_run", lambda kind: output)
    monkeypatch.setattr(audit, "collect_clips", lambda: [{"path": str(path), "id": "fixture", "group": "bonafide", "language": "en"}])

    class Detector:
        def predict_array(self, samples):
            return {"spoof_score": .9, "spoof_margin": 2.0}

    class RuntimeVad:
        def reset(self):
            pass

        def is_speech(self, chunk):
            return {"is_speech": False}

    class ContextVad:
        threshold = .5

        def __init__(self, path):
            pass

        def reset(self):
            pass

        def prob(self, chunk):
            return .9

    monkeypatch.setattr(audit, "SpoofDetector", Detector)
    monkeypatch.setattr(audit, "VadGate", RuntimeVad)
    monkeypatch.setattr(audit, "VadGateFixed", ContextVad)
    audit.main()
    report = json.loads((output / "audio_diagnostics.json").read_text())
    assert report["summary"]["bonafide"]["clips_reaching_acoustic_only_warn"] == 1
    assert report["rows"][0]["context_vad_speech_fraction"] == (1.0 if length >= 512 else None)
    json.dumps(report, allow_nan=False)
