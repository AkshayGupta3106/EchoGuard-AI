"""No real accuracy run: simulate input/model failures and verify report safety."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from echoguard.evaluation import run_eval as evaluation
from echoguard.paths import REPO_ROOT


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "dev.json"
    path.write_text(json.dumps([{"id": "example", "language": "en", "category": "fixture", "label": "benign", "turns": [{"text": "hello"}]}]), encoding="utf-8")
    return path


def arguments(monkeypatch, dataset, output, *extra):
    monkeypatch.setattr("sys.argv", ["eval", "--repo", str(REPO_ROOT), "--config", "rules", "--dev", str(dataset), "--outdir", str(output), *extra])


@pytest.mark.parametrize("kind", ["missing_dev", "missing_heldout", "invalid_dataset", "empty_dataset", "missing_audio", "empty_audio", "missing_vad", "nan_threshold", "zero_repeat"])
def test_incomplete_inputs_fail_before_reports(tmp_path, dataset, monkeypatch, kind):
    output = tmp_path / "run"
    extra = ["--heldout", "--no-acoustic"]
    if kind == "missing_dev":
        dataset.unlink()
    elif kind == "missing_heldout":
        extra = ["--heldout", str(tmp_path / "missing.json"), "--no-acoustic"]
    elif kind == "invalid_dataset":
        dataset.write_text('[{"label": "unknown"}]', encoding="utf-8")
    elif kind == "empty_dataset":
        dataset.write_text("[]", encoding="utf-8")
    elif kind in ("missing_audio", "empty_audio"):
        directory = tmp_path / "audio"
        if kind == "empty_audio":
            directory.mkdir()
        extra = ["--heldout", "--audio-dir", str(directory)]
    elif kind == "missing_vad":
        extra = ["--heldout", "--vad-model", str(tmp_path / "missing.onnx")]
    elif kind == "nan_threshold":
        extra += ["--threshold", "nan"]
    else:
        extra += ["--repeat", "0"]
    arguments(monkeypatch, dataset, output, *extra)
    with pytest.raises(SystemExit) as error:
        evaluation.main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("requested", ["minilm", "both"])
def test_missing_minilm_is_not_a_successful_rules_only_run(requested):
    class UnavailableClassifier:
        def __init__(self):
            self.semantic_scorer = SimpleNamespace(available=False, model=None)

    with pytest.raises(ValueError, match="Requested MiniLM configuration"):
        evaluation.build_classifiers(UnavailableClassifier, requested)
    classifiers, active = evaluation.build_classifiers(UnavailableClassifier, "rules")
    assert list(classifiers) == ["rules"] and active is False


def test_acoustic_failure_saves_diagnostics_and_exits_unsuccessfully(tmp_path, dataset, monkeypatch):
    output = tmp_path / "run"
    arguments(monkeypatch, dataset, output, "--heldout")

    class Classifier:
        def scam_score(self, text):
            return SimpleNamespace(score=0, rule_hits=[], explain=lambda: "fixture")

    monkeypatch.setattr(evaluation, "build_classifiers", lambda *args: ({"rules": Classifier()}, False))

    def acoustic_failure(*args):
        raise RuntimeError("simulated acoustic failure")

    monkeypatch.setattr(evaluation, "eval_acoustic", acoustic_failure)
    with pytest.raises(SystemExit, match="saved partial diagnostics"):
        evaluation.main()

    def invalid_constant(value):
        pytest.fail("Invalid JSON constant: " + value)

    report = json.loads((output / "eval_results.json").read_text(encoding="utf-8"), parse_constant=invalid_constant)
    assert report["evaluation_complete"] is False
    assert report["acoustic"]["error"] == "RuntimeError: simulated acoustic failure"
    assert report["text"]["DEV (in-sample, rules were tuned on it)"]["rules"]["n_calls"] == 1
    assert "Evaluation complete: **False**" in (output / "RESULTS.md").read_text(encoding="utf-8")


def test_report_writer_rejects_late_output_collision(tmp_path):
    path = tmp_path / "RESULTS.md"
    path.write_text("existing report", encoding="utf-8")
    report = {"meta": {"date": "fixture", "repo_head": "unknown", "minilm_loadable": False, "threshold": 0.35,
                       "machine": {"platform": "fixture", "cpu_count": 1, "onnxruntime": None}}, "text": {}}
    with pytest.raises(FileExistsError):
        evaluation.write_report(path, report)
    assert path.read_text(encoding="utf-8") == "existing report"


def test_empty_audio_window_rejected_without_nan_metrics():
    with pytest.raises(ValueError, match="nonempty"):
        evaluation.windows_of(np.zeros(0), None)
