"""Fast architecture checks; no model downloads or accuracy benchmarks.

Run with: python -m unittest discover -s tools/tests -p test_structure_compatibility.py -v
"""
import ast
import subprocess
import sys
import unittest
import tempfile
from argparse import Namespace
from pathlib import Path

from echoguard.paths import REPO_ROOT, ACOUSTIC_MODEL_DIR, SEMANTIC_ASSETS_DIR, ANDROID_ASSETS_DIR
from echoguard.paths import (DATASETS_DIR, SYNTHETIC_AUDIO_DIR, BONAFIDE_AUDIO_DIR,
    PHONE_AUDIO_DIR, EVALUATION_ROOT, EVALUATION_RESULTS_DIR,
    resolve_data_path, resolve_artifact_path, normalize_evaluation_paths,
    new_evaluation_run, check_evaluation_output)
from echoguard.semantic import scam_classifier as sc
from echoguard.fusion.fusion_engine import FusionEngine, StreamSignal


class StructureCompatibility(unittest.TestCase):
    def test_evaluation_uses_canonical_classes(self):
        from echoguard.evaluation.run_eval import load_repo
        classes = load_repo(REPO_ROOT)
        self.assertIs(classes[0], sc.ScamClassifier)
        self.assertIs(classes[1], FusionEngine)

    def test_package_import_is_lightweight(self):
        code = ("import sys; import echoguard; import echoguard.semantic.scam_classifier; "
                "import echoguard.fusion.fusion_engine; "
                "assert 'torch' not in sys.modules; "
                "assert 'sentence_transformers' not in sys.modules; "
                "assert 'onnxruntime' not in sys.modules")
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_existing_asset_paths(self):
        self.assertEqual(ACOUSTIC_MODEL_DIR, REPO_ROOT / "assets" / "acoustic")
        self.assertEqual(SEMANTIC_ASSETS_DIR, REPO_ROOT / "assets" / "semantic")
        self.assertEqual(ANDROID_ASSETS_DIR, REPO_ROOT / "app/app/src/main/assets")

    def test_current_rules_and_exemplars_are_not_expanded(self):
        self.assertEqual(len(sc.RULE_CATEGORIES), 17)
        self.assertEqual(len(sc.SCAM_EXEMPLARS), 16)
        self.assertEqual(len(sc.BENIGN_EXEMPLARS), 17)

    def test_intentional_joke_and_serious_behavior(self):
        class RulesOnly:
            def score(self, text):
                return {"score": 0.0, "available": False, "closest_exemplar": None}

        clf = sc.ScamClassifier.__new__(sc.ScamClassifier)
        clf.rule_weight = clf.semantic_weight = 0.5
        clf.semantic_scorer = RulesOnly()
        joke = clf.scam_score("Share your OTP. Just kidding, it's a prank.")
        serious = clf.scam_score("Just kidding. Actually I am serious, share your OTP.")
        self.assertTrue(joke.is_joke_override)
        self.assertEqual(joke.score, 0.0)
        self.assertFalse(serious.is_joke_override)
        self.assertGreater(serious.score, 0.5)

    def test_current_fusion_is_preserved(self):
        self.assertEqual(FusionEngine().combine(StreamSignal(1), StreamSignal(0)).risk_score, 0.45)

    def test_evaluation_metric_identity(self):
        from echoguard.evaluation import metrics
        from echoguard.evaluation import run_eval, run_eval_e2e
        self.assertIs(run_eval.st, metrics)
        self.assertIs(run_eval_e2e.st, metrics)

    def test_web_contract_and_static_location(self):
        # Parse without loading a model or importing optional web dependencies.
        source = (REPO_ROOT / "src/echoguard/web/application.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        routes = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                        if decorator.func.attr in ("get", "post") and decorator.args:
                            routes.add(ast.literal_eval(decorator.args[0]))
        self.assertEqual(routes, {"/api/analyze", "/api/health", "/api/live/start",
            "/api/live/turn", "/api/live/audio-chunk", "/api/live/end"})
        self.assertIn('static_dir = REPO_ROOT / "webdemo" / "static"', source)

    def test_existing_dataset_defaults(self):
        from echoguard.evaluation import run_eval, run_eval_e2e
        self.assertEqual(run_eval.HERE, REPO_ROOT)
        self.assertEqual(run_eval_e2e.HERE, REPO_ROOT)

    def test_retained_evaluation_helpers_are_available(self):
        from echoguard.evaluation.tools.vad_fixed import VadGateFixed
        from echoguard.evaluation.tools import phone_channel
        self.assertTrue(callable(VadGateFixed))
        self.assertTrue((Path(phone_channel.__file__)).is_file())

    def test_root_has_no_executable_python_files(self):
        self.assertEqual(list(REPO_ROOT.glob("*.py")), [])

    def test_consolidated_root_folders(self):
        for old in ("examples", "archives", "patches", "backend", "maintenance", "requirements.txt",
                    "acoustic", "semantic", "scripts", "tests", "evaluation", "requirements"):
            self.assertFalse((REPO_ROOT / old).exists(), old)
        for current in ("assets/acoustic", "assets/semantic", "tools/tests",
                        "tools/evaluation", "tools/requirements", "webdemo/static"):
            self.assertTrue((REPO_ROOT / current).is_dir(), current)
        excluded = (REPO_ROOT / ".dockerignore").read_text().splitlines()
        self.assertIn("tools", excluded)
        self.assertIn("assets/semantic/minilm_model", excluded)
        self.assertIn("COPY assets ./assets", (REPO_ROOT / "webdemo/Dockerfile").read_text())
        publishable = (REPO_ROOT / ".gitignore").read_text().splitlines()
        self.assertIn("!assets/acoustic/aasist_l.onnx", publishable)
        self.assertIn("!assets/acoustic/models/AASIST-L.pth", publishable)

    def test_organized_entry_points_exist(self):
        for relative in ("tools/models/download_kroko.py", "tools/models/download_indicconformer.py",
                         "tools/models/export_aasist.py", "tools/models/export_minilm.py",
                         "tools/models/export_exemplars.py", "tools/inspect_onnx.py",
                         "tools/examples/audio_demo.py", "tools/examples/analyze_wav.py",
                         "tools/examples/demo_pipeline.py", "tools/evaluation/run_eval.py", "tools/evaluation/device_eval.py",
                         "tools/evaluation/prepare_bonafide.py"):
            self.assertTrue((REPO_ROOT / relative).is_file(), relative)

    def test_new_evaluations_cannot_overwrite_retained_results(self):
        with self.assertRaises(ValueError):
            check_evaluation_output(EVALUATION_RESULTS_DIR, ("RESULTS.md",))
        with self.assertRaises(ValueError):
            check_evaluation_output(EVALUATION_RESULTS_DIR / "nested", ())
        with self.assertRaises(ValueError):
            check_evaluation_output(EVALUATION_ROOT, ())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "report.json").write_text("{}")
            with self.assertRaises(ValueError):
                check_evaluation_output(output, ("report.json",))
            check_evaluation_output(output, ("fresh.json",))
        self.assertNotEqual(new_evaluation_run("web"), new_evaluation_run("web"))
        for kind, target in (("python", "python"), ("web", "webdemo"), ("android-device", "android-device")):
            self.assertEqual(new_evaluation_run(kind).parent, EVALUATION_ROOT / target)

    def test_device_evaluation_is_explicitly_text_only(self):
        from echoguard.evaluation.device_eval import summarize
        rows = [{"label": "scam", "flagged": True, "language": "en"},
                {"label": "benign", "flagged": False, "language": "en"}]
        self.assertEqual(summarize(rows)["overall"]["f1"], 1.0)

    def test_compat_input_paths_resolve_to_grouped_data(self):
        for old, new in (("audio", SYNTHETIC_AUDIO_DIR), ("bonafide", BONAFIDE_AUDIO_DIR),
                         ("bonafide_phone", PHONE_AUDIO_DIR),
                         ("dataset.json", DATASETS_DIR / "dataset.json")):
            self.assertEqual(resolve_data_path(old), new)
            self.assertEqual(resolve_data_path(REPO_ROOT / old), new)

    def test_custom_existing_inputs_are_not_redirected(self):
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "audio"
            custom.mkdir()
            self.assertEqual(resolve_data_path(custom), custom)
            self.assertEqual(resolve_artifact_path(custom), custom)

    def test_compat_output_paths_resolve_to_artifacts(self):
        self.assertEqual(resolve_artifact_path("results/new.json"), EVALUATION_RESULTS_DIR / "new.json")

    def test_cli_normalization_preserves_scoring_options(self):
        args = Namespace(dev="dataset.json", heldout=["test_heldout.json"],
                         bonafide_dir=["bonafide", "bonafide_phone"], audio_dir="audio",
                         outdir="results", threshold=0.35, config="both")
        normalize_evaluation_paths(args)
        self.assertEqual(args.dev, str(DATASETS_DIR / "dataset.json"))
        self.assertEqual(args.bonafide_dir, [str(BONAFIDE_AUDIO_DIR), str(PHONE_AUDIO_DIR)])
        self.assertEqual(args.outdir, str(EVALUATION_RESULTS_DIR))
        self.assertEqual((args.threshold, args.config), (0.35, "both"))


if __name__ == "__main__":
    unittest.main()
