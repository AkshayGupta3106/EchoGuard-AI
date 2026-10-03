"""Keep private/generated files local without hiding required repository inputs."""
import shutil
import subprocess

import pytest

from echoguard.paths import REPO_ROOT


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="Git is not installed")


def ignored(path):
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path],
        cwd=REPO_ROOT, capture_output=True,
    )
    assert result.returncode in (0, 1), result.stderr.decode(errors="replace")
    return result.returncode == 0


def test_recordings_history_and_secrets_stay_local():
    for path in (
        "data/audio/bonafide_phone/manifest.csv", "data/audio/synthetic/clip.wav",
        "recordings/clip.m4a", "recordings/clip.pcm", "exports/call_history.json",
        ".env", ".env.production", "app/signing/release.jks",
    ):
        assert ignored(path), path


def test_generated_outputs_and_caches_stay_local():
    for path in (
        "artifacts/evaluation/python/run/RESULTS.md", "artifacts/session.json",
        "app/app/build/outputs/apk/debug/app-debug.apk", "app/.gradle/cache.bin",
        ".pytest_cache/cache", "src/echoguard/__pycache__/paths.pyc",
        ".venv/pyvenv.cfg", ".hf_cache/download", "node_modules/package/index.js",
        "coverage/report.html", ".vscode/settings.json",
        "app/app/src/main/assets/models/minilm.onnx", "app/app/libs/sherpa.aar",
    ):
        assert ignored(path), path


def test_source_datasets_licenses_and_deployment_assets_remain_publishable():
    for path in (
        "README.md", "docs/guide.md", "docs/evaluation.md", ".env.example",
        "src/echoguard/paths.py", "tools/check_source.py", "webdemo/Dockerfile",
        "webdemo/static/index.html", "data/datasets/dataset.json",
        "app/gradle/wrapper/gradle-wrapper.jar", "app/app/src/main/AndroidManifest.xml",
        "assets/acoustic/aasist_l.onnx", "assets/acoustic/models/AASIST-L.pth",
        "assets/acoustic/LICENSE_AASIST", "assets/acoustic/models/AASIST.py",
        "assets/semantic/exemplar_embeddings.json",
    ):
        assert not ignored(path), path
