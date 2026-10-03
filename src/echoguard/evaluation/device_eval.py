"""Run debug-only Android instrumentation without microphone capture.

Evaluates supplied-text semantic/fusion backends on a real Android device.
This is NOT end-to-end microphone/ASR or audio accuracy evaluation.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from echoguard.paths import REPO_ROOT, resolve_data_path, check_evaluation_output, check_dataset_names, new_evaluation_run
from echoguard.evaluation import metrics

PACKAGE = "com.echoguard"
RUNNER = "com.echoguard.test/androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS = "com.echoguard.evaluation.DeviceEvaluationTest"


def upload_dataset(adb, path):
    """Push a dataset and verify its exact bytes before instrumentation."""
    payload = path.read_bytes()
    staging = f"/data/local/tmp/echoguard-eval-{uuid4().hex}.json"
    destination = f"files/evaluation/{path.name}"
    try:
        subprocess.run([*adb, "push", str(path), staging], check=True)
        subprocess.run([*adb, "shell", "chmod", "644", staging], check=True)
        subprocess.run([*adb, "shell", "run-as", PACKAGE, "cp", staging, destination], check=True)
        uploaded = subprocess.check_output([*adb, "exec-out", "run-as", PACKAGE, "cat", destination])
        if uploaded != payload:
            raise RuntimeError(
                f"Dataset transfer verification failed for {path.name}: "
                f"expected {len(payload)} bytes, received {len(uploaded)}. Evaluation not started."
            )
    finally:
        subprocess.run([*adb, "shell", "rm", "-f", staging], check=False)
    return payload


def instrumentation(adb, method, dataset=None):
    command = [*adb, "shell", "am", "instrument", "-w", "-e", "class", f"{TEST_CLASS}#{method}"]
    if dataset:
        command.extend(["-e", "dataset", dataset])
    result = subprocess.run([*command, RUNNER], capture_output=True, text=True, timeout=600)
    print(result.stdout, end="")
    if result.returncode or not re.search(r"OK \([1-9]\d* tests?\)", result.stdout) or "FAILURES!!!" in result.stdout:
        raise RuntimeError(f"Instrumentation failed; no valid evaluation produced.\n{result.stderr}")


def summarize(rows):
    summary = {"overall": metrics.confusion([r["label"] == "scam" for r in rows],
                                            [r["flagged"] for r in rows])}
    for language in sorted({r["language"] for r in rows}):
        subset = [r for r in rows if r["language"] == language]
        summary[language] = metrics.confusion([r["label"] == "scam" for r in subset],
                                              [r["flagged"] for r in subset])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="ADB device serial; required when multiple devices are connected")
    parser.add_argument("--dataset", nargs="+", help="Dataset JSON files; omit for native-model smoke only")
    parser.add_argument("--skip-build", action="store_true", help="Use already built debug app/test APKs")
    parser.add_argument("--outdir", type=Path, default=None, help="New output directory (never overwrite protected results)")
    args = parser.parse_args()
    adb_exe = shutil.which("adb")
    if not adb_exe:
        parser.error("adb is not on PATH; add Android SDK/platform-tools")
    devices = subprocess.check_output([adb_exe, "devices"], text=True)
    connected = [line.split()[0] for line in devices.splitlines()[1:]
                 if len(line.split()) >= 2 and line.split()[1] == "device"]
    serial = args.serial or (connected[0] if len(connected) == 1 else None)
    if serial not in connected:
        parser.error("Connect/unlock and authorize a device; specify --serial when needed")
    adb = [adb_exe, "-s", serial]
    datasets = [resolve_data_path(value) for value in args.dataset or []]
    for path in datasets:
        if not path.is_file() or not re.fullmatch(r"[A-Za-z0-9_-]+\.json", path.name):
            parser.error(f"Missing dataset or unsupported filename: {path}")
        try:
            calls = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(calls, list) or not calls:
                raise ValueError("Dataset must be a nonempty list")
            for call in calls:
                if call["label"] not in ("scam", "benign") or not call["turns"]:
                    raise ValueError("Calls need a scam/benign label and nonempty turns")
                if not isinstance(call["id"], str) or not isinstance(call["language"], str):
                    raise ValueError("IDs and languages must be strings")
                if any(not isinstance(turn["text"], str) for turn in call["turns"]):
                    raise ValueError("Turn text must be a string")
            if len({call["id"] for call in calls}) != len(calls):
                raise ValueError("Duplicate call IDs")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"Invalid dataset {path}: {exc}")
    try:
        check_dataset_names(datasets)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.outdir or new_evaluation_run("android-device")
    if output.exists():
        parser.error(f"Output directory already exists; choose a new directory: {output}")
    try:
        check_evaluation_output(output, ())
    except ValueError as exc:
        parser.error(str(exc))
    if not args.skip_build:
        wrapper = "gradlew.bat" if os.name == "nt" else "gradlew"
        subprocess.run([str(REPO_ROOT / "app" / wrapper), "--no-daemon", "assembleDebug", "assembleDebugAndroidTest"],
                       cwd=REPO_ROOT / "app", check=True)
    for relative in ("app/app/build/outputs/apk/debug/app-debug.apk",
                     "app/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"):
        subprocess.run([*adb, "install", "-r", str(REPO_ROOT / relative)], check=True)
    # Native readiness must pass before any metrics are accepted.
    instrumentation(adb, "nativeModelSmoke")
    if not datasets:
        print("Native-model/English-ASR smoke passed. No benchmark was run and no microphone was used.")
        return
    output.mkdir(parents=True)
    try:
        subprocess.run([*adb, "shell", "run-as", PACKAGE, "mkdir", "-p", "files/evaluation"], check=True)
        for path in datasets:
            payload = upload_dataset(adb, path)
            subprocess.run([*adb, "shell", "run-as", PACKAGE, "rm", "-f", "files/evaluation/result.json"], check=True)
            instrumentation(adb, "evaluateTextDataset", path.name)
            raw = subprocess.check_output([*adb, "exec-out", "run-as", PACKAGE, "cat", "files/evaluation/result.json"])
            result = json.loads(raw)
            expected = json.loads(payload)
            if not result.get("minilm_available") or [r["id"] for r in result["rows"]] != [r["id"] for r in expected]:
                raise RuntimeError("Device result is incomplete or uses a fallback")
            result["dataset_sha256"] = hashlib.sha256(payload).hexdigest()
            result["metrics"] = summarize(result["rows"])
            result["model_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (REPO_ROOT / "app/app/src/main/assets/models").iterdir() if p.is_file()}
            (output / path.name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(path.name, result["metrics"]["overall"])
    finally:
        # Remove only the test inputs/results we placed in the app, not app history.
        for name in [p.name for p in datasets] + ["result.json"]:
            subprocess.run([*adb, "shell", "run-as", PACKAGE, "rm", "-f", f"files/evaluation/{name}"], check=False)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
