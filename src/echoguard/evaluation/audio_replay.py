"""Local-only Android ASR file replay. This does not capture calls or microphone audio."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from echoguard.paths import REPO_ROOT, new_evaluation_run
from echoguard.evaluation.metrics import confusion, json_safe
from echoguard.evaluation.device_eval import PACKAGE, RUNNER, upload_dataset


def tokens(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c if unicodedata.category(c)[0] in "LNM" or c.isspace() else " " for c in text).split()


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def add_error_rates(row, reference):
    ref, hyp = tokens(reference), tokens(row["transcript"])
    ref_chars, hyp_chars = list("".join(ref)), list("".join(hyp))
    if not ref:
        raise ValueError("Empty reference transcript")
    row.update(reference_words=len(ref), word_edits=edit_distance(ref, hyp),
               reference_characters=len(ref_chars), character_edits=edit_distance(ref_chars, hyp_chars))
    row["wer"] = row["word_edits"] / len(ref)
    row["cer"] = row["character_edits"] / len(ref_chars)


def collect_clips():
    dataset = json.loads((REPO_ROOT / "data/datasets/dataset.json").read_text(encoding="utf-8"))
    by_id = {call["id"]: call for call in dataset}
    public_refs = {}
    for line in (REPO_ROOT / "docs/evaluation.md").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"\| ((?:ljspeech_|jfk_)[^ |]+\.wav) \| (.+) \|", line)
        if match:
            public_refs[match[1]] = match[2]
    clips = []
    for group in ("bonafide", "synthetic", "bonafide_phone"):
        for path in sorted((REPO_ROOT / "data/audio" / group).glob("*.wav")):
            call = by_id.get(path.stem) if group == "synthetic" else None
            clips.append({"id": f"{group}/{path.stem}", "path": str(path), "group": group,
                          "language": call["language"] if call else "en",
                          "language_basis": "dataset" if call else ("public_source" if group == "bonafide" else "English_model_probe_language_unverified"),
                          "label": call["label"] if call else None,
                          "reference": public_refs.get(path.name),
                          "candidate_dataset_script": " ".join(turn["text"] for turn in call["turns"]) if call else None})
    return clips


def summarize(rows):
    output = {}
    for group in sorted({row["group"] for row in rows}):
        subset = [row for row in rows if row["group"] == group]
        refs = [row for row in subset if "word_edits" in row]
        entry = {"n": len(subset), "nonempty": sum(bool(row["transcript"]) for row in subset),
                 "total_audio_seconds": sum(row["duration_seconds"] for row in subset),
                 "decode_rtf": sum(row["asr_ms"] for row in subset) / 1000 / sum(row["duration_seconds"] for row in subset),
                 "verified_reference_count": len(refs)}
        if refs:
            entry.update(wer=sum(row["word_edits"] for row in refs) / sum(row["reference_words"] for row in refs),
                         cer=sum(row["character_edits"] for row in refs) / sum(row["reference_characters"] for row in refs))
        for language in sorted({row["language"] for row in subset}):
            labeled = [row for row in subset if row["language"] == language and row["label"] is not None]
            if labeled:
                entry[language + "_final_asr_text_semantic"] = confusion([row["label"] == "scam" for row in labeled],
                    [row["scam_score"] >= 0.35 for row in labeled])
        output[group] = entry
    return json_safe(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--limit", type=int, help="Maximum clips per language/group (pilot only)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("Limit must be positive")
    adb = [shutil.which("adb") or "adb", "-s", args.serial]
    clips = collect_clips()
    if args.limit:
        counts = {}
        selected = []
        for clip in clips:
            key = clip["group"], clip["language"]
            counts[key] = counts.get(key, 0) + 1
            if counts[key] <= args.limit:
                selected.append(clip)
        clips = selected
    output = new_evaluation_run("android-device")
    output.mkdir(parents=True, exist_ok=False)
    model_root = REPO_ROOT / "app/app/src/main/assets"
    metadata = {"app_apk_sha256": hashlib.sha256((REPO_ROOT / "app/app/build/outputs/apk/debug/app-debug.apk").read_bytes()).hexdigest(),
        "model_sha256": {str(path.relative_to(model_root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                         for folder in ("kroko-128l", "indicconformer-hi", "models")
                         for path in sorted((model_root / folder).rglob("*")) if path.is_file()},
        "device_model": subprocess.check_output([*adb, "shell", "getprop", "ro.product.model"], text=True).strip(),
        "sdk": subprocess.check_output([*adb, "shell", "getprop", "ro.build.version.sdk"], text=True).strip()}
    with (output / "metadata.json").open("x", encoding="utf-8") as report:
        json.dump(metadata, report, indent=2)
    print("Local run:", output, flush=True)
    rows = []
    for language in ("en", "hi"):
        selected = [clip for clip in clips if clip["language"] == language]
        if not selected:
            continue
        name = "asr-" + uuid4().hex
        remote = f"files/evaluation/{name}"
        staged_names = []
        try:
            subprocess.run([*adb, "shell", "run-as", PACKAGE, "mkdir", "-p", remote], check=True)
            temp_root = Path(tempfile.gettempdir()) / "opencode"
            with tempfile.TemporaryDirectory(prefix="echoguard-asr-", dir=temp_root if temp_root.is_dir() else None) as temp:
                manifest_clips = []
                for i, clip in enumerate(selected):
                    audio, rate = sf.read(clip["path"], dtype="float32")
                    if rate != 16000 or audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
                        raise ValueError(f"Replay requires finite/nonempty mono 16 kHz: {clip['id']}")
                    filename = f"clip-{i}.pcm"
                    payload = audio.astype("<f4").tobytes()
                    path = Path(temp) / filename
                    path.write_bytes(payload)
                    stage = "/data/local/tmp/" + name + "-" + filename
                    staged_names.append(stage)
                    subprocess.run([*adb, "push", str(path), stage], check=True, stdout=subprocess.DEVNULL)
                    subprocess.run([*adb, "shell", "chmod", "644", stage], check=True)
                    subprocess.run([*adb, "shell", "run-as", PACKAGE, "cp", stage, remote + "/" + filename], check=True)
                    received = subprocess.check_output([*adb, "exec-out", "run-as", PACKAGE, "cat", remote + "/" + filename])
                    if received != payload:
                        raise RuntimeError("PCM transfer mismatch: " + clip["id"])
                    clip["audio_sha256"] = hashlib.sha256(Path(clip["path"]).read_bytes()).hexdigest()
                    manifest_clips.append({"id": clip["id"], "pcm_file": filename})
                    subprocess.run([*adb, "shell", "rm", "-f", stage], check=True)
                manifest_path = Path(temp) / (name + ".json")
                manifest_path.write_text(json.dumps({"language": language, "clips": manifest_clips}), encoding="utf-8")
                upload_dataset(adb, manifest_path)
                subprocess.run([*adb, "shell", "run-as", PACKAGE, "mv", "files/evaluation/" + manifest_path.name, remote + "/manifest.json"], check=True)
            result = subprocess.run([*adb, "shell", "am", "instrument", "-w", "-r", "-e", "class",
                "com.echoguard.evaluation.AudioReplayTest#replay", "-e", "audio_run", name, RUNNER], capture_output=True, text=True, timeout=3600)
            (output / f"instrumentation-{language}.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode or "OK (1 test)" not in result.stdout or "INSTRUMENTATION_STATUS_CODE: -3" in result.stdout or "FAILURES!!!" in result.stdout:
                raise RuntimeError("ASR replay failed; see " + str(output))
            raw = json.loads(subprocess.check_output([*adb, "exec-out", "run-as", PACKAGE, "cat", remote + "/result.json"]))
            if not raw["complete"] or [row["id"] for row in raw["rows"]] != [clip["id"] for clip in selected]:
                raise RuntimeError("Incomplete ASR replay")
            for row, clip in zip(raw["rows"], selected):
                row.update({key: value for key, value in clip.items() if key != "path"})
                if clip["reference"]:
                    add_error_rates(row, clip["reference"])
                if clip["candidate_dataset_script"]:
                    probe = {"transcript": row["transcript"]}
                    add_error_rates(probe, clip["candidate_dataset_script"])
                    row["unverified_script_agreement"] = {key: probe[key] for key in ("wer", "cer")}
                rows.append(row)
            with (output / f"asr-{language}.json").open("x", encoding="utf-8") as report:
                json.dump({"scope": "offline_android_asr_replay", "rows": raw["rows"]}, report, ensure_ascii=False, indent=2)
            print(language, "completed clips:", len(selected), flush=True)
        finally:
            # Only our unique test directory/staging files; never touch recordings/history.
            subprocess.run([*adb, "shell", "run-as", PACKAGE, "rm", "-r", remote], check=False)
            for stage in staged_names:
                subprocess.run([*adb, "shell", "rm", "-f", stage], check=False)
            subprocess.run([*adb, "shell", "run-as", PACKAGE, "rm", "-f", f"files/evaluation/{name}.json"], check=False)
    summary = summarize(rows)
    summary_document = {"scope": "offline_android_asr_replay", "pilot": args.limit is not None,
                   "limitations": ["Not microphone/call capture, not real-time/asynchronous pipeline accuracy",
                       "Synthetic generation scripts unverified: script-agreement is not verified WER",
                       "Private phone recordings have no references; English is only a decoding probe"],
                   "metrics": summary}
    serialized = json.dumps(summary_document, ensure_ascii=False, indent=2, allow_nan=False)
    with (output / "summary.json").open("x", encoding="utf-8") as report:
        report.write(serialized)
    print(json.dumps(summary, indent=2), flush=True)
    print("Local results:", output, flush=True)


if __name__ == "__main__":
    main()
