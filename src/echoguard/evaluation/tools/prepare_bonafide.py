"""Convert consented recordings to numbered 16 kHz mono WAV files with FFmpeg.

Writes a filename/duration/source-hash manifest. Use a fresh destination: outputs
are overwritten. Numbered filenames do not anonymize voices or spoken data;
keep recordings and manifests private. FFmpeg must be on PATH or supplied via --ffmpeg.
Run: python tools/evaluation/prepare_bonafide.py --help
"""
import argparse
import csv
import hashlib
import subprocess
import wave
from pathlib import Path
from echoguard.paths import PHONE_AUDIO_DIR, resolve_data_path

EXT = {".m4a", ".mp3", ".amr", ".ogg", ".opus", ".wav", ".aac", ".flac", ".3gp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default=str(PHONE_AUDIO_DIR))
    ap.add_argument("--ffmpeg", default="ffmpeg",
                     help="path to ffmpeg(.exe) if it's not on PATH, e.g. C:/ffmpeg/bin/ffmpeg.exe")
    a = ap.parse_args()
    src, dst = Path(a.src), resolve_data_path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run([a.ffmpeg, "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        raise SystemExit(
            f"\nCould not find/run '{a.ffmpeg}'.\n"
            f"Either install ffmpeg and make sure it's on PATH (winget install ffmpeg, "
            f"then open a new terminal), or pass its exact location:\n"
            f"    python tools/evaluation/prepare_bonafide.py --src ... --dst ... "
            f"--ffmpeg \"C:/path/to/ffmpeg.exe\"\n"
        )

    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXT)
    if not files:
        raise SystemExit(f"No audio files found under {src} (looked for {sorted(EXT)}).")
    rows = []
    for i, f in enumerate(files, 1):
        out = dst / f"phone_{i:03d}.wav"
        r = subprocess.run([a.ffmpeg, "-loglevel", "error", "-y", "-i", str(f), "-ar", "16000", "-ac", "1", str(out)])
        if r.returncode != 0:
            print("skip (ffmpeg failed): item", i, "-", f.name)
            continue
        with wave.open(str(out)) as w:
            dur = w.getnframes() / w.getframerate()
        rows.append([out.name, round(dur, 1), hashlib.sha256(f.read_bytes()).hexdigest()[:16]])
    with open(dst / "manifest.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["file", "seconds", "source_sha256_16"])
        wr.writerows(rows)
    print(f"wrote {len(rows)} files to {dst}")


if __name__ == "__main__":
    main()
