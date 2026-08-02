#!/usr/bin/env python3
"""
download_kroko.py
------------------
Downloads the Kroko-128L English streaming ASR model (sherpa-onnx
transducer format: encoder/decoder/joiner + tokens) and places it
in the correct app asset directory.

Usage:
    python download_kroko.py

Requirements:
    pip install huggingface_hub

Repository structure (hudaiapa88/sherpa-stt-onnx):
    models/en/kroko_128l/encoder.int8.onnx
    models/en/kroko_128l/decoder.int8.onnx
    models/en/kroko_128l/joiner.int8.onnx
    models/en/kroko_128l/tokens.txt
    models/<lang>/kroko_128l/...   <- other languages available in the
                                       same repo if needed later (de, es,
                                       fr, tr, it, pt, ...)

NOTE: This is a community re-export of Banafo's Kroko ASR project,
converted to be sherpa-onnx compatible. It is NOT bundled or downloaded
automatically by the app or by Gradle - unlike the Indic model, no
official single-repo HF path exists for Kroko, so this script pins a
known-good source instead of leaving the four files to be copied in by
hand (which is how they went missing after a fresh git clone).

After running:
    Build the Android app in Android Studio - the model will be bundled
    into the APK's assets automatically. The sherpa-onnx AAR library is
    handled separately by download_indicconformer.py; run that too if
    you haven't already.
"""

import sys
import shutil
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
ASSETS_DIR = SCRIPT_DIR / "app" / "app" / "src" / "main" / "assets"
DEST_DIR = ASSETS_DIR / "kroko-128l"
DEST_DIR.mkdir(parents=True, exist_ok=True)

# Use a temp cache outside the assets dir so partial downloads don't pollute it
CACHE_DIR = SCRIPT_DIR / ".hf_cache"
CACHE_DIR.mkdir(exist_ok=True)

REPO_ID = "hudaiapa88/sherpa-stt-onnx"

# Local names must match exactly what KrokoLiveTranscriber.kt expects:
#   kroko-128l/encoder.int8.onnx
#   kroko-128l/decoder.int8.onnx
#   kroko-128l/joiner.int8.onnx
#   kroko-128l/tokens.txt
FILES = [
    ("models/en/kroko_128l/encoder.int8.onnx", "encoder.int8.onnx"),
    ("models/en/kroko_128l/decoder.int8.onnx", "decoder.int8.onnx"),
    ("models/en/kroko_128l/joiner.int8.onnx",  "joiner.int8.onnx"),
    ("models/en/kroko_128l/tokens.txt",        "tokens.txt"),
]

print(f"Downloading Kroko-128L English model from {REPO_ID} ...")
print(f"Destination: {DEST_DIR}\n")

for repo_path, local_name in FILES:
    dest_file = DEST_DIR / local_name
    if dest_file.exists() and dest_file.stat().st_size > 1000:
        print(f"  [skip]  {local_name} already present ({dest_file.stat().st_size // 1_000_000} MB)")
        continue

    print(f"  [download] {repo_path} ...", end="", flush=True)
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=repo_path,
        cache_dir=str(CACHE_DIR),
    )
    shutil.copy2(downloaded, dest_file)
    size_mb = dest_file.stat().st_size // 1_000_000
    print(f" done ({size_mb} MB)")

# Clean up HF cache to save disk space
shutil.rmtree(CACHE_DIR, ignore_errors=True)

print(f"""
Done. Kroko-128L assets successfully prepared.

Next:
  1. If you haven't already, also run download_indicconformer.py
     (handles the Hindi model + the sherpa-onnx AAR library).
  2. Open the Android project in Android Studio (app/)
  3. Click "Sync Project with Gradle Files"
  4. Build & run on your device
  5. English live transcription should now work via Kroko-128L.
""")
