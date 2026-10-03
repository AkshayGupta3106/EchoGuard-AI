#!/usr/bin/env python3
"""Download Hindi IndicConformer assets and sherpa-onnx into the Android project.

Run: python tools/models/download_indicconformer.py
Requires huggingface_hub. Model: hi/model.int8.onnx; shared tokens: tokens.txt.
"""

import sys
import shutil
import urllib.request

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

from echoguard.paths import REPO_ROOT

SCRIPT_DIR = REPO_ROOT
ASSETS_DIR = SCRIPT_DIR / "app" / "app" / "src" / "main" / "assets"
DEST_DIR = ASSETS_DIR / "indicconformer-hi"

# Use a temp cache outside the assets dir so partial downloads don't pollute it
CACHE_DIR = SCRIPT_DIR / ".hf_cache"

REPO_ID = "parismitaglobalsolutions/indicconformer-sherpa-onnx"

# The repository shares root-level tokens across language models.
FILES = [
    ("hi/model.int8.onnx", "model.int8.onnx"),
    ("tokens.txt",         "tokens.txt"),
]

# ---------------------------------------------------------------------------
# Download sherpa-onnx AAR (since it's ignored in .gitignore)
# ---------------------------------------------------------------------------
AAR_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.4/sherpa-onnx-1.13.4.aar"
LIBS_DIR = SCRIPT_DIR / "app" / "app" / "libs"
AAR_DEST = LIBS_DIR / "sherpa-onnx-1.13.4.aar"

def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    print(f"Downloading IndicConformer Hindi model from {REPO_ID} ...")
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

    shutil.rmtree(CACHE_DIR, ignore_errors=True)

    print(f"\nChecking for sherpa-onnx AAR library in {LIBS_DIR} ...")
    if not AAR_DEST.exists() or AAR_DEST.stat().st_size < 1000:
        print(f"  [download] {AAR_URL} ...", end="", flush=True)
        LIBS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(AAR_URL, AAR_DEST)
        size_mb = AAR_DEST.stat().st_size // 1_000_000
        print(f" done ({size_mb} MB)")
    else:
        print(f"  [skip] sherpa-onnx-1.13.4.aar already present")

    print(f"""
Done. Assets and libraries successfully prepared.

Next:
  1. Open the Android project in Android Studio (app/)
  2. Click "Sync Project with Gradle Files"
  3. Build & run on your device
   4. Select Hindi, then tap "Start microphone".
""")


if __name__ == "__main__":
    main()
