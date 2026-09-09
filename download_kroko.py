"""
download_kroko.py
-----------------
Downloads the Kroko-128L English streaming ASR model from Hugging Face
and places the required sherpa-onnx model files into:

    app/app/src/main/assets/kroko-128l/

The Android app expects:

    kroko-128l/
        encoder.int8.onnx
        decoder.int8.onnx
        joiner.int8.onnx
        tokens.txt

Hugging Face repository:
    hudaiapa88/sherpa-stt-onnx

IMPORTANT:
The actual repository paths are:

    en/kroko_128l/encoder.int8.onnx
    en/kroko_128l/decoder.int8.onnx
    en/kroko_128l/joiner.int8.onnx
    en/kroko_128l/tokens.txt

Do NOT use models/en/... as the hf_hub_download() filename.

Usage:
    python download_kroko.py

Dependency:
    python -m pip install "huggingface_hub<1.0,>=0.34.0"
"""

from pathlib import Path
import shutil
import sys


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print()
    print("ERROR: huggingface_hub is not installed.")
    print()
    print("Install it with:")
    print('  python -m pip install "huggingface_hub<1.0,>=0.34.0"')
    print()
    sys.exit(1)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# EchoGuard-AI/
# ├── download_kroko.py
# └── app/
#     └── app/
#         └── src/
#             └── main/
#                 └── assets/

ASSETS_DIR = (
    SCRIPT_DIR
    / "app"
    / "app"
    / "src"
    / "main"
    / "assets"
)

DEST_DIR = ASSETS_DIR / "kroko-128l"

# Keep Hugging Face's temporary/cache files outside the Android assets.
CACHE_DIR = SCRIPT_DIR / ".hf_cache"


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

REPO_ID = "hudaiapa88/sherpa-stt-onnx"


# IMPORTANT:
# These are the ACTUAL paths in the Hugging Face repository.
#
# They are NOT:
#   models/en/kroko_128l/...
#
FILES = [
    (
        "en/kroko_128l/encoder.int8.onnx",
        "encoder.int8.onnx",
    ),
    (
        "en/kroko_128l/decoder.int8.onnx",
        "decoder.int8.onnx",
    ),
    (
        "en/kroko_128l/joiner.int8.onnx",
        "joiner.int8.onnx",
    ),
    (
        "en/kroko_128l/tokens.txt",
        "tokens.txt",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_valid_file(path: Path) -> bool:
    """
    Return True if the file exists and is non-empty.

    ONNX model files should be large. tokens.txt is small, so we only
    require the file to have non-zero size here.
    """
    return path.exists() and path.is_file() and path.stat().st_size > 0


def format_size(size_bytes: int) -> str:
    """Return a human-readable file size."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"

    return f"{size_bytes} bytes"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("EchoGuard-AI — Kroko-128L English ASR downloader")
    print("=" * 72)
    print()
    print(f"Hugging Face repository:")
    print(f"  {REPO_ID}")
    print()
    print(f"Android assets destination:")
    print(f"  {DEST_DIR}")
    print()

    # Create directories.
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_count = 0
    skipped_count = 0

    for repo_path, local_name in FILES:
        destination = DEST_DIR / local_name

        print(f"[{local_name}]")
        print(f"  Repository: {repo_path}")

        # Don't download an already valid file again.
        if is_valid_file(destination):
            size = destination.stat().st_size
            print(f"  [skip]     already exists ({format_size(size)})")
            skipped_count += 1
            print()
            continue

        try:
            print("  [download] downloading from Hugging Face...")

            cached_file = hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_path,
                cache_dir=str(CACHE_DIR),
            )

            cached_path = Path(cached_file)

            if not is_valid_file(cached_path):
                raise RuntimeError(
                    f"Hugging Face returned an invalid file: {cached_path}"
                )

            # Copy the downloaded file into Android assets.
            shutil.copy2(cached_path, destination)

            if not is_valid_file(destination):
                raise RuntimeError(
                    f"Downloaded file was not copied correctly: {destination}"
                )

            size = destination.stat().st_size

            print(f"  [done]     {destination.name} ({format_size(size)})")
            downloaded_count += 1

        except Exception as exc:
            print()
            print("  [ERROR] Download failed.")
            print(f"  {type(exc).__name__}: {exc}")
            print()
            print("  Repository:")
            print(f"    https://huggingface.co/{REPO_ID}")
            print()
            print("  Requested file:")
            print(f"    {repo_path}")
            print()
            print("  No further files will be downloaded.")
            print()
            return 1

        print()

    # -----------------------------------------------------------------------
    # Verify everything
    # -----------------------------------------------------------------------

    print("-" * 72)
    print("Verifying Kroko-128L assets...")
    print("-" * 72)

    all_valid = True

    for _, local_name in FILES:
        path = DEST_DIR / local_name

        if is_valid_file(path):
            print(f"[OK] {local_name:24s} {format_size(path.stat().st_size)}")
        else:
            print(f"[FAIL] {local_name}")
            all_valid = False

    print()

    if not all_valid:
        print("ERROR: One or more Kroko files are missing or invalid.")
        return 1

    # -----------------------------------------------------------------------
    # Success
    # -----------------------------------------------------------------------

    print("=" * 72)
    print("Kroko-128L assets successfully prepared.")
    print("=" * 72)
    print()
    print(f"Downloaded: {downloaded_count}")
    print(f"Skipped:    {skipped_count}")
    print()
    print("Assets are located at:")
    print(f"  {DEST_DIR}")
    print()
    print("Expected files:")

    for _, local_name in FILES:
        print(f"  {DEST_DIR / local_name}")

    print()
    print("Next steps:")
    print()
    print("  1. Make sure IndicConformer assets are also present.")
    print()
    print("  2. Open the Android project in Android Studio.")
    print()
    print("  3. Sync Project with Gradle Files.")
    print()
    print("  4. Build the Android app.")
    print()
    print("  5. Run it on the Android device.")
    print()
    print("  6. Start Live Protection Demo.")
    print()
    print("English streaming ASR will use Kroko-128L.")
    print()

    # The cache is only temporary for this downloader.
    # The actual model files have already been copied to assets.
    try:
        shutil.rmtree(CACHE_DIR)
        print("Temporary Hugging Face cache removed.")
    except Exception as exc:
        print(
            f"Warning: could not remove temporary cache: {exc}"
        )

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
