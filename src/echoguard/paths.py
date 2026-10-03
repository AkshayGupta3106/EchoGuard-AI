"""Checkout-relative asset, data, and evaluation output paths.

Install this checkout with ``pip install -e . --no-deps``.
"""
from pathlib import Path
from datetime import datetime
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
ACOUSTIC_MODEL_DIR = ASSETS_DIR / "acoustic"
SEMANTIC_ASSETS_DIR = ASSETS_DIR / "semantic"
ANDROID_ASSETS_DIR = REPO_ROOT / "app" / "app" / "src" / "main" / "assets"

DATA_DIR = REPO_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
AUDIO_DIR = DATA_DIR / "audio"
SYNTHETIC_AUDIO_DIR = AUDIO_DIR / "synthetic"
BONAFIDE_AUDIO_DIR = AUDIO_DIR / "bonafide"
PHONE_AUDIO_DIR = AUDIO_DIR / "bonafide_phone"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
EVALUATION_ROOT = ARTIFACTS_DIR / "evaluation"
# Protected local report directory.
EVALUATION_RESULTS_DIR = EVALUATION_ROOT / "python" / "python-20260921-161000-6e751cd9"

DATASET_FILENAMES = ("dataset.json", "test_heldout.json", "test_heldout2.json")


def new_evaluation_run(kind: str) -> Path:
    """Create a unique output path in the target's evaluation directory."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    targets = {"python": "python", "web": "webdemo", "android-device": "android-device"}
    return EVALUATION_ROOT / targets[kind] / f"{kind}-{stamp}-{uuid4().hex[:8]}"


def check_evaluation_output(directory, filenames):
    """Reject existing report files and protected output locations."""
    directory = Path(directory)
    resolved = directory.resolve()
    if resolved == EVALUATION_ROOT.resolve() or resolved.is_relative_to(EVALUATION_RESULTS_DIR.resolve()):
        raise ValueError(f"{EVALUATION_RESULTS_DIR} contains protected results; choose a new --outdir/--out")
    existing = [str(directory / name) for name in filenames if (directory / name).exists()]
    if existing:
        raise ValueError(f"Refusing to overwrite existing evaluation outputs: {existing}")


def check_dataset_names(paths):
    """Reject duplicate output basenames, including case-only filesystem aliases."""
    stems = [Path(value).stem.casefold() for value in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Dataset basenames must be unique within a run; rename the inputs or run them separately")


def _compat_relative_path(path: Path) -> Path:
    """Recognize checkout-root paths without redirecting external absolute paths."""
    if path.is_absolute():
        try:
            return path.relative_to(REPO_ROOT)
        except ValueError:
            return path
    # Resolve relative paths such as ../EchoGuard-AI/audio when they point here.
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def resolve_data_path(value) -> Path:
    """Resolve supported data aliases; existing custom paths always take precedence."""
    path = Path(value).expanduser()
    if path.exists():
        return path
    aliases = {Path("audio"): SYNTHETIC_AUDIO_DIR,
               Path("bonafide"): BONAFIDE_AUDIO_DIR,
               Path("bonafide_phone"): PHONE_AUDIO_DIR}
    aliases.update({Path(name): DATASETS_DIR / name for name in DATASET_FILENAMES})
    return aliases.get(_compat_relative_path(path), path)


def resolve_artifact_path(value) -> Path:
    """Resolve results/... shorthand while preserving existing custom paths."""
    path = Path(value).expanduser()
    if path.exists():
        return path
    relative = _compat_relative_path(path)
    if not relative.is_absolute():
        parts = relative.parts
        if parts and parts[0] == "results":
            return EVALUATION_RESULTS_DIR.joinpath(*parts[1:])
    return path


def normalize_evaluation_paths(args):
    """Normalize only path arguments, never dataset membership or scoring options."""
    for name in ("dev", "canary_from", "audio_dir"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, str(resolve_data_path(value)))
    for name in ("heldout", "dataset"):
        values = getattr(args, name, None)
        if values is not None:
            setattr(args, name, [str(resolve_data_path(value)) for value in values])
    value = getattr(args, "bonafide_dir", None)
    if value is not None:
        args.bonafide_dir = ([str(resolve_data_path(item)) for item in value]
                             if isinstance(value, list) else str(resolve_data_path(value)))
    for name in ("outdir", "out"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, str(resolve_artifact_path(value)))
    return args
