"""Check first-party source without importing models or scanning local environments.

Optional analysis tools must be installed explicitly; this command never invokes
pip, downloads assets, or executes an accuracy benchmark.
"""
import argparse
import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

from echoguard.paths import REPO_ROOT


SOURCE_DIRS = ("src", "tools", "webdemo")
EXCLUDED_PARTS = {"__pycache__", ".venv", "venv", "env", ".git", "build", "_build"}


def layout_errors(root, files):
    """Validate internal import targets without importing optional ML backends."""
    errors = [f"Root executable should be grouped under tools: {p.name}"
              for p in root.glob("*.py")]
    obsolete = {"scam_classifier", "fusion_engine", "supervisor_agent", "spoof_detector",
                "spoof_detector_onnx", "vad_gate", "evalstats", "phone_channel", "vad_fixed"}
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.level == 0
                       else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                if not module:
                    continue
                if module in obsolete:
                    errors.append(f"{path.relative_to(root)}:{node.lineno}: obsolete flat import {module}")
                if module == "echoguard" or module.startswith("echoguard."):
                    target = root / "src" / Path(*module.split("."))
                    if not target.with_suffix(".py").is_file() and not (target / "__init__.py").is_file():
                        errors.append(f"{path.relative_to(root)}:{node.lineno}: missing package module {module}")
    return errors


def source_files(root):
    files = set(root.glob("*.py"))
    for directory in SOURCE_DIRS:
        files.update((root / directory).rglob("*.py"))
    return sorted(path for path in files
                  if not EXCLUDED_PARTS.intersection(path.relative_to(root).parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?", type=Path, default=REPO_ROOT)
    parser.add_argument("--analysis", action="store_true", help="Also run installed pyflakes/bandit/radon/vulture")
    parser.add_argument("--layout", action="store_true", help="Check root organization and internal package import targets")
    args = parser.parse_args()
    root = args.repo.resolve()
    files = source_files(root)
    if not files:
        parser.error(f"No first-party Python source found in {root}")
    errors = []
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(root)}: {exc}")
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Syntax checked {len(files)} first-party Python files; {len(errors)} errors.")
    if args.layout and not errors:
        layout = layout_errors(root, files)
        for error in layout:
            print(error, file=sys.stderr)
        errors.extend(layout)
        print(f"Layout/internal-import checks: {len(layout)} errors.")
    status = int(bool(errors))
    if args.analysis:
        names = [str(path) for path in files]
        commands = {
            "pyflakes": names,
            "bandit": ["-q", *names],
            "radon": ["cc", "-s", "-n", "C", *names],
            "vulture": [*names, "--min-confidence", "80"],
        }
        for tool, command in commands.items():
            if importlib.util.find_spec(tool) is None:
                print(f"{tool} is not installed; install it explicitly to run this optional check.", file=sys.stderr)
                status = 1
                continue
            print(f"Running {tool}...", flush=True)
            result = subprocess.run([sys.executable, "-m", tool, *command], cwd=root)
            status = max(status, int(result.returncode != 0))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
