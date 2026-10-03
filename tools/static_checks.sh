#!/usr/bin/env bash
# Activate the project environment first.
# Optional analysis tools are not installed automatically.
set -eu
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
python "$SCRIPT_DIR/check_source.py" "$@" --analysis
