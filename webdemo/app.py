"""Uvicorn entry point for echoguard.web.application, run from webdemo/."""
# Expose the checkout source tree when the editable package is not installed.
try:
    import echoguard
except ModuleNotFoundError as exc:
    if exc.name != "echoguard":
        raise
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from echoguard.web.application import *  # noqa: F401,F403
from echoguard.web import application as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
