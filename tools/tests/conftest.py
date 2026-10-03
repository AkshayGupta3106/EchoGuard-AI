"""pytest setup: default to this checkout, regardless of its directory name."""
import os
from pathlib import Path

import pytest
from echoguard.paths import REPO_ROOT

REPO = Path(os.environ.get("ECHOGUARD_REPO", REPO_ROOT)).resolve()
KIT = REPO_ROOT

@pytest.fixture(scope="session")
def repo():
    if not REPO.exists():
        pytest.exit(f"Set ECHOGUARD_REPO to your EchoGuard-AI clone (looked in {REPO})")
    return REPO


@pytest.fixture(scope="session")
def kit():
    return KIT
