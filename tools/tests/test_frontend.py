"""Run dependency-free browser-flow regressions when Node.js is available."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_frontend_regressions():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for frontend regression tests")
    script = Path(__file__).with_name("frontend_regressions.cjs")
    result = subprocess.run([node, "--test", str(script)], capture_output=True, text=True, timeout=30, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
