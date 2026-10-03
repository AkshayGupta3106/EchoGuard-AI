"""Opt-in real-container checks: set ECHOGUARD_DOCKER_IMAGE to a built image."""
import io
import json
import os
import subprocess
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests


IMAGE = os.environ.get("ECHOGUARD_DOCKER_IMAGE")
pytestmark = pytest.mark.skipif(not IMAGE, reason="Set ECHOGUARD_DOCKER_IMAGE to run container smoke checks")


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, timeout=120).strip()


@pytest.fixture(scope="module")
def deployed():
    name = "echoguard-test-" + uuid.uuid4().hex[:10]
    container = docker("run", "-d", "--rm", "--name", name, "-p", "127.0.0.1::7860", IMAGE)
    try:
        info = json.loads(docker("inspect", container))[0]
        port = info["NetworkSettings"]["Ports"]["7860/tcp"][0]["HostPort"]
        base = f"http://127.0.0.1:{port}"
        for _ in range(60):
            try:
                if requests.get(base + "/api/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            pytest.fail("Container startup failed: " + docker("logs", container))
        yield base, container
    finally:
        docker("stop", container)


def post(base, route, **kwargs):
    response = requests.post(base + route, timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


def test_deployment_scoring_static_and_dependency_boundaries(deployed):
    base, container = deployed
    assert requests.get(base + "/api/health", timeout=10).json() == {"status": "ok"}
    page = requests.get(base + "/", timeout=10)
    assert page.status_code == 200 and "innerHTML" not in page.text
    for text, expected in [
        ("This is your bank. Share your OTP right now or your account will be blocked.", "block"),
        ("Hello, your parcel arrives tomorrow afternoon.", "monitor"),
        ("Share your OTP. Just kidding, it's a prank.", "monitor"),
    ]:
        result = post(base, "/api/analyze", data={"transcript": text})
        assert result["action"] == expected
    docker("exec", container, "python", "-c",
           "import importlib.util; from pathlib import Path; "
           "assert importlib.util.find_spec('torch') is None; "
           "assert importlib.util.find_spec('sentence_transformers') is None; "
           "assert not any(Path('/app', p).exists() for p in ('tools', 'data', 'artifacts', 'app'))")


def test_deployment_live_idempotency_audio_and_end(deployed):
    base, _ = deployed
    sid = post(base, "/api/live/start")["session_id"]
    try:
        turn = {"session_id": sid, "text": "Please share your OTP right now", "turn_id": "one"}
        first = post(base, "/api/live/turn", data=turn)
        assert first["risk_score"] >= 0.65 and first["revision"] == 1
        assert post(base, "/api/live/turn", data=turn) == first
        assert requests.post(base + "/api/live/turn", data={**turn, "text": "different"}, timeout=10).status_code == 409
        joke = post(base, "/api/live/turn", data={"session_id": sid, "text": "Just kidding, it's a prank.", "turn_id": "two"})
        assert joke["scam_score"] == 0 and joke["revision"] == 2
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\0\0" * 80000)
        reading = post(base, "/api/live/audio-chunk", data={"session_id": sid},
                       files={"audio": ("silence.wav", buffer.getvalue(), "audio/wav")})
        assert 0 <= reading["spoof_score"] <= 1 and reading["revision"] == 3
        assert reading["scam_score"] == 0
    finally:
        post(base, "/api/live/end", data={"session_id": sid})
    assert requests.post(base + "/api/live/turn", data=turn, timeout=10).status_code == 404


def test_deployment_concurrent_retries_commit_once(deployed):
    base, _ = deployed
    sid = post(base, "/api/live/start")["session_id"]
    try:
        turn = {"session_id": sid, "text": "Please share your OTP", "turn_id": "same"}
        with ThreadPoolExecutor(4) as executor:
            results = list(executor.map(lambda _: post(base, "/api/live/turn", data=turn), range(4)))
        assert all(result == results[0] for result in results)
        assert results[0]["revision"] == 1
    finally:
        post(base, "/api/live/end", data={"session_id": sid})


def test_deployment_rejects_oversized_and_invalid_inputs(deployed):
    base, _ = deployed
    assert requests.post(base + "/api/analyze", data={"transcript": "x" * 50001}, timeout=30).status_code == 413
    assert requests.post(base + "/api/analyze", data={"transcript": "hi"},
                         files={"audio": ("bad.wav", b"not audio", "audio/wav")}, timeout=30).status_code == 400
    assert requests.post(base + "/api/analyze", data=b"x" * (10 * 1024 * 1024 + 1), timeout=30).status_code == 413
