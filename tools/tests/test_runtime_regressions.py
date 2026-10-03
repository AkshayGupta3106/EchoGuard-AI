"""Regression coverage for API safety, supervisor wording, and evaluator failures."""
import asyncio
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from echoguard.fusion.fusion_engine import FusionEngine, StreamSignal
from echoguard.fusion.supervisor_agent import SupervisorAgent
from echoguard.paths import check_dataset_names


class Classifier:
    def scam_score(self, text):
        return SimpleNamespace(score=0.6 if "second" in text else 0.4, explain=lambda: "same signals")


@pytest.fixture
def api(monkeypatch):
    from echoguard.web import application as web
    monkeypatch.setattr(web, "_sessions", {})
    monkeypatch.setattr(web, "ScamClassifier", Classifier)
    with TestClient(web.app) as client:
        yield web, client


def start(client):
    response = client.post("/api/live/start")
    assert response.status_code == 200
    return response.json()["session_id"]


def wav(samples, rate=16000, subtype="PCM_16"):
    import soundfile as sf
    output = io.BytesIO()
    sf.write(output, samples, rate, format="WAV", subtype=subtype)
    return ("audio.wav", output.getvalue(), "audio/wav")


def test_analyze_returns_current_risk_even_without_new_timeline_entry(api):
    _, client = api
    response = client.post("/api/analyze", data={"transcript": "first. second."}).json()
    assert response["risk_score"] == pytest.approx(0.6)
    assert response["action"] == "warn"
    assert response["reasoning_log"][-1]["risk_score"] == 40


@pytest.mark.parametrize("route", ["turn", "audio-chunk", "end"])
def test_expired_session_is_not_revived(api, route):
    web, client = api
    sid = start(client)
    web._sessions[sid].last_active -= web.SESSION_TTL_SECONDS + 1
    kwargs = {"data": {"session_id": sid, "text": "hi"}}
    if route == "audio-chunk":
        kwargs["files"] = {"audio": wav(np.zeros(512))}
    result = client.post("/api/live/" + route, **kwargs)
    if route == "end":
        assert result.json() == {"ended": False}
    else:
        assert result.status_code == 404
    assert sid not in web._sessions


def test_turn_retry_is_idempotent_and_conflicting_reuse_rejected(api):
    web, client = api
    sid = start(client)
    turn = {"session_id": sid, "text": "first", "turn_id": "turn-1"}
    first = client.post("/api/live/turn", data=turn).json()
    assert client.post("/api/live/turn", data=turn).json() == first
    assert web._sessions[sid].running_transcript == " first"
    assert web._sessions[sid].turn_count == 1
    assert client.post("/api/live/turn", data={**turn, "text": "second"}).status_code == 409


def test_failed_inference_does_not_duplicate_text_on_retry(api, monkeypatch):
    web, client = api
    sid = start(client)
    session = web._sessions[sid]
    original = session.scam_classifier.scam_score

    def fail(text):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(session.scam_classifier, "scam_score", fail)
    turn = {"session_id": sid, "text": "first", "turn_id": "one"}
    with pytest.raises(RuntimeError, match="inference failed"):
        client.post("/api/live/turn", data=turn)
    assert session.running_transcript == "" and session.turn_count == 0
    monkeypatch.setattr(session.scam_classifier, "scam_score", original)
    assert client.post("/api/live/turn", data=turn).status_code == 200
    assert session.running_transcript == " first" and session.turn_count == 1


@pytest.mark.parametrize("turn_id", ["bad id", "x" * 65, "<script>"])
def test_invalid_turn_id(api, turn_id):
    _, client = api
    sid = start(client)
    assert client.post("/api/live/turn", data={"session_id": sid, "text": "hi", "turn_id": turn_id}).status_code == 400


def test_transcript_limits_and_no_mutation_on_rejection(api, monkeypatch):
    web, client = api
    monkeypatch.setattr(web, "MAX_TRANSCRIPT_CHARS", 12)
    assert client.post("/api/analyze", data={"transcript": "x" * 13}).status_code == 413
    sid = start(client)
    client.post("/api/live/turn", data={"session_id": sid, "text": "first"})
    previous = web._sessions[sid].running_transcript
    assert client.post("/api/live/turn", data={"session_id": sid, "text": "second"}).status_code == 413
    assert web._sessions[sid].running_transcript == previous


def test_session_count_and_update_limits(api, monkeypatch):
    web, client = api
    monkeypatch.setattr(web, "MAX_LIVE_SESSIONS", 1)
    monkeypatch.setattr(web, "MAX_SESSION_UPDATES", 1)
    sid = start(client)
    assert client.post("/api/live/start").status_code == 429
    turn = {"session_id": sid, "text": "first", "turn_id": "one"}
    assert client.post("/api/live/turn", data=turn).status_code == 200
    assert client.post("/api/live/turn", data=turn).status_code == 200
    assert client.post("/api/live/turn", data={**turn, "turn_id": "two"}).status_code == 413
    client.post("/api/live/end", data={"session_id": sid})
    start(client)


def test_active_session_not_pruned_and_updates_serialized(api):
    web, client = api
    sid = start(client)
    entered, release, second_entered = threading.Event(), threading.Event(), threading.Event()

    def first():
        with web._locked_session(sid) as session:
            session.last_active -= web.SESSION_TTL_SECONDS + 1
            entered.set()
            assert release.wait(5)
            session.revision += 1

    def second():
        with web._locked_session(sid) as session:
            second_entered.set()
            assert session.revision == 1

    with ThreadPoolExecutor(2) as executor:
        one = executor.submit(first)
        assert entered.wait(5)
        web._prune_sessions()
        assert sid in web._sessions
        two = executor.submit(second)
        try:
            assert not second_entered.wait(0.05)
        finally:
            release.set()
        one.result(timeout=5)
        two.result(timeout=5)


@pytest.mark.parametrize("kind", ["invalid", "empty", "long", "nan", "too_large"])
def test_audio_validation(api, monkeypatch, kind):
    web, client = api
    if kind == "invalid":
        audio = ("bad.wav", b"not audio", "audio/wav")
    elif kind == "empty":
        audio = wav(np.zeros(0))
    elif kind == "long":
        monkeypatch.setattr(web, "MAX_AUDIO_SECONDS", 0.01)
        audio = wav(np.zeros(512))
    elif kind == "nan":
        audio = wav(np.array([float("nan")]), subtype="FLOAT")
    else:
        monkeypatch.setattr(web, "MAX_AUDIO_BYTES", 10)
        audio = wav(np.zeros(512))
    result = client.post("/api/analyze", data={"transcript": "hi"}, files={"audio": audio})
    assert result.status_code == (413 if kind == "too_large" else 400)


def test_raw_request_limit_with_and_without_content_length(api, monkeypatch):
    web, client = api
    monkeypatch.setattr(web, "MAX_REQUEST_BYTES", 10)
    assert client.post("/api/analyze", content=b"x" * 11).status_code == 413

    async def chunked_request():
        chunks = iter([b"x" * 6, b"x" * 6])
        sent = []

        async def receive():
            return {"type": "http.request", "body": next(chunks), "more_body": True}

        async def send(message):
            sent.append(message)

        async def downstream(*args):
            pytest.fail("Oversized request reached downstream parsing")

        await web.RequestSizeLimit(downstream)({"type": "http", "method": "POST", "headers": []}, receive, send)
        assert sent[0]["status"] == 413

    asyncio.run(chunked_request())


@pytest.mark.parametrize("before,after,direction", [(0.1, 0.9, "up"), (0.1, 0.5, "up"), (0.9, 0.1, "down"), (0.9, 0.5, "down")])
def test_supervisor_uses_risk_order_not_alphabetical_order(before, after, direction):
    agent, fusion = SupervisorAgent(), FusionEngine()
    agent.update(fusion.combine(StreamSignal(0), StreamSignal(before)))
    entry = agent.update(fusion.combine(StreamSignal(0), StreamSignal(after)))
    assert f", {direction} from " in entry.reasoning


@pytest.mark.parametrize("paths", [["a/calls.json", "b/calls.json"], ["a/Calls.json", "b/calls.JSON"]])
def test_duplicate_dataset_basenames_rejected(paths):
    with pytest.raises(ValueError, match="basenames must be unique"):
        check_dataset_names(paths)
    check_dataset_names(["a/dev.json", "b/heldout.json"])


def test_all_failed_http_evaluation_saves_diagnostics(tmp_path, monkeypatch, capsys):
    from echoguard.evaluation import run_eval_e2e as evaluation
    dataset = tmp_path / "calls.json"
    dataset.write_text(json.dumps([{"id": "failed-call", "label": "scam", "language": "en", "turns": [{"text": "hi"}]}]), encoding="utf-8")
    output = tmp_path / "diagnostics.json"
    monkeypatch.setattr("sys.argv", ["eval", "--url", "http://example.invalid", "--dataset", str(dataset), "--canary-from", str(dataset), "--out", str(output)])
    monkeypatch.setattr(evaluation, "wait_until_up", lambda base: 0)
    monkeypatch.setattr(evaluation, "measure_rtt", lambda base: None)

    def failed_call(*args):
        raise RuntimeError("simulated request failure")

    monkeypatch.setattr(evaluation, "run_call", failed_call)
    with pytest.raises(SystemExit, match="saved partial diagnostics"):
        evaluation.main()
    def invalid_constant(value):
        pytest.fail(f"Non-standard JSON constant: {value}")

    report = json.loads(output.read_text(encoding="utf-8"), parse_constant=invalid_constant)
    assert report["evaluation_complete"] is False
    assert report["datasets"]["calls.json"]["n_completed"] == 0
    assert report["datasets"]["calls.json"]["errors"][0]["id"] == "failed-call"
    assert "p50=n/a" in capsys.readouterr().out
