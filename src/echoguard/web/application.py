"""FastAPI acoustic/text scoring, process-local sessions, and browser frontend.

Audio inference is server-side; text is supplied, not transcribed by this API.
Uvicorn entry point: webdemo/app.py. Setup: docs/guide.md.
"""
from __future__ import annotations

import re
import tempfile
import time
import uuid
from pathlib import Path
from threading import Lock
from contextlib import contextmanager
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from echoguard.paths import REPO_ROOT

# Use ONNX acoustic inference to keep PyTorch out of the web environment.
from echoguard.acoustic.spoof_detector_onnx import SpoofDetector, RollingSpoofScorer
from echoguard.semantic.scam_classifier import ScamClassifier
from echoguard.fusion.fusion_engine import FusionEngine, StreamSignal
from echoguard.fusion.supervisor_agent import SupervisorAgent
from echoguard.fusion.supervisor_agent import _risk_level, ACTION_FOR_LEVEL

TARGET_SR = 16000
SESSION_TTL_SECONDS = 20 * 60  # Prune abandoned sessions after 20 idle minutes.
MAX_REQUEST_BYTES = 10 * 1024 * 1024
MAX_AUDIO_BYTES = 8 * 1024 * 1024
MAX_AUDIO_SECONDS = 60
MAX_TRANSCRIPT_CHARS = 50000
MAX_LIVE_SESSIONS = 16
MAX_SESSION_TURNS = 1000
MAX_SESSION_UPDATES = 5000

app = FastAPI(title="EchoGuard-AI live demo")


class RequestSizeLimit:
    """Bound the raw request before multipart/form parsing allocates resources."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"error": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
        if declared < 0 or declared > MAX_REQUEST_BYTES:
            return await JSONResponse({"error": "Request is too large"}, status_code=413)(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                return await JSONResponse({"error": "Request is too large"}, status_code=413)(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


app.add_middleware(RequestSizeLimit)


@app.exception_handler(HTTPException)
async def request_error(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail}, headers=exc.headers)

print("[startup] loading AASIST-L via ONNX Runtime (this happens once, shared across all sessions)...")
_spoof_detector = SpoofDetector()
_inference_lock = Lock()
print("[startup] ready.")


class LiveSession:
    """Per-session transcript, inference buffers, decision state, and retry bookkeeping."""
    def __init__(self):
        self.scam_classifier = ScamClassifier()
        self.fusion = FusionEngine()
        self.agent = SupervisorAgent()
        self.rolling_spoof = RollingSpoofScorer(detector=_spoof_detector)
        self.running_transcript = ""
        self.last_scam_signal = StreamSignal(score=0.0, explain="no scam signals detected")
        self.last_active = time.monotonic()
        self.busy = False
        self.lock = Lock()
        self.turn_results = {}
        self.turn_count = 0
        self.revision = 0


_sessions: dict[str, LiveSession] = {}
_sessions_lock = Lock()


def _prune_sessions():
    now = time.monotonic()
    with _sessions_lock:
        dead = [sid for sid, s in _sessions.items() if not s.busy and now - s.last_active > SESSION_TTL_SECONDS]
        for sid in dead:
            _sessions.pop(sid, None)


def _get_session(session_id: str) -> Optional[LiveSession]:
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session and not session.busy and time.monotonic() - session.last_active > SESSION_TTL_SECONDS:
            _sessions.pop(session_id, None)
            return None
        return session


@contextmanager
def _locked_session(session_id):
    session = _get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found or expired; start a new session")
    with session.lock:
        with _sessions_lock:
            if _sessions.get(session_id) is not session or time.monotonic() - session.last_active > SESSION_TTL_SECONDS:
                _sessions.pop(session_id, None)
                raise HTTPException(404, "Session not found or expired; start a new session")
            session.last_active = time.monotonic()
            session.busy = True
        try:
            yield session
        finally:
            with _sessions_lock:
                session.last_active = time.monotonic()
                session.busy = False


def _split_turns(transcript: str) -> List[str]:
    """Split pasted text into sentence/newline turns for incremental analysis."""
    parts = re.split(r"(?<=[.!?।])\s+|\n+", transcript.strip())
    return [p.strip() for p in parts if p.strip()]


def _load_and_resample(path: str) -> np.ndarray:
    """Validate decoded audio, downmix stereo, and linearly resample to mono 16 kHz."""
    import soundfile as sf
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("Audio must contain samples")
    if info.duration > MAX_AUDIO_SECONDS or info.channels > 2 or info.samplerate > 96000:
        raise ValueError("Audio must be at most 60 seconds, mono/stereo, and at most 96 kHz")
    samples, sr = sf.read(path, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != TARGET_SR:
        duration = len(samples) / sr
        n_target = int(round(duration * TARGET_SR))
        x_old = np.linspace(0, duration, num=len(samples), endpoint=False)
        x_new = np.linspace(0, duration, num=n_target, endpoint=False)
        samples = np.interp(x_new, x_old, samples).astype(np.float32)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Audio samples must be nonempty and finite")
    return samples


def _read_audio(audio):
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
            size = 0
            while True:
                block = audio.file.read(65536)
                if not block:
                    break
                size += len(block)
                if size > MAX_AUDIO_BYTES:
                    raise HTTPException(413, "Audio upload is too large")
                tmp.write(block)
        return _load_and_resample(path)
    except HTTPException:
        raise
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(400, "Audio could not be decoded or violates the input limits") from exc
    finally:
        audio.file.close()
        if path is not None:
            Path(path).unlink(missing_ok=True)


def _check_transcript(text):
    if len(text) > MAX_TRANSCRIPT_CHARS:
        raise HTTPException(413, "Transcript is too long")


def _check_session_updates(session):
    if session.revision >= MAX_SESSION_UPDATES:
        raise HTTPException(413, "Too many updates in this session; start a new session")


def _action(risk):
    return ACTION_FOR_LEVEL[_risk_level(risk)].value


class AnalyzeResponse(BaseModel):
    risk_score: float
    action: str
    scam_score: float
    scam_explain: str
    spoof_score: Optional[float]
    spoof_used: bool
    reasoning_log: list


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(transcript: str = Form(...), audio: Optional[UploadFile] = File(None)):
    _check_transcript(transcript)
    turns = _split_turns(transcript) or [""]
    if len(turns) > MAX_SESSION_TURNS:
        raise HTTPException(413, "Transcript contains too many turns")
    scam_classifier = ScamClassifier()
    fusion = FusionEngine()
    agent = SupervisorAgent()

    spoof_used = False
    spoof_score = 0.0
    if audio is not None and audio.filename:
        samples = _read_audio(audio)
        with _inference_lock:
            acoustic_result = _spoof_detector.predict_array(samples)
        spoof_score = acoustic_result["spoof_score"]
        spoof_used = True

    spoof_signal = StreamSignal(
        score=spoof_score,
        explain=("likely AI-generated voice" if spoof_score > 0.5 else "no spoof detected")
                if spoof_used else "no audio provided",
    )

    running_transcript = ""
    log = []
    final_scam_result = None
    for turn in turns:
        running_transcript += " " + turn
        final_scam_result = scam_classifier.scam_score(running_transcript)
        scam_signal = StreamSignal(score=final_scam_result.score, explain=final_scam_result.explain())
        fusion_result = fusion.combine(spoof_signal, scam_signal)
        entry = agent.update(fusion_result)
        if entry:
            log.append(entry.to_ui_dict())

    if not log:
        # nothing ever crossed a threshold change from the initial MONITOR state
        log.append({"time": "00:00", "text": "no scam signals detected", "risk_score": 0, "action": "monitor"})

    return AnalyzeResponse(
        risk_score=fusion_result.risk_score,
        action=_action(fusion_result.risk_score),
        scam_score=final_scam_result.score if final_scam_result else 0.0,
        scam_explain=final_scam_result.explain() if final_scam_result else "no scam signals detected",
        spoof_score=spoof_score if spoof_used else None,
        spoof_used=spoof_used,
        reasoning_log=log,
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}


# =============================================================================
# Streaming session endpoints: independent text/audio updates share fusion state.
# =============================================================================

@app.post("/api/live/start")
def live_start():
    _prune_sessions()
    session_id = uuid.uuid4().hex
    with _sessions_lock:
        if len(_sessions) >= MAX_LIVE_SESSIONS:
            raise HTTPException(429, "Too many live sessions; end a session or retry later")
        _sessions[session_id] = LiveSession()
    return {"session_id": session_id}


@app.post("/api/live/turn")
def live_turn(session_id: str = Form(...), text: str = Form(...), turn_id: Optional[str] = Form(None)):
    text = text.strip()
    if turn_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", turn_id):
        raise HTTPException(400, "Invalid turn ID")
    with _locked_session(session_id) as session:
        if not text:
            return {"skipped": True}
        if turn_id in session.turn_results:
            previous_text, result = session.turn_results[turn_id]
            if text != previous_text:
                raise HTTPException(409, "Turn ID was already used with different text")
            return result
        _check_session_updates(session)
        _check_transcript(session.running_transcript + " " + text)
        if session.turn_count >= MAX_SESSION_TURNS:
            raise HTTPException(413, "Too many turns in this session")
        transcript = session.running_transcript + " " + text
        scam_result = session.scam_classifier.scam_score(transcript)
        scam_signal = StreamSignal(score=scam_result.score, explain=scam_result.explain())
        with _inference_lock:
            spoof_reading = session.rolling_spoof.score()
        spoof_signal = StreamSignal(
            score=spoof_reading["spoof_score"],
            explain=("likely AI-generated voice" if spoof_reading["spoof_score"] > 0.5 else "no spoof detected")
                    if "note" not in spoof_reading else "no audio yet",
        )
        fusion_result = session.fusion.combine(spoof_signal, scam_signal)
        entry = session.agent.update(fusion_result)
        session.running_transcript = transcript
        session.turn_count += 1
        session.last_scam_signal = scam_signal
        session.revision += 1
        result = {
            "risk_score": fusion_result.risk_score, "action": _action(fusion_result.risk_score),
            "revision": session.revision,
            "scam_score": scam_result.score, "spoof_score": spoof_reading["spoof_score"],
            "new_entry": entry.to_ui_dict() if entry else None,
        }
        if turn_id is not None:
            session.turn_results[turn_id] = (text, result)
        return result


@app.post("/api/live/audio-chunk")
def live_audio_chunk(session_id: str = Form(...), audio: UploadFile = File(...)):
    with _locked_session(session_id) as session:
        _check_session_updates(session)
        samples = _read_audio(audio)
        session.rolling_spoof.push(samples)
        with _inference_lock:
            spoof_reading = session.rolling_spoof.score()
        spoof_signal = StreamSignal(
            score=spoof_reading["spoof_score"],
            explain="likely AI-generated voice" if spoof_reading["spoof_score"] > 0.5 else "no spoof detected",
        )
        fusion_result = session.fusion.combine(spoof_signal, session.last_scam_signal)
        entry = session.agent.update(fusion_result)
        session.revision += 1
        return {
            "risk_score": fusion_result.risk_score, "action": _action(fusion_result.risk_score),
            "revision": session.revision,
            "scam_score": session.last_scam_signal.score,
            "spoof_score": spoof_reading["spoof_score"],
            "inference_ms": spoof_reading.get("inference_ms"),
            "new_entry": entry.to_ui_dict() if entry else None,
        }


@app.post("/api/live/end")
def live_end(session_id: str = Form(...)):
    session = _get_session(session_id)
    if session is None:
        return {"ended": False}
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
    if session is None:
        return {"ended": False}
    with session.lock:
        return {"ended": True, "final_log": [e.to_ui_dict() for e in session.agent.timeline]}


static_dir = REPO_ROOT / "webdemo" / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
