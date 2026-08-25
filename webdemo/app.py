"""
webdemo/app.py
--------------
Live demo API for EchoGuard-AI. This does NOT reimplement any detection
logic -- it imports your real acoustic/semantic/fusion/agent modules
unchanged and exposes them over HTTP + a small static frontend, so the
actual pipeline is reachable from a public URL.

Expected to live at <repo_root>/webdemo/app.py, as a sibling of
acoustic/, semantic/, fusion/ (it adds those to sys.path at import time,
same pattern demo_pipeline.py already uses).

Run locally:
    cd webdemo && uvicorn app:app --reload --port 7860

Docker / deployment: see webdemo/Dockerfile.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("acoustic", "semantic", "fusion"):
    sys.path.insert(0, str(REPO_ROOT / sub))

# Web-service-specific: use the ONNX Runtime spoof detector, not the torch
# one. `import torch` alone uses ~490MB RSS - nearly the whole 512MB
# free-tier RAM budget before FastAPI/uvicorn/a single request even runs.
# `import onnxruntime` uses ~40MB for numerically identical output (see
# acoustic/spoof_detector_onnx.py for the verification). That memory gap was
# causing OOM crash-restart loops under normal use.
from spoof_detector_onnx import SpoofDetector, RollingSpoofScorer  # noqa: E402
from scam_classifier import ScamClassifier                     # noqa: E402
from fusion_engine import FusionEngine, StreamSignal            # noqa: E402
from supervisor_agent import SupervisorAgent                    # noqa: E402

TARGET_SR = 16000
SESSION_TTL_SECONDS = 20 * 60  # abandoned live-call sessions get pruned after 20 idle minutes

app = FastAPI(title="EchoGuard-AI live demo")

print("[startup] loading AASIST-L via ONNX Runtime (this happens once, shared across all sessions)...")
_spoof_detector = SpoofDetector()
print("[startup] ready.")


class LiveSession:
    """One in-progress 'live call' -- mirrors what the Android app holds per call:
    its own running transcript, its own ScamClassifier/FusionEngine/SupervisorAgent,
    and a RollingSpoofScorer accumulating whatever audio chunks have arrived so far."""
    def __init__(self):
        self.scam_classifier = ScamClassifier()
        self.fusion = FusionEngine()
        self.agent = SupervisorAgent()
        self.rolling_spoof = RollingSpoofScorer(detector=_spoof_detector)
        self.running_transcript = ""
        self.last_scam_signal = StreamSignal(score=0.0, explain="no scam signals detected")
        self.last_active = time.time()


_sessions: dict[str, LiveSession] = {}
_sessions_lock = Lock()


def _prune_sessions():
    now = time.time()
    with _sessions_lock:
        dead = [sid for sid, s in _sessions.items() if now - s.last_active > SESSION_TTL_SECONDS]
        for sid in dead:
            _sessions.pop(sid, None)


def _get_session(session_id: str) -> Optional[LiveSession]:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session:
        session.last_active = time.time()
    return session


def _split_turns(transcript: str) -> List[str]:
    """Break a pasted transcript into rough turns so the agent reasons over
    it incrementally (Observe -> Reason -> Explain -> Recommend -> Act per
    turn), the same way it would over a live call instead of one lump sum."""
    parts = re.split(r"(?<=[.!?।])\s+|\n+", transcript.strip())
    return [p.strip() for p in parts if p.strip()]


def _load_and_resample(path: str) -> np.ndarray:
    """predict_file() requires exactly 16kHz mono and raises otherwise --
    browser-recorded audio is usually 44.1/48kHz, so resample here (simple
    linear interpolation; fine for a demo, not meant to be broadcast-grade)."""
    import soundfile as sf
    samples, sr = sf.read(path, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != TARGET_SR:
        duration = len(samples) / sr
        n_target = int(round(duration * TARGET_SR))
        x_old = np.linspace(0, duration, num=len(samples), endpoint=False)
        x_new = np.linspace(0, duration, num=n_target, endpoint=False)
        samples = np.interp(x_new, x_old, samples).astype(np.float32)
    return samples


class AnalyzeResponse(BaseModel):
    risk_score: float
    action: str
    scam_score: float
    scam_explain: str
    spoof_score: Optional[float]
    spoof_used: bool
    reasoning_log: list


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(transcript: str = Form(...), audio: Optional[UploadFile] = File(None)):
    scam_classifier = ScamClassifier()
    fusion = FusionEngine()
    agent = SupervisorAgent()

    spoof_used = False
    spoof_score = 0.0
    if audio is not None and audio.filename:
        suffix = Path(audio.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        try:
            samples = _load_and_resample(tmp_path)
            acoustic_result = _spoof_detector.predict_array(samples)
            spoof_score = acoustic_result["spoof_score"]
            spoof_used = True
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"couldn't read audio: {e}"})
        finally:
            os.unlink(tmp_path)

    spoof_signal = StreamSignal(
        score=spoof_score,
        explain=("likely AI-generated voice" if spoof_score > 0.5 else "no spoof detected")
                if spoof_used else "no audio provided",
    )

    running_transcript = ""
    log = []
    final_scam_result = None
    for turn in _split_turns(transcript) or [""]:
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
        risk_score=log[-1]["risk_score"] / 100.0,
        action=log[-1]["action"],
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
# Live-call session endpoints -- mirrors the Android app's actual live loop:
# text turns and audio chunks arrive independently and both drive the SAME
# running FusionEngine + SupervisorAgent for that call.
# =============================================================================

@app.post("/api/live/start")
def live_start():
    _prune_sessions()
    session_id = uuid.uuid4().hex
    with _sessions_lock:
        _sessions[session_id] = LiveSession()
    return {"session_id": session_id}


@app.post("/api/live/turn")
def live_turn(session_id: str = Form(...), text: str = Form(...)):
    session = _get_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "session not found or expired -- start a new live call"})

    text = text.strip()
    if not text:
        return {"skipped": True}

    session.running_transcript += " " + text
    scam_result = session.scam_classifier.scam_score(session.running_transcript)
    session.last_scam_signal = StreamSignal(score=scam_result.score, explain=scam_result.explain())

    spoof_reading = session.rolling_spoof.score()
    spoof_signal = StreamSignal(
        score=spoof_reading["spoof_score"],
        explain=("likely AI-generated voice" if spoof_reading["spoof_score"] > 0.5 else "no spoof detected")
                if "note" not in spoof_reading else "no audio yet",
    )
    fusion_result = session.fusion.combine(spoof_signal, session.last_scam_signal)
    entry = session.agent.update(fusion_result)

    return {
        "risk_score": fusion_result.risk_score,
        "scam_score": scam_result.score,
        "spoof_score": spoof_reading["spoof_score"],
        "new_entry": entry.to_ui_dict() if entry else None,
    }


@app.post("/api/live/audio-chunk")
async def live_audio_chunk(session_id: str = Form(...), audio: UploadFile = File(...)):
    session = _get_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "session not found or expired -- start a new live call"})

    suffix = Path(audio.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        samples = _load_and_resample(tmp_path)
    except Exception as e:
        os.unlink(tmp_path)
        return JSONResponse(status_code=400, content={"error": f"couldn't read audio chunk: {e}"})
    os.unlink(tmp_path)

    session.rolling_spoof.push(samples)
    spoof_reading = session.rolling_spoof.score()
    spoof_signal = StreamSignal(
        score=spoof_reading["spoof_score"],
        explain="likely AI-generated voice" if spoof_reading["spoof_score"] > 0.5 else "no spoof detected",
    )
    # Acoustic-only re-fusion: lets a voice that suddenly reads as synthetic escalate
    # the risk even between text turns, same as the real rolling design intends.
    fusion_result = session.fusion.combine(spoof_signal, session.last_scam_signal)
    entry = session.agent.update(fusion_result)

    return {
        "risk_score": fusion_result.risk_score,
        "spoof_score": spoof_reading["spoof_score"],
        "inference_ms": spoof_reading.get("inference_ms"),
        "new_entry": entry.to_ui_dict() if entry else None,
    }


@app.post("/api/live/end")
def live_end(session_id: str = Form(...)):
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
    if session is None:
        return {"ended": False}
    return {"ended": True, "final_log": [e.to_ui_dict() for e in session.agent.timeline]}


static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
