"""
online_mode_api.py
------------------
Owner: whoever picks up the online-mode stretch goal (optional, Day 4+)

The optional "online mode" backend from the architecture: the app sends
ONLY fused signals + transcript text (never raw audio) here, gets back a
richer explanation or threat-intel lookup, and the app falls back to
offline mode automatically if this is unreachable.

Scope, deliberately narrow. Per the earlier discussion, most standard
backend-maturity items (rate limiting, DDoS defense, CDN, caching, API
integrations, monitoring dashboards, audits, load/stress testing) are
OUT of scope here - they solve problems a 4-day hackathon demo serving a
handful of calls doesn't have. This file only covers the four items that
were actually worth the time for THIS use case:

  1. Input sanitation  - reject malformed/out-of-range payloads before
                          touching them (a crash mid-demo looks bad).
  2. Secret management  - any real API key lives in an environment
                          variable, never hardcoded.
  3. Idempotency        - the client's fallback design implies retries
                          will happen on flaky connections; the same
                          call_id must not get double-processed.
  4. Fail tolerance     - this endpoint fails clearly and fast rather
                          than hanging, so the client's offline-fallback
                          logic actually has something to fall back FROM.

Run locally: uvicorn online_mode_api:app --reload
"""

from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="EchoGuard-AI online mode (optional)")

# --- 2. Secret management ---------------------------------------------------
# Never hardcode a real key here. If you wire up an actual threat-intel
# lookup, read it like this and fail loudly at startup if it's missing,
# rather than silently sending unauthenticated requests later.
THREAT_INTEL_API_KEY = os.environ.get("THREAT_INTEL_API_KEY")


# --- 1. Input sanitation -----------------------------------------------------
# Pydantic validates shape and ranges automatically; the custom validators
# below add the domain-specific checks (score bounds, non-empty call_id).
class FusedSignalPayload(BaseModel):
    call_id: str = Field(..., min_length=1, max_length=128)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    spoof_score: float = Field(..., ge=0.0, le=1.0)
    scam_score: float = Field(..., ge=0.0, le=1.0)
    transcript: str = Field(..., max_length=8000)  # generous cap, not unbounded

    @field_validator("call_id")
    @classmethod
    def call_id_no_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("call_id cannot be blank")
        return v.strip()

    @field_validator("transcript")
    @classmethod
    def transcript_not_just_whitespace(cls, v: str) -> str:
        return v.strip()


class ExplanationResponse(BaseModel):
    call_id: str
    richer_explanation: str
    threat_intel_hit: bool
    processing_ms: float


# --- 3. Idempotency ----------------------------------------------------------
# In-memory for a hackathon demo - swap for Redis/DB if this needs to survive
# a process restart. Keyed on call_id, since that's what the client would
# retry with unchanged.
_PROCESSED: dict[str, ExplanationResponse] = {}
_IDEMPOTENCY_TTL_SECONDS = 300  # don't grow this dict forever during a demo
_processed_at: dict[str, float] = {}


def _cleanup_stale_entries():
    now = time.time()
    stale = [k for k, t in _processed_at.items() if now - t > _IDEMPOTENCY_TTL_SECONDS]
    for k in stale:
        _PROCESSED.pop(k, None)
        _processed_at.pop(k, None)


# --- 4. Fail tolerance --------------------------------------------------------
# Explicit, fast, clearly-typed failures rather than hangs or 500s with no
# useful detail - the client's fallback logic needs a fast, unambiguous
# signal to fall back FROM, not a 10-second timeout to guess at.

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never leak internals to the client, but log them server-side for you.
    print(f"[online_mode_api] unhandled error on {request.url.path}: {exc!r}")
    return JSONResponse(
        status_code=503,
        content={"error": "online_mode_unavailable", "detail": "temporarily unavailable, use offline mode"},
    )


@app.post("/v1/explain", response_model=ExplanationResponse)
def explain(payload: FusedSignalPayload):
    _cleanup_stale_entries()

    # Idempotency check - same call_id returns the same cached result
    # instead of reprocessing (and potentially double-billing a real
    # threat-intel API call).
    if payload.call_id in _PROCESSED:
        return _PROCESSED[payload.call_id]

    start = time.time()

    # Fail tolerance: if a real downstream call (e.g. threat-intel lookup)
    # would go here, wrap it explicitly and degrade gracefully rather than
    # letting an exception bubble up as an opaque 500.
    threat_intel_hit = False
    if THREAT_INTEL_API_KEY:
        try:
            threat_intel_hit = _check_threat_intel(payload.call_id)
        except Exception as e:
            print(f"[online_mode_api] threat-intel lookup failed, degrading gracefully: {e!r}")
            threat_intel_hit = False  # fail closed, don't crash the request
    else:
        # No key configured - this is expected for most of the hackathon,
        # not an error. Skip the lookup rather than raising.
        pass

    explanation = _build_richer_explanation(payload, threat_intel_hit)

    result = ExplanationResponse(
        call_id=payload.call_id,
        richer_explanation=explanation,
        threat_intel_hit=threat_intel_hit,
        processing_ms=round((time.time() - start) * 1000, 1),
    )

    _PROCESSED[payload.call_id] = result
    _processed_at[payload.call_id] = time.time()
    return result


def _check_threat_intel(call_id: str) -> bool:
    """Placeholder - wire up a real threat-intel API call here if you build
    this out. Deliberately not implemented further: this is a stretch goal,
    not core scope."""
    return False


def _build_richer_explanation(payload: FusedSignalPayload, threat_intel_hit: bool) -> str:
    """Placeholder for a richer, server-side explanation (e.g. an LLM call
    with more context/compute than an on-device model could afford). Kept
    as simple templating here - the point of this file is the four
    reliability properties above, not this text."""
    risk_word = "high" if payload.risk_score >= 0.65 else "medium" if payload.risk_score >= 0.35 else "low"
    note = " This caller ID has been reported before." if threat_intel_hit else ""
    return f"Risk assessed as {risk_word} ({payload.risk_score:.0%}).{note}"


@app.get("/health")
def health():
    """Cheap endpoint for the client to check reachability before deciding
    whether to attempt online mode at all, rather than waiting for a
    request to time out."""
    return {"status": "ok"}
