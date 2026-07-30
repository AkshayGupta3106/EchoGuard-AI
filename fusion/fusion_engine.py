"""
fusion_engine.py
------------------
Owner: Person A + Person B together (Day 3 in the plan)

Combines Person A's spoof_score() and Person B's scam_score() into one
fraud-risk score. This is the MVP weighted-sum version from the
architecture (marked USABLE WITH CAVEAT throughout) - a trained
logistic-regression fusion is the stretch goal, contingent on hand-labeling
~30-50 real test-call examples, and is NOT what this implements.

Design note: this stays deliberately simple on purpose. It only combines
numbers - all the "why" (which rules fired, closest scam exemplar, etc.)
stays attached and flows through to SupervisorAgent unchanged, since that's
what actually gets explained to the user, not the fusion math itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class StreamSignal:
    """Common shape both streams' outputs get wrapped into before fusion.
    Keeps FusionEngine decoupled from exactly how each score was produced -
    it doesn't need to know AASIST-L or MiniLM exist."""
    score: float                  # 0-1
    explain: str = ""             # human-readable reason string
    raw: Optional[dict] = None    # original result dict, kept for debugging/UI


@dataclass
class FusionResult:
    risk_score: float
    spoof_signal: StreamSignal
    scam_signal: StreamSignal

    def explain(self) -> str:
        parts = []
        if self.spoof_signal.explain:
            parts.append(f"voice: {self.spoof_signal.explain}")
        if self.scam_signal.explain:
            parts.append(f"conversation: {self.scam_signal.explain}")
        return "; ".join(parts) if parts else "no signals"


class FusionEngine:
    """Asymmetric Corroboration model. Scam sets baseline, Spoof acts as an accelerant."""

    def __init__(self):
        pass

    def combine(self, spoof_signal: StreamSignal, scam_signal: StreamSignal) -> FusionResult:
        max_spoof_boost = 0.45
        risk = scam_signal.score + spoof_signal.score * (1.0 - scam_signal.score) * max_spoof_boost
        return FusionResult(
            risk_score=round(risk, 4),
            spoof_signal=spoof_signal,
            scam_signal=scam_signal,
        )


# ---------------------------------------------------------------------------
# Self-test with synthetic signals (no models needed - this only tests the
# fusion math itself; see demo_pipeline.py for the full real-model test).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = FusionEngine()

    scenarios = [
        ("benign call", StreamSignal(0.05, "no spoof detected"), StreamSignal(0.03, "no scam signals detected")),
        ("cloned voice, benign talk", StreamSignal(0.92, "likely AI-generated voice"), StreamSignal(0.08, "no scam signals detected")),
        ("real voice, scam script", StreamSignal(0.10, "no spoof detected"), StreamSignal(0.87, "otp request; claimed authority")),
        ("both fire", StreamSignal(0.95, "likely AI-generated voice"), StreamSignal(0.90, "otp request; urgency; secrecy pressure")),
    ]

    for name, spoof, scam in scenarios:
        result = engine.combine(spoof, scam)
        print(f"\n--- {name} ---")
        print(f"risk_score = {result.risk_score}")
        print(f"why: {result.explain()}")
