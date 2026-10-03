"""
supervisor_agent.py
------------------
Rule-based supervisor with an explicit decision loop on fusion updates:

    Observe -> Reason -> Explain -> Recommend -> Act

Produces MONITOR, WARN, or BLOCK recommendations and explanation entries.
The timeline appends when risk level or contributing signals change, rather
than on every scoring tick. Recommendations do not directly block calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from echoguard.fusion.fusion_engine import FusionResult


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Action(Enum):
    MONITOR = "monitor"       # keep listening, nothing shown to the user yet
    WARN = "warn"             # show a warning banner
    BLOCK = "block"           # recommend hanging up / block the caller


# Recommendation thresholds; not calibrated probabilities of real-call fraud.
RISK_THRESHOLDS = {
    RiskLevel.LOW: 0.0,
    RiskLevel.MEDIUM: 0.35,
    RiskLevel.HIGH: 0.65,
}

ACTION_FOR_LEVEL = {
    RiskLevel.LOW: Action.MONITOR,
    RiskLevel.MEDIUM: Action.WARN,
    RiskLevel.HIGH: Action.BLOCK,
}


def _risk_level(score: float) -> RiskLevel:
    if score >= RISK_THRESHOLDS[RiskLevel.HIGH]:
        return RiskLevel.HIGH
    if score >= RISK_THRESHOLDS[RiskLevel.MEDIUM]:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


@dataclass
class TimelineEntry:
    """Risk assessment and recommendation for one timeline entry."""
    timestamp: float
    elapsed_str: str
    observation: str      # Observe: what came in this tick
    reasoning: str         # Reason: risk level + what changed
    explanation: str       # Explain: the human-readable "why"
    recommendation: Action # Recommend: what the agent suggests
    risk_score: float

    def to_ui_dict(self) -> dict:
        """Serialize an entry for the browser timeline."""
        return {
            "time": self.elapsed_str,
            "text": self.explanation,
            "risk_score": round(self.risk_score * 100),
            "action": self.recommendation.value,
        }


class SupervisorAgent:
    def __init__(self):
        self._call_start = time.time()
        self._timeline: List[TimelineEntry] = []
        self._last_level: Optional[RiskLevel] = None
        self._last_signals_key: Optional[str] = None

    def reset(self):
        """Clear timeline and decision state for an independent session."""
        self._call_start = time.time()
        self._timeline = []
        self._last_level = None
        self._last_signals_key = None

    @property
    def timeline(self) -> List[TimelineEntry]:
        return list(self._timeline)

    def _elapsed_str(self) -> str:
        secs = int(time.time() - self._call_start)
        return f"{secs//60:02d}:{secs%60:02d}"

    def _signals_key(self, fusion_result: FusionResult) -> str:
        """Identify explanation changes independently of risk level."""
        return f"{fusion_result.spoof_signal.explain}|{fusion_result.scam_signal.explain}"

    def update(self, fusion_result: FusionResult) -> Optional[TimelineEntry]:
        """Append an entry when risk level or contributing signals change; otherwise return None."""
        # --- Observe ---
        observation = (f"spoof_score={fusion_result.spoof_signal.score:.2f}, "
                        f"scam_score={fusion_result.scam_signal.score:.2f}")

        # --- Reason ---
        level = _risk_level(fusion_result.risk_score)
        signals_key = self._signals_key(fusion_result)
        level_changed = level != self._last_level
        signals_changed = signals_key != self._last_signals_key

        if not level_changed and not signals_changed:
            return None  # nothing new - don't spam the timeline

        reasoning = f"risk level is now {level.value} ({fusion_result.risk_score:.2f})"
        if level_changed and self._last_level is not None:
            reasoning += f", up from {self._last_level.value}" if RISK_THRESHOLDS[level] > RISK_THRESHOLDS[self._last_level] \
                         else f", down from {self._last_level.value}"

        # --- Explain ---
        explanation = fusion_result.explain()

        # --- Recommend ---
        action = ACTION_FOR_LEVEL[level]

        # Record the recommendation for display; this agent does not block calls.
        entry = TimelineEntry(
            timestamp=time.time(),
            elapsed_str=self._elapsed_str(),
            observation=observation,
            reasoning=reasoning,
            explanation=explanation,
            recommendation=action,
            risk_score=fusion_result.risk_score,
        )
        self._timeline.append(entry)
        self._last_level = level
        self._last_signals_key = signals_key
        return entry


# ---------------------------------------------------------------------------
# Self-test: simulates a call escalating over time using synthetic fusion
# results (no real models needed here - see demo_pipeline.py for that).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from echoguard.fusion.fusion_engine import StreamSignal

    agent = SupervisorAgent()

    # Simulate a call that starts benign and escalates into a scam script.
    timeline_inputs = [
        FusionResult(0.04, StreamSignal(0.05, "no spoof detected"), StreamSignal(0.03, "no scam signals detected")),
        FusionResult(0.04, StreamSignal(0.05, "no spoof detected"), StreamSignal(0.03, "no scam signals detected")),
        FusionResult(0.55, StreamSignal(0.10, "no spoof detected"), StreamSignal(0.85, "claimed authority (\"calling from your bank's security department\")")),
        FusionResult(0.90, StreamSignal(0.92, "likely AI-generated voice"), StreamSignal(0.88, "otp request; urgency; secrecy pressure")),
        FusionResult(0.90, StreamSignal(0.92, "likely AI-generated voice"), StreamSignal(0.88, "otp request; urgency; secrecy pressure")),
    ]

    for fusion_result in timeline_inputs:
        entry = agent.update(fusion_result)
        if entry:
            print(f"[{entry.elapsed_str}] risk={entry.risk_score:.2f} "
                  f"action={entry.recommendation.value} | {entry.explanation}")
        else:
            print("(no change - skipped)")

    print("\n--- Full timeline for UI ---")
    for e in agent.timeline:
        print(e.to_ui_dict())
