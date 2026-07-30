"""
supervisor_agent.py
------------------
Owner: Person A + Person B together (Day 3 in the plan)

This is what the architecture calls SupervisorAgent - same decision logic
as a plain rule-based agent, framed as an explicit five-step loop that runs
on every fusion update:

    Observe -> Reason -> Explain -> Recommend -> Act

Why bother with the loop framing instead of just "if risk > threshold:
warn"? Two reasons: (1) it maps directly onto the fraud-timeline UI - each
step below produces exactly what gets rendered as one timeline entry, and
(2) it's a much better story to tell judges than a bare number, which was
the whole point of the original "make it explain its reasoning over time"
idea this project started from.

Design choice worth knowing: this only appends a new timeline entry when
something actually changes (risk crosses a threshold, or a new signal
starts firing) - not on every 1-2s tick. A timeline with an entry every
single second is noise, not a story.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from fusion_engine import FusionResult


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Action(Enum):
    MONITOR = "monitor"       # keep listening, nothing shown to the user yet
    WARN = "warn"             # show a warning banner
    BLOCK = "block"           # recommend hanging up / block the caller


# Tune these against real test calls once you have a few - these are
# starting thresholds, not calibrated values.
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
    """One row in the fraud timeline UI - this is the direct output of one
    pass through the five-step loop."""
    timestamp: float
    elapsed_str: str
    observation: str      # Observe: what came in this tick
    reasoning: str         # Reason: risk level + what changed
    explanation: str       # Explain: the human-readable "why"
    recommendation: Action # Recommend: what the agent suggests
    risk_score: float

    def to_ui_dict(self) -> dict:
        """Shape a timeline UI component would actually consume."""
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
        """Call at the start of each new call."""
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
        """A cheap fingerprint of *which* signals are firing, so we can
        detect 'something new happened' even if the risk score barely moved."""
        return f"{fusion_result.spoof_signal.explain}|{fusion_result.scam_signal.explain}"

    def update(self, fusion_result: FusionResult) -> Optional[TimelineEntry]:
        """
        Call this every time the fusion engine produces a new result
        (same 1-2s cadence as the two underlying streams).

        Returns a new TimelineEntry only when something actually changed
        (risk level crossed a threshold, or the specific signals firing
        changed) - returns None on ticks where nothing new happened, so
        the caller knows not to touch the UI.
        """
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
            reasoning += f", up from {self._last_level.value}" if level.value > self._last_level.value \
                         else f", down from {self._last_level.value}"

        # --- Explain ---
        explanation = fusion_result.explain()

        # --- Recommend ---
        action = ACTION_FOR_LEVEL[level]

        # --- Act (here: record it; the actual UI/OS-level act - showing a
        # banner, offering to block the number - happens in the app layer
        # that consumes this timeline) ---
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
    from fusion_engine import StreamSignal

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
