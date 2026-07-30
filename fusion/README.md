# Fusion + SupervisorAgent — Person A + Person B, together (Day 3)

## Files

| File | Status | What it is |
|---|---|---|
| `fusion_engine.py` | **Tested, working** | Weighted-sum MVP: combines spoof_score + scam_score into one risk_score |
| `supervisor_agent.py` | **Tested, working** | The Observe→Reason→Explain→Recommend→Act loop; produces the fraud timeline |
| `demo_pipeline.py` | **Tested, working end-to-end** | Wires real AASIST-L + real scam_classifier.py through fusion + agent on two simulated calls |
| `android/FusionEngine.kt` | Integration skeleton | Kotlin port, same logic |
| `android/SupervisorAgent.kt` | Integration skeleton | Kotlin port, same logic |

`demo_pipeline.py` is the actual Day 3 integration test — run it once `/acoustic/` and `/semantic/` sit next to `/fusion/` as sibling folders (matching the repo structure from the plan):

```
python3 demo_pipeline.py
```

Verified output: a benign call stays low-risk throughout; an escalating scam call (bank impersonation → account threat → OTP request) correctly climbs from `warn` to `block` as more signals fire, with the timeline only recording an entry when something actually changed — not every tick.

## What "done together" means here

This is the seam where Person A's and Person B's independently-built modules have to agree on a contract. `demo_pipeline.py` is the proof that they do: it imports `RollingSpoofScorer` from `/acoustic/` and `ScamClassifier` from `/semantic/` completely unmodified, wraps both outputs in the same `StreamSignal` shape, and feeds them through fusion → agent without either module needing to know the other exists.

## Tuning notes

- **Fusion weights** (`spoof_weight=0.4, scam_weight=0.6` in `FusionEngine.__init__`) are a starting guess, not calibrated. Once you have a handful of real test calls, check whether risk scores land where you'd expect and adjust.
- **Risk thresholds** (`RISK_THRESHOLDS` in `supervisor_agent.py`: medium=0.35, high=0.65) are also untuned starting points — same advice.
- **Timeline noise filtering**: `SupervisorAgent.update()` returns `None` when neither the risk level nor the specific firing signals changed, so ticks where nothing new happened don't spam the UI. If your demo call feels like it's missing updates, check this logic before assuming something's broken — it's filtering on purpose.

## On the "online mode" backend (optional, stretch goal only)

Per the earlier scope discussion: most standard backend-maturity concerns (rate limiting, DDoS defense, caching, CDN, monitoring dashboards, audits) are explicitly **out of scope** for this timeline — they solve problems a 4-day hackathon demo doesn't have. The four that were flagged as actually worth the time (fail tolerance, input sanitation, secret management, idempotency) are implemented minimally in `backend/online_mode_api.py`, alongside this folder — see that file's own docstring for what's covered and, just as importantly, what's deliberately left out.
