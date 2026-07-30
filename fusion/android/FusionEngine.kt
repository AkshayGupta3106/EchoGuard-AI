/**
 * FusionEngine.kt
 * Owner: Person A + Person B together
 *
 * Kotlin port of fusion_engine.py - same weighted-sum MVP, same normalized
 * weights, same StreamSignal/FusionResult shape. Verified logic against the
 * Python version's test scenarios before porting.
 */

package com.echoguard.fusion

data class StreamSignal(
    val score: Float,           // 0-1
    val explain: String = "",   // human-readable reason
)

data class FusionResult(
    val riskScore: Float,
    val spoofSignal: StreamSignal,
    val scamSignal: StreamSignal,
) {
    fun explain(): String {
        val parts = mutableListOf<String>()
        if (spoofSignal.explain.isNotBlank()) parts.add("voice: ${spoofSignal.explain}")
        if (scamSignal.explain.isNotBlank()) parts.add("conversation: ${scamSignal.explain}")
        return if (parts.isNotEmpty()) parts.joinToString("; ") else "no signals"
    }
}

class FusionEngine(spoofWeight: Float = 0.4f, scamWeight: Float = 0.6f) {
    private val total = spoofWeight + scamWeight
    private val normSpoofWeight = spoofWeight / total
    private val normScamWeight = scamWeight / total

    fun combine(spoofSignal: StreamSignal, scamSignal: StreamSignal): FusionResult {
        // Reflects the CURRENT fused signal only - no peak-hold. A previous
        // version latched onto the highest score ever seen in the call and
        // could never report a lower one afterward, which is exactly why
        // the risk score got permanently stuck (famously at 39%, from an
        // AASIST-L emulator-mic artifact early in a call). If you want
        // "escalation sticks" behavior back, do it explicitly and visibly
        // in SupervisorAgent/PipelineRunner (e.g. track peak alongside
        // current, and show both) - don't bury it back in here where it
        // silently breaks "the score updates dynamically."
        val weighted = normSpoofWeight * spoofSignal.score + normScamWeight * scamSignal.score

        // Signal floor: a pure weighted average lets a near-certain single
        // signal get diluted by a weak one - e.g. spoof=0.92 (near-certain
        // AI-cloned voice), scam=0.08 (benign script) previously produced
        // risk=0.416, barely above "medium", despite a near-certain
        // deepfake-voice detection being a severe red flag on its own.
        // Verified via fusion_engine.py's self-test before porting here:
        // 0.416 -> 0.644 with this floor, 0.038 (benign) and 0.92 (both
        // fire) unaffected. 0.7, not 1.0 - fusion exists precisely because
        // either single modality could still be a false positive, so one
        // strong signal shouldn't automatically max out risk alone.
        val floor = 0.7f * maxOf(spoofSignal.score, scamSignal.score)
        val risk = maxOf(weighted, floor)

        return FusionResult(
            riskScore = risk,
            spoofSignal = spoofSignal,
            scamSignal = scamSignal,
        )
    }

    fun reset() {
        // No latched state to clear anymore, but kept as a no-op so callers
        // (PipelineRunner.start()) don't need to change.
    }
}
