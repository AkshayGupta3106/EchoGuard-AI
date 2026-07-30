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

class FusionEngine {
    private var peakRiskScore = 0f
    private val lock = Any()

    fun combine(spoofSignal: StreamSignal, scamSignal: StreamSignal): FusionResult = synchronized(lock) {
        // Asymmetric Boost: Scam sets baseline, Spoof boosts by max 45%
        val maxSpoofBoost = 0.45f
        val risk = scamSignal.score + spoofSignal.score * (1.0f - scamSignal.score) * maxSpoofBoost
        
        if (risk > peakRiskScore) {
            peakRiskScore = risk
        }
        
        FusionResult(
            riskScore = peakRiskScore,
            spoofSignal = spoofSignal,
            scamSignal = scamSignal,
        )
    }

    fun reset() = synchronized(lock) {
        peakRiskScore = 0f
    }
}
