/**
 * SupervisorAgent.kt
 * Owner: Person A + Person B together
 *
 * Kotlin port of supervisor_agent.py - same Observe -> Reason -> Explain ->
 * Recommend -> Act loop, same "only emit a timeline entry when something
 * actually changed" behavior. This is what your fraud-timeline UI binds to.
 */

package com.echoguard.fusion

import com.echoguard.fusion.FusionResult

enum class RiskLevel { LOW, MEDIUM, HIGH }
enum class Action { MONITOR, WARN, BLOCK }

// Tune against real test calls once you have a few - starting guesses only.
private val MEDIUM_THRESHOLD = 0.35f
private val HIGH_THRESHOLD = 0.65f

private fun riskLevel(score: Float): RiskLevel = when {
    score >= HIGH_THRESHOLD -> RiskLevel.HIGH
    score >= MEDIUM_THRESHOLD -> RiskLevel.MEDIUM
    else -> RiskLevel.LOW
}

private fun actionForLevel(level: RiskLevel): Action = when (level) {
    RiskLevel.LOW -> Action.MONITOR
    RiskLevel.MEDIUM -> Action.WARN
    RiskLevel.HIGH -> Action.BLOCK
}

data class TimelineEntry(
    val elapsedStr: String,
    val reasoning: String,
    val explanation: String,
    val recommendation: Action,
    val riskScore: Float,
) {
    /** Shape a timeline UI component (e.g. a Compose LazyColumn item) consumes directly. */
    fun toUiModel() = mapOf(
        "time" to elapsedStr,
        "text" to explanation,
        "riskScore" to (riskScore * 100).toInt(),
        "action" to recommendation.name.lowercase(),
    )
}

class SupervisorAgent {
    private var callStartMs = System.currentTimeMillis()
    private val _timeline = mutableListOf<TimelineEntry>()
    private var lastLevel: RiskLevel? = null
    private var lastSignalsKey: String? = null

    val timeline: List<TimelineEntry> get() = _timeline.toList()

    fun reset() {
        callStartMs = System.currentTimeMillis()
        _timeline.clear()
        lastLevel = null
        lastSignalsKey = null
    }

    private fun elapsedStr(): String {
        val secs = ((System.currentTimeMillis() - callStartMs) / 1000).toInt()
        return "%02d:%02d".format(secs / 60, secs % 60)
    }

    private fun signalsKey(result: FusionResult) =
        "${result.spoofSignal.explain}|${result.scamSignal.explain}"

    /**
     * Call every time the fusion engine produces a new result (same 1-2s
     * cadence as the two streams). Returns null on ticks where nothing new
     * happened - the caller shouldn't touch the UI in that case.
     */
    fun update(result: FusionResult): TimelineEntry? {
        val level = riskLevel(result.riskScore)
        val signalsKey = signalsKey(result)
        val levelChanged = level != lastLevel
        val signalsChanged = signalsKey != lastSignalsKey

        if (!levelChanged && !signalsChanged) return null

        val reasoning = "risk level is now ${level.name.lowercase()} (${"%.2f".format(result.riskScore)})"
        val action = actionForLevel(level)

        val entry = TimelineEntry(
            elapsedStr = elapsedStr(),
            reasoning = reasoning,
            explanation = result.explain(),
            recommendation = action,
            riskScore = result.riskScore,
        )
        _timeline.add(entry)
        lastLevel = level
        lastSignalsKey = signalsKey
        return entry
    }
}

/**
 * --- Day 3 integration sketch ---
 *
 * val fusion = FusionEngine()
 * val agent = SupervisorAgent()
 *
 * // every 1-2s, once both streams have produced a score:
 * val spoofSignal = StreamSignal(score = spoofDetector.score().spoofScore,
 *     explain = if (spoofScore > 0.5f) "likely AI-generated voice" else "no spoof detected")
 * val scamSignal = StreamSignal(score = scamResult.score, explain = scamResult.explain)
 * val fusionResult = fusion.combine(spoofSignal, scamSignal)
 * agent.update(fusionResult)?.let { entry ->
 *     timelineUiState.add(entry.toUiModel())  // only touch UI when something changed
 * }
 */
