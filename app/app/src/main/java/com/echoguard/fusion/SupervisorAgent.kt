/**
 * Produces risk levels, recommendations, and localized timeline explanations.
 * Emits entries when risk level or contributing signals change.
 */

package com.echoguard.fusion

import com.echoguard.pipeline.AppLanguage

enum class RiskLevel { LOW, MEDIUM, HIGH }
enum class Action { MONITOR, WARN, BLOCK }

// Recommendation thresholds, not calibrated fraud probabilities.
private const val MEDIUM_THRESHOLD = 0.35f
private const val HIGH_THRESHOLD = 0.65f

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
)

class SupervisorAgent {
    private var callStartMs = System.currentTimeMillis()
    private val _timeline = mutableListOf<TimelineEntry>()
    private var lastLevel: RiskLevel? = null
    private var lastSignalsKey: String? = null

    private val lock = Any()

    fun reset() = synchronized(lock) {
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
     * Append a timeline entry when risk level or contributing signals change.
     * Returns null for unchanged results.
     */
    fun update(result: FusionResult, language: AppLanguage = AppLanguage.ENGLISH): TimelineEntry? = synchronized(lock) {
        val level = riskLevel(result.riskScore)
        val signalsKey = signalsKey(result)
        val levelChanged = level != lastLevel
        val signalsChanged = signalsKey != lastSignalsKey

        if (!levelChanged && !signalsChanged) return null

        val action = actionForLevel(level)
        
        val explanation = if (language == AppLanguage.HINDI) {
            getHindiExplanation(result)
        } else {
            result.explain()
        }

        val reasoning = "risk level is now ${level.name.lowercase()} (${"%.2f".format(result.riskScore)})"

        val entry = TimelineEntry(
            elapsedStr = elapsedStr(),
            reasoning = reasoning,
            explanation = explanation,
            recommendation = action,
            riskScore = result.riskScore,
        )
        _timeline.add(entry)
        lastLevel = level
        lastSignalsKey = signalsKey
        entry
    }

    private fun getHindiExplanation(result: FusionResult): String {
        val parts = mutableListOf<String>()
        val spoof = result.spoofSignal.explain
        val scam = result.scamSignal.explain
        
        if (spoof.contains("likely AI-generated")) {
            parts.add("आवाज़: एआई-जनित लग रही है")
        }
        
        if (scam.isNotBlank()) {
            var localizedScam = scam
                .replace("otp request", "OTP की मांग")
                .replace("urgency", "जल्दबाजी का दबाव")
                .replace("claimed authority", "फर्जी अधिकारी")
                .replace("account threat", "खाता बंद करने की धमकी")
                .replace("secrecy pressure", "गोपनीयता का दबाव")
                .replace("remote access", "रिमोट एक्सेस ऐप")
                .replace("identity theft", "KYC धोखाधड़ी")
                .replace("lottery scam", "लॉटरी का लालच")
                .replace("utility scam", "बिजली/गैस बिल फ्रॉड")
                .replace("reward scam", "कैशबैक/इनाम का झांसा")
                .replace("tech support scam", "फर्जी टेक सपोर्ट")
                .replace("extortion kidnapping", "फिरौती/अपहरण की धमकी")
                .replace("legal threat arrest", "कानूनी/गिरफ्तारी की धमकी")
                .replace("payment demand", "पैसे की मांग")
                .replace("conversation:", "बातचीत:")
                .replace("joke/prank detected (overridden)", "मज़ाक/शरारत पकड़ी गई (स्कोर रद्द)")
                .replace("no scam signals detected", "कोई धोखाधड़ी का संकेत नहीं")
                .replace("semantically similar to known scam phrasing", "धोखाधड़ी के वाक्यांशों के समान")
            
            parts.add(localizedScam)
        }
        
        return if (parts.isNotEmpty()) parts.joinToString("; ") else "कोई संकेत नहीं"
    }
}
