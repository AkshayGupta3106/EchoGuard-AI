/**
 * Observable pipeline and timeline state shared by the runner and Compose UI.
 */

package com.echoguard.pipeline

import com.echoguard.fusion.Action

data class TimelineUiEntry(
    val time: String,
    val text: String,
    val riskScorePercent: Int,
    val action: Action,
)

sealed class MonitorStatus {
    object Idle : MonitorStatus()                 // no analysis active
    object Monitoring : MonitorStatus()            // actively capturing + scoring
}

enum class AppLanguage { ENGLISH, HINDI }

data class PipelineUiState(
    val status: MonitorStatus = MonitorStatus.Idle,
    val isInitializing: Boolean = false,
    val uiLanguage: AppLanguage = AppLanguage.ENGLISH,
    val currentRiskPercent: Int = 0,
    val currentAction: Action = Action.MONITOR,
    val timeline: List<TimelineUiEntry> = emptyList(),
    val liveTranscript: String = "",
    val errorMessage: String? = null,
    val warningMessage: String? = null,
)
