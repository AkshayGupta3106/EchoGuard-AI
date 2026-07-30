/**
 * PipelineState.kt
 * Owner: together (Day 3)
 *
 * The shape CallMonitorService publishes and MainActivity's UI observes.
 * Kept in its own file since both the service (producer) and the UI
 * (consumer) need it, and neither should depend on the other's internals.
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
    object Idle : MonitorStatus()                 // no call active
    object ListeningNoSpeaker : MonitorStatus()    // call active, speaker not confirmed on yet
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
)
