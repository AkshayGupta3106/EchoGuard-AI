package com.echoguard.pipeline

import com.echoguard.fusion.Action

data class CallLog(
    val id: String,
    val timestamp: Long,
    val title: String, // "Demo Call - Unknown" or real number
    val riskScorePercent: Int,
    val action: Action,
    val transcriptSnippet: String,
    val bytesSent: Long
)
