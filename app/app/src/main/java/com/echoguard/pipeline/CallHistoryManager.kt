package com.echoguard.pipeline

import android.content.Context
import com.echoguard.fusion.Action
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

class CallHistoryManager(private val context: Context) {

    private val historyFile = File(context.filesDir, "call_history.json")
    
    private val _history = MutableStateFlow<List<CallLog>>(emptyList())
    val history: StateFlow<List<CallLog>> = _history.asStateFlow()

    init {
        loadHistory()
    }

    private fun loadHistory() {
        if (!historyFile.exists()) {
            _history.value = emptyList()
            return
        }
        
        try {
            val jsonString = historyFile.readText()
            val jsonArray = JSONArray(jsonString)
            val loaded = mutableListOf<CallLog>()
            
            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                val actionStr = obj.optString("action", Action.MONITOR.name)
                val action = try { Action.valueOf(actionStr) } catch (e: Exception) { Action.MONITOR }
                
                loaded.add(
                    CallLog(
                        id = obj.getString("id"),
                        timestamp = obj.getLong("timestamp"),
                        title = obj.getString("title"),
                        riskScorePercent = obj.getInt("riskScorePercent"),
                        action = action,
                        transcriptSnippet = obj.optString("transcriptSnippet", ""),
                        bytesSent = obj.optLong("bytesSent", 0L)
                    )
                )
            }
            _history.value = loaded.sortedByDescending { it.timestamp }
        } catch (e: Exception) {
            e.printStackTrace()
            _history.value = emptyList()
        }
    }

    private fun saveHistory(logs: List<CallLog>) {
        try {
            val jsonArray = JSONArray()
            logs.forEach { log ->
                val obj = JSONObject().apply {
                    put("id", log.id)
                    put("timestamp", log.timestamp)
                    put("title", log.title)
                    put("riskScorePercent", log.riskScorePercent)
                    put("action", log.action.name)
                    put("transcriptSnippet", log.transcriptSnippet)
                    put("bytesSent", log.bytesSent)
                }
                jsonArray.put(obj)
            }
            historyFile.writeText(jsonArray.toString())
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun addLog(log: CallLog) {
        val currentList = _history.value.toMutableList()
        currentList.add(0, log) // Add to top
        _history.value = currentList
        saveHistory(currentList)
    }

    fun deleteLog(id: String) {
        val currentList = _history.value.toMutableList()
        currentList.removeAll { it.id == id }
        _history.value = currentList
        saveHistory(currentList)
    }

    fun clearAll() {
        _history.value = emptyList()
        if (historyFile.exists()) {
            historyFile.delete()
        }
    }
}
