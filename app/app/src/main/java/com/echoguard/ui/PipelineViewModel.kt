package com.echoguard.ui

import android.app.Application
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.echoguard.pipeline.PipelineRunner
import com.echoguard.pipeline.PipelineUiState
import com.echoguard.pipeline.CallLog
import com.echoguard.pipeline.CallHistoryManager
import com.echoguard.acoustic.VadGate.Companion.CHUNK_SAMPLES
import com.echoguard.acoustic.VadGate.Companion.SAMPLE_RATE
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

enum class ThemeMode { SYSTEM, LIGHT, DARK }

class PipelineViewModel(application: Application) : AndroidViewModel(application) {

    private val prefs = application.getSharedPreferences("echoguard_prefs", android.content.Context.MODE_PRIVATE)
    
    private val _themeMode = MutableStateFlow(
        runCatching { ThemeMode.valueOf(prefs.getString("theme", ThemeMode.SYSTEM.name) ?: ThemeMode.SYSTEM.name) }.getOrDefault(ThemeMode.SYSTEM)
    )
    val themeMode: StateFlow<ThemeMode> = _themeMode

    fun setThemeMode(mode: ThemeMode) {
        prefs.edit().putString("theme", mode.name).apply()
        _themeMode.value = mode
    }

    private val pipelineRunner = PipelineRunner(application)
    private val switching = MutableStateFlow(false)
    val isSwitching: StateFlow<Boolean> = switching
    val uiState: StateFlow<PipelineUiState> = pipelineRunner.uiState

    val callHistoryManager = CallHistoryManager(application)
    val history: StateFlow<List<CallLog>> = callHistoryManager.history

    @Volatile private var audioRecord: AudioRecord? = null
    private var captureJob: Job? = null

    fun clearCache() {
        if (switching.value) return
        switching.value = true
        stopDemo()
        val job = captureJob
        viewModelScope.launch(Dispatchers.IO) {
            try {
                job?.join()
                PipelineRunner.clearCache(getApplication())
            } finally { switching.value = false }
        }
    }

    fun startDemo() {
        if (switching.value || captureJob?.isCompleted == false) return
        val app = getApplication<Application>()

        captureJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                // Initialize models in background to avoid ANR
                pipelineRunner.start(app.assets)
                currentCoroutineContext().ensureActive()
                if (pipelineRunner.uiState.value.status != com.echoguard.pipeline.MonitorStatus.Monitoring) return@launch

                val minBufferSize = AudioRecord.getMinBufferSize(
                    SAMPLE_RATE.toInt(), AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
                )
                if (minBufferSize <= 0) {
                    android.util.Log.e("PipelineViewModel", "AudioRecord.getMinBufferSize() returned error $minBufferSize — mic may be in use by another app")
                    pipelineRunner.stop()
                    pipelineRunner.reportCaptureError("Microphone buffer unavailable; the microphone may be in use by another app.")
                    return@launch
                }
                
                // 1 second buffer (16000 samples * 2 bytes) to prevent data loss when processing blocks
                val recordingBufferBytes = SAMPLE_RATE.toInt() * 2
                
                @android.annotation.SuppressLint("MissingPermission")
                val record = AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    SAMPLE_RATE.toInt(),
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    maxOf(minBufferSize, recordingBufferBytes)
                )
                audioRecord = record
                currentCoroutineContext().ensureActive()

                if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                    android.util.Log.e("PipelineViewModel", "AudioRecord failed to initialize (state=${audioRecord?.state}) — check mic permissions")
                    audioRecord?.release()
                    audioRecord = null
                    pipelineRunner.stop()
                    pipelineRunner.reportCaptureError("Microphone initialization failed. Check permissions and other recording apps.")
                    return@launch
                }

                record.startRecording()

                val buffer = ShortArray(CHUNK_SAMPLES)
                val floatBuffer = FloatArray(CHUNK_SAMPLES)
                while (isActive) {
                    val read = record.read(buffer, 0, CHUNK_SAMPLES)
                    if (read > 0) {
                        for (i in 0 until read) {
                            floatBuffer[i] = buffer[i] / 32768.0f
                        }
                        pipelineRunner.onAudioChunk(floatBuffer.copyOf(read))
                    } else if (read < 0) {
                        pipelineRunner.reportCaptureError("Microphone capture failed ($read).")
                        break
                    } else {
                        kotlinx.coroutines.delay(10)
                    }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                android.util.Log.e("PipelineViewModel", "Error in startDemo", e)
                pipelineRunner.reportCaptureError("Microphone analysis failed: ${e.localizedMessage}")
            } finally {
                stopDemoInternal()
            }
        }
    }

    fun stopDemo() {
        captureJob?.cancel()
        runCatching { audioRecord?.stop() }
    }

    fun setLanguage(language: com.echoguard.pipeline.AppLanguage) {
        if (isSwitching.value || captureJob?.isCompleted == false) return
        pipelineRunner.setLanguage(language)
    }

    private fun stopDemoInternal() {
        try {
            // Save to history if we did any work
            val demoState = pipelineRunner.uiState.value
            if (demoState.timeline.isNotEmpty() || demoState.liveTranscript.isNotBlank()) {
                
                // Name sequentially based on history size
                val demoNumber = callHistoryManager.history.value.size + 1
                val callTitle = "Microphone Session $demoNumber"
                
                callHistoryManager.addLog(
                    CallLog(
                        id = java.util.UUID.randomUUID().toString(),
                        timestamp = System.currentTimeMillis(),
                        title = callTitle,
                        riskScorePercent = demoState.currentRiskPercent,
                        action = demoState.currentAction,
                        transcriptSnippet = demoState.liveTranscript.take(100),
                        bytesSent = 0L,
                        transcript = demoState.liveTranscript,
                    )
                )
            }

        } catch (e: Throwable) {
            e.printStackTrace()
        } finally {
            runCatching { audioRecord?.stop() }
            audioRecord?.release()
            audioRecord = null
            pipelineRunner.stop()
        }
    }

    override fun onCleared() {
        super.onCleared()
        stopDemo()
        val job = captureJob
        kotlinx.coroutines.CoroutineScope(Dispatchers.IO).launch {
            job?.join()
            pipelineRunner.release()
        }
    }
}
