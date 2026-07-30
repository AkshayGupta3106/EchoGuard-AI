package com.echoguard.ui

import android.app.Application
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.echoguard.pipeline.PipelineRunner
import com.echoguard.pipeline.PipelineUiState
import com.echoguard.acoustic.VadGate.Companion.CHUNK_SAMPLES
import com.echoguard.acoustic.VadGate.Companion.SAMPLE_RATE
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class PipelineViewModel(application: Application) : AndroidViewModel(application) {

    private val pipelineRunner = PipelineRunner(application)
    val uiState: StateFlow<PipelineUiState> = pipelineRunner.uiState

    private var audioRecord: AudioRecord? = null
    private var captureJob: Job? = null

    fun clearCache() {
        try {
            stopDemo()
            PipelineRunner.clearCache(getApplication())
        } catch (_: Throwable) {}
    }

    fun startDemo() {
        if (captureJob?.isActive == true) return
        val app = getApplication<Application>()

        captureJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                // Initialize models in background to avoid ANR
                pipelineRunner.start(app.assets)

                val minBufferSize = AudioRecord.getMinBufferSize(
                    SAMPLE_RATE.toInt(), AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
                )
                if (minBufferSize <= 0) {
                    android.util.Log.e("PipelineViewModel", "AudioRecord.getMinBufferSize() returned error $minBufferSize — mic may be in use by another app")
                    pipelineRunner.stop()
                    return@launch
                }
                
                // 1 second buffer (16000 samples * 2 bytes) to prevent data loss when processing blocks
                val recordingBufferBytes = SAMPLE_RATE.toInt() * 2
                
                audioRecord = AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    SAMPLE_RATE.toInt(),
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    maxOf(minBufferSize, recordingBufferBytes)
                )

                if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                    android.util.Log.e("PipelineViewModel", "AudioRecord failed to initialize (state=${audioRecord?.state}) — check mic permissions")
                    audioRecord?.release()
                    audioRecord = null
                    pipelineRunner.stop()
                    return@launch
                }

                audioRecord?.startRecording()

                val buffer = ShortArray(CHUNK_SAMPLES)
                val floatBuffer = FloatArray(CHUNK_SAMPLES)
                while (isActive) {
                    val read = audioRecord?.read(buffer, 0, CHUNK_SAMPLES) ?: 0
                    if (read > 0) {
                        for (i in 0 until read) {
                            floatBuffer[i] = buffer[i] / 32768.0f
                        }
                        pipelineRunner.onAudioChunk(floatBuffer.copyOf(read))
                    } else if (read < 0) {
                        break
                    }
                }
            } catch (e: Throwable) {
                android.util.Log.e("PipelineViewModel", "Error in startDemo", e)
            } finally {
                stopDemoInternal()
            }
        }
    }

    fun stopDemo() {
        captureJob?.cancel()
    }

    fun setLanguage(language: com.echoguard.pipeline.AppLanguage) {
        pipelineRunner.setLanguage(language)
    }

    private fun stopDemoInternal() {
        try {
            if (audioRecord?.state == AudioRecord.STATE_INITIALIZED) {
                audioRecord?.stop()
            }
            audioRecord?.release()
        } catch (e: Throwable) {
            e.printStackTrace()
        }
        audioRecord = null
        pipelineRunner.stop()
    }

    override fun onCleared() {
        super.onCleared()
        stopDemo()
    }
}
