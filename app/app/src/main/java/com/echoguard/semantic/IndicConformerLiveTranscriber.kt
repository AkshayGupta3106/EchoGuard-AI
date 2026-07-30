/**
 * IndicConformerLiveTranscriber.kt
 * Owner: Person B (semantic stream - Hindi/Hinglish ASR path)
 *
 * Uses AI4Bharat's IndicConformer (NeMo CTC) via sherpa-onnx's OfflineRecognizer.
 * Transcribes Hindi and Hinglish speech natively with high accuracy and low memory footprint.
 */

package com.echoguard.semantic

import android.content.res.AssetManager
import android.util.Log
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineNemoEncDecCtcModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig

class IndicConformerLiveTranscriber(
    private val assetManager: AssetManager,
    private val onPartialResult: (text: String, isFinal: Boolean) -> Unit,
    private val onLatencyMeasured: (millis: Long) -> Unit = {},
    private val languageCode: String = "hi",
) {
    private var recognizer: OfflineRecognizer? = null
    private val pcmBuffer = mutableListOf<Float>()

    companion object {
        private const val TAG = "IndicConformerDebug"
        const val SAMPLE_RATE = 16000
        private const val MIN_SAMPLES_TO_DECODE = 16000 * 2 // Decode every 2 seconds of audio
    }

    private val modelDir get() = "indicconformer-$languageCode"
    private val modelPath get() = "$modelDir/model.int8.onnx"
    private val tokensPath get() = "$modelDir/tokens.txt"

    private fun validateAssetsExist() {
        val required = listOf(modelPath, tokensPath)
        val missing = required.filterNot { path ->
            try { assetManager.open(path).close(); true }
            catch (e: java.io.IOException) { false }
        }
        if (missing.isNotEmpty()) {
            throw IllegalStateException(
                "IndicConformerLiveTranscriber: missing required asset(s): $missing.\n" +
                "Download from https://huggingface.co/parismitaglobalsolutions/indicconformer-sherpa-onnx"
            )
        }
    }

    fun init() {
        try {
            validateAssetsExist()

            val nemoConfig = OfflineNemoEncDecCtcModelConfig(model = modelPath)
            val modelConfig = OfflineModelConfig(
                nemo = nemoConfig,
                tokens = tokensPath,
                numThreads = 1, // 1 thread to minimize RAM and CPU usage
                provider = "cpu",
            )
            val featConfig = FeatureConfig(
                sampleRate = SAMPLE_RATE,
                featureDim = 80,
            )
            val config = OfflineRecognizerConfig(
                featConfig = featConfig,
                modelConfig = modelConfig,
            )

            recognizer = OfflineRecognizer(assetManager, config)
            Log.i(TAG, "IndicConformer Hindi ASR Engine initialized successfully.")
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to initialize IndicConformer OfflineRecognizer", e)
            release()
            throw e
        }
    }

    private var silenceChunks = 0

    /**
     * Feed PCM samples from the microphone audio loop.
     * Accumulates audio and transcribes Hindi speech into text.
     */
    fun acceptWaveform(samples: FloatArray) {
        val r = recognizer ?: return
        try {
            synchronized(this) {
                var energy = 0f
                for (s in samples) {
                    pcmBuffer.add(s)
                    energy += s * s
                }
                energy /= samples.size

                // Simple energy-based VAD for natural chunking
                if (energy < 0.0005f) {
                    silenceChunks++
                } else {
                    silenceChunks = 0
                }

                // Decode if we hit a natural pause (~400ms = 12 chunks of 32ms) and have at least 1.5s of audio,
                // OR if the buffer is getting too long (force decode at 6s to prevent massive lag).
                val isNaturalPause = silenceChunks >= 12 && pcmBuffer.size >= (16000 * 1.5).toInt()
                val isBufferFull = pcmBuffer.size >= 16000 * 6

                if (!isNaturalPause && !isBufferFull) return

                val audioArray = pcmBuffer.toFloatArray()
                pcmBuffer.clear()
                silenceChunks = 0

                val t0 = System.currentTimeMillis()
                val stream = r.createStream()
                stream.acceptWaveform(audioArray, SAMPLE_RATE)
                r.decode(stream)
                val text = r.getResult(stream).text
                stream.release()

                if (text.isNotBlank()) {
                    Log.d(TAG, "Hindi ASR Transcribed text: '$text'")
                    onPartialResult(text, true)
                }
                onLatencyMeasured(System.currentTimeMillis() - t0)
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Error decoding Hindi audio chunk", e)
        }
    }

    fun release() {
        try {
            recognizer?.release()
        } catch (_: Throwable) {}
        recognizer = null
        synchronized(this) { pcmBuffer.clear() }
    }
}
