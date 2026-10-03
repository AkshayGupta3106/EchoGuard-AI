/**
 * English streaming recognition through sherpa-onnx's Kroko transducer.
 * Requires packaged INT8 encoder, decoder, joiner, and tokens under kroko-128l/.
 * Accepts 16 kHz mono PCM frames; audio capture is handled by the caller.
 */

package com.echoguard.semantic

import com.k2fsa.sherpa.onnx.OnlineRecognizer
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineStream
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig
import com.k2fsa.sherpa.onnx.FeatureConfig

class KrokoLiveTranscriber(
    private val assetManager: android.content.res.AssetManager,
    private val onPartialResult: (text: String, isFinal: Boolean) -> Unit,
    private val onLatencyMeasured: (millis: Long) -> Unit = {}
) {
    private var recognizer: OnlineRecognizer? = null
    private var stream: OnlineStream? = null
    @Volatile var lastError: String? = null
        private set

    companion object {
        private const val ENCODER = "kroko-128l/encoder.int8.onnx"
        private const val DECODER = "kroko-128l/decoder.int8.onnx"
        private const val JOINER = "kroko-128l/joiner.int8.onnx"
        private const val TOKENS = "kroko-128l/tokens.txt"
        // Optional contextual biasing file; absent assets disable hotword biasing.
        private const val HOTWORDS_FILE = "Kroko/hotwords.txt"
        private const val HOTWORDS_SCORE = 2.0f  // default score for terms in the file without their own
        const val SAMPLE_RATE = 16000
    }

    /**
     * Validate required assets in Kotlin before constructing the native recognizer.
     */
    private fun validateAssetsExist() {
        val required = listOf(ENCODER, DECODER, JOINER, TOKENS)
        val missing = required.filterNot { path ->
            try {
                assetManager.open(path).close()
                true
            } catch (e: java.io.IOException) {
                false
            }
        }
        if (missing.isNotEmpty()) {
            throw IllegalStateException(
                "KrokoLiveTranscriber: missing required model asset(s): $missing. " +
                "Check app/src/main/assets/kroko-128l/ - all of encoder.int8.onnx, decoder.int8.onnx, " +
                "joiner.int8.onnx, and tokens.txt must be present, from the SAME model export " +
                "(matching precision - don't mix an int8 file with fp32 files)."
            )
        }
        // Missing hotwords disable contextual biasing, not recognition startup.
        try {
            assetManager.open(HOTWORDS_FILE).close()
        } catch (e: java.io.IOException) {
            android.util.Log.w("KrokoLiveTranscriber",
                "$HOTWORDS_FILE not found - contextual biasing disabled, " +
                "OTP/CVV/etc. recognition accuracy will be lower than with it. " +
                "Optionally bundle Kroko/hotwords.txt to enable contextual biasing.")
        }
    }

    fun init() {
        lastError = null
        try {
            validateAssetsExist()

            val transducerConfig = OnlineTransducerModelConfig(
                encoder = ENCODER,
                decoder = DECODER,
                joiner = JOINER,
            )
            val modelConfig = OnlineModelConfig(
                transducer = transducerConfig,
                tokens = TOKENS,
                numThreads = 1,          // 1 thread to minimize RAM and CPU footprint
                provider = "cpu",
            )
            val featConfig = FeatureConfig(
                sampleRate = SAMPLE_RATE,
                featureDim = 80,
            )

            val hotwordsAvailable = try {
                assetManager.open(HOTWORDS_FILE).close(); true
            } catch (e: java.io.IOException) { false }

            val config = OnlineRecognizerConfig(
                featConfig = featConfig,
                modelConfig = modelConfig,
                decodingMethod = "modified_beam_search",
                maxActivePaths = 1,                      // 1 path for mobile CPU & low RAM
                enableEndpoint = true,
                hotwordsFile = if (hotwordsAvailable) HOTWORDS_FILE else "",
                hotwordsScore = HOTWORDS_SCORE,
            )
            recognizer = OnlineRecognizer(assetManager, config)
            stream = recognizer?.createStream()
        } catch (e: Throwable) {
            android.util.Log.e("KrokoLiveTranscriber", "Error initializing Kroko recognizer", e)
            release()
            throw e
        }
    }

    /** Start an independent utterance without reloading immutable model weights. */
    fun reset() {
        stream?.release()
        stream = recognizer?.createStream()
        lastError = null
    }

    /**
     * Feed one frame of 16kHz mono float PCM samples (range -1.0..1.0).
      * PipelineRunner forwards all microphone frames, without VAD gating.
     */
    fun acceptWaveform(samples: FloatArray) {
        val s = stream ?: return
        val r = recognizer ?: return
        
        val startTime = System.currentTimeMillis()

        try {
            s.acceptWaveform(samples, sampleRate = SAMPLE_RATE)
            while (r.isReady(s)) {
                r.decode(s)
            }

            val text = r.getResult(s).text ?: ""
            val isEndpoint = r.isEndpoint(s)

            if (text.isNotEmpty() || isEndpoint) {
                android.util.Log.d("KrokoDebug", "Transcript characters=${text.length}, isEndpoint=$isEndpoint")
            }

            onPartialResult(text, isEndpoint)
            if (isEndpoint) {
                r.reset(s)
            }

            onLatencyMeasured(System.currentTimeMillis() - startTime)
        } catch (e: Throwable) {
            lastError = "English ASR decode failed: ${e.localizedMessage}"
            android.util.Log.e("KrokoLiveTranscriber", "Error in acceptWaveform decoding", e)
        }
    }

    fun release() {
        try { stream?.release() } catch (_: Throwable) {}
        try { recognizer?.release() } catch (_: Throwable) {}
        stream = null
        recognizer = null
    }
}
