/**
 * KrokoLiveTranscriber.kt
 * Owner: Person B (semantic stream - live path)
 *
 * Wraps sherpa-onnx's streaming Kroko for the LIVE transcription path.
 * This is the load-bearing Day-1 test: measure real latency on an actual
 * phone before anything else gets built on top of this class.
 *
 * Setup you need to do first (not code - one-time project setup):
 *   1. Add the sherpa-onnx Android AAR / build sherpa-onnx for Android per
 *      the official k2-fsa/sherpa-onnx build instructions.
 *   2. Download a streaming Kroko model (encoder/decoder/joiner .onnx +
 *      tokens.txt) - bundle these under app assets, they're the one-time
 *      network dependency, not a runtime one.
 *
 * This class does NOT do audio capture itself - feed it 16kHz mono PCM
 * frames from AudioRecord (ideally gated by the VAD, once Person A's VAD
 * module is wired in - for Day 1 testing, feeding it raw mic audio directly
 * is fine).
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

    companion object {
        // High-performance int8 quantized models
        private const val ENCODER = "kroko-128l/encoder.int8.onnx"
        private const val DECODER = "kroko-128l/decoder.int8.onnx"
        private const val JOINER = "kroko-128l/joiner.int8.onnx"
        private const val TOKENS = "kroko-128l/tokens.txt"
        // Contextual biasing - see generate_hotwords.py. Boosts recognition
        // of short acronyms/brand names (OTP, CVV, AnyDesk, ...) that a
        // general-purpose ASR model doesn't have strong priors on. This is
        // the actual fix for "ASR mishears the scam keyword" - biasing the
        // model itself, not just pattern-matching around its mistakes
        // downstream (see ScamClassifier.kt's fuzzy matching for the
        // defense-in-depth layer on top of this).
        private const val HOTWORDS_FILE = "Kroko/hotwords.txt"
        private const val HOTWORDS_SCORE = 2.0f  // default score for terms in the file without their own
        const val SAMPLE_RATE = 16000
    }

    /**
     * Throws a clear, specific exception if any required model asset is
     * missing - this is the fix for a real bug hit during testing: sherpa-onnx's
     * native layer doesn't reliably surface a Kotlin-catchable error when an
     * asset is missing, so a missing encoder.onnx was silently producing
     * garbage transcripts instead of a clear crash. Fail loud, fail early.
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
                "Check app/src/main/assets/Kroko/ - all of encoder.onnx, decoder.onnx, " +
                "joiner.onnx, and tokens.txt must be present, from the SAME model export " +
                "(matching precision - don't mix an int8 file with fp32 files)."
            )
        }
        // Hotwords are a soft dependency - missing this file degrades
        // accuracy on jargon/acronyms but shouldn't block startup entirely,
        // unlike the four files above which the recognizer can't run without.
        try {
            assetManager.open(HOTWORDS_FILE).close()
        } catch (e: java.io.IOException) {
            android.util.Log.w("KrokoLiveTranscriber",
                "$HOTWORDS_FILE not found - contextual biasing disabled, " +
                "OTP/CVV/etc. recognition accuracy will be lower than with it. " +
                "Run generate_hotwords.py and bundle the output to fix.")
        }
    }

    fun init() {
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

    /**
     * Feed one frame of 16kHz mono float PCM samples (range -1.0..1.0).
     * Call this continuously from your AudioRecord callback, ideally only
     * when the VAD gate says someone is speaking.
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
                android.util.Log.d("KrokoDebug", "Kroko text: '$text', isEndpoint=$isEndpoint")
            }

            onPartialResult(text, isEndpoint)
            if (isEndpoint) {
                r.reset(s)
            }

            onLatencyMeasured(System.currentTimeMillis() - startTime)
        } catch (e: Throwable) {
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

/**
 * --- Day-1 test harness ---
 * Minimal usage sketch for the latency test. Wire this to a button in a
 * throwaway test Activity - don't wait until the full app shell exists to
 * run this test.
 *
 * val transcriber = KrokoLiveTranscriber(
 *     assetManager = assets,
 *     onPartialResult = { text, isFinal -> Log.d("KrokoTest", "text=$text final=$isFinal") },
 *     onLatencyMeasured = { ms -> Log.d("KrokoTest", "frame latency=${ms}ms") }
 * )
 * transcriber.init()
 * // ... feed it real AudioRecord frames, watch logcat for the latency line
 */
