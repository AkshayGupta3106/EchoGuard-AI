/**
 * IndicConformerLiveTranscriber.kt
 * Owner: Person B (semantic stream - live path)
 *
 * Replaces ZipformerLiveTranscriber as the live transcription engine.
 * Uses AI4Bharat's IndicConformer (NeMo CTC) via sherpa-onnx, giving us
 * real Hindi/Hinglish recognition instead of the English-only Zipformer.
 *
 * Why IndicConformer instead of Zipformer:
 *   - Indian scam callers switch between English and Hindi mid-sentence
 *     (code-switching). Zipformer's English-only vocabulary either drops
 *     Hindi words or hallucinates English homophones for them - both
 *     outcomes miss scam signals.
 *   - IndicConformer is trained on all 22 official Indian languages and
 *     handles Hinglish code-switching natively.
 *   - The sherpa-onnx API difference is minimal: OnlineNeMoCtcModelConfig
 *     instead of OnlineTransducerModelConfig (no separate decoder/joiner).
 *
 * Model files required (bundle as app assets):
 *   assets/indicconformer-hi/model.int8.onnx   (~150 MB)
 *   assets/indicconformer-hi/tokens.txt
 *
 * Download from:
 *   https://huggingface.co/parismitaglobalsolutions/indicconformer-sherpa-onnx
 *   → hi/ folder
 *
 * Same interface contract as ZipformerLiveTranscriber:
 *   - init() to load model
 *   - acceptWaveform(FloatArray) to feed 16kHz mono PCM
 *   - release() to clean up
 *   PipelineRunner.kt drops in the new class with zero other changes.
 */

package com.echoguard.semantic

import android.content.res.AssetManager
import android.util.Log
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineNeMoCtcModelConfig
import com.k2fsa.sherpa.onnx.OnlineRecognizer
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineStream

class IndicConformerLiveTranscriber(
    private val assetManager: AssetManager,
    private val onPartialResult: (text: String, isFinal: Boolean) -> Unit,
    private val onLatencyMeasured: (millis: Long) -> Unit = {},
    /** Language code. "hi" = Hindi (default). Change to "ta", "te", "bn" etc.
     *  to switch to other IndicConformer language models. The chosen code must
     *  match the model subfolder name under assets/indicconformer-{lang}/. */
    private val languageCode: String = "hi",
) {
    private var recognizer: OnlineRecognizer? = null
    private var stream: OnlineStream? = null

    companion object {
        private const val TAG = "IndicConformerDebug"
        const val SAMPLE_RATE = 16000
    }

    /** Asset paths derived from the chosen language code. */
    private val modelDir get() = "indicconformer-$languageCode"
    private val modelPath get() = "$modelDir/model.int8.onnx"
    private val tokensPath get() = "$modelDir/tokens.txt"
    private val hotwordsPath get() = "indicconformer/hotwords.txt"

    /**
     * Fails loudly if the model assets are missing — a missing model.onnx
     * causes sherpa-onnx's native layer to produce garbage silently, which is
     * harder to debug than a clear exception here.
     */
    private fun validateAssetsExist() {
        val required = listOf(modelPath, tokensPath)
        val missing = required.filterNot { path ->
            try { assetManager.open(path).close(); true }
            catch (e: java.io.IOException) { false }
        }
        if (missing.isNotEmpty()) {
            throw IllegalStateException(
                "IndicConformerLiveTranscriber: missing required asset(s): $missing.\n" +
                "Download from https://huggingface.co/parismitaglobalsolutions/indicconformer-sherpa-onnx\n" +
                "and place under app/src/main/assets/$modelDir/"
            )
        }

        // Hotwords are a soft dependency — missing degrades OTP/CVV accuracy
        // but shouldn't block startup.
        try {
            assetManager.open(hotwordsPath).close()
        } catch (e: java.io.IOException) {
            Log.w(TAG,
                "$hotwordsPath not found — contextual biasing for OTP/CVV/AnyDesk disabled. " +
                "Place indicconformer/hotwords.txt in assets to re-enable.")
        }
    }

    fun init() {
        validateAssetsExist()

        val hotwordsAvailable = try {
            assetManager.open(hotwordsPath).close(); true
        } catch (e: java.io.IOException) { false }

        // IndicConformer is a NeMo CTC model — no separate decoder/joiner
        // unlike Zipformer's Transducer. OnlineNeMoCtcModelConfig takes only
        // the single .onnx model path.
        val neMoCtcConfig = OnlineNeMoCtcModelConfig(model = modelPath)

        val modelConfig = OnlineModelConfig(
            neMoCtc = neMoCtcConfig,
            tokens = tokensPath,
            numThreads = 4,
            provider = "cpu",
        )

        val featConfig = FeatureConfig(
            sampleRate = SAMPLE_RATE,
            featureDim = 80,
        )

        // modified_beam_search enables hotword/contextual biasing.
        // greedy_search silently ignores hotwordsFile — don't use it here.
        val config = OnlineRecognizerConfig(
            featConfig = featConfig,
            modelConfig = modelConfig,
            decodingMethod = "modified_beam_search",
            maxActivePaths = 4,           // balanced for mobile CPU
            enableEndpoint = true,
            rule1MinTrailingSilence = 2.4f,  // seconds of silence → endpoint
            rule2MinTrailingSilence = 1.2f,  // stricter rule for mid-sentence pauses
            rule3MinUtteranceLength = 20.0f, // force endpoint for very long utterances
            hotwordsFile = if (hotwordsAvailable) hotwordsPath else "",
            hotwordsScore = 2.0f,
        )

        recognizer = OnlineRecognizer(assetManager, config)
        stream = recognizer?.createStream()

        Log.i(TAG, "IndicConformer ($languageCode) initialized. " +
              "Hotwords: ${if (hotwordsAvailable) "enabled" else "disabled"}.")
    }

    /**
     * Feed one frame of 16kHz mono float32 PCM (range −1.0..1.0).
     * Call continuously from the AudioRecord loop, gated by VAD.
     */
    fun acceptWaveform(samples: FloatArray) {
        val s = stream ?: return
        val r = recognizer ?: return

        val t0 = System.currentTimeMillis()

        s.acceptWaveform(samples, sampleRate = SAMPLE_RATE)
        while (r.isReady(s)) {
            r.decode(s)
        }

        val text = r.getResult(s).text ?: ""
        val isEndpoint = r.isEndpoint(s)

        if (text.isNotEmpty() || isEndpoint) {
            Log.d(TAG, "text='$text'  endpoint=$isEndpoint  lang=$languageCode")
        }

        onPartialResult(text, isEndpoint)
        if (isEndpoint) {
            r.reset(s)
        }

        onLatencyMeasured(System.currentTimeMillis() - t0)
    }

    fun release() {
        stream?.release()
        recognizer?.release()
        stream = null
        recognizer = null
    }
}

/**
 * --- Day-1 latency test harness ---
 *
 * val transcriber = IndicConformerLiveTranscriber(
 *     assetManager = assets,
 *     onPartialResult = { text, isFinal -> Log.d("ICTest", "text=$text final=$isFinal") },
 *     onLatencyMeasured = { ms -> Log.d("ICTest", "frame latency=${ms}ms") },
 *     languageCode = "hi"
 * )
 * transcriber.init()
 * // Feed AudioRecord frames (16kHz mono FloatArray), watch logcat for latency lines.
 * // Target: comfortably < frame duration (~100ms per 2000-sample chunk at 16kHz).
 */
