package com.echoguard.evaluation

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.echoguard.acoustic.SpoofDetector
import com.echoguard.acoustic.VadGate
import com.echoguard.fusion.Action
import com.echoguard.fusion.FusionEngine
import com.echoguard.fusion.StreamSignal
import com.echoguard.fusion.SupervisorAgent
import com.echoguard.semantic.KrokoLiveTranscriber
import com.echoguard.semantic.IndicConformerLiveTranscriber
import com.echoguard.semantic.ScamClassifier
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Real-device tests. No microphone permission or call monitoring is used.
 * Dataset replay measures the compiled Android semantic/fusion backends with
 * supplied text, NOT microphone/ASR accuracy or the asynchronous full pipeline.
 */
@RunWith(AndroidJUnit4::class)
class DeviceEvaluationTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun nativeHindiAsrSmoke() {
        var decodedWindows = 0
        val asr = IndicConformerLiveTranscriber(
            assetManager = context.assets,
            onPartialResult = { _, _ -> },
            onLatencyMeasured = { decodedWindows++ },
            languageCode = "hi",
        )
        try {
            asr.init()
            repeat(64) { asr.acceptWaveform(FloatArray(512)) }
            assertTrue("Hindi ASR did not complete a native decode window", decodedWindows > 0)
        } finally { asr.release() }
    }

    @Test fun nativeModelSmoke() {
        val vad = VadGate(context.assets)
        val spoof = SpoofDetector(context)
        val classifier = ScamClassifier(context)
        try {
            assertTrue("VAD failed to load; inspect logcat", vad.available)
            assertTrue("AASIST failed to load; inspect logcat", spoof.available)
            assertTrue("MiniLM failed to load; rules-only fallback is not a model test", classifier.semanticAvailable)
            assertFalse("VAD silence inference failed", vad.isSpeech(FloatArray(512)).isSpeech)
            spoof.push(FloatArray(SpoofDetector.WINDOW_SAMPLES))
            val audio = spoof.score()
            assertTrue(audio.spoofScore.isFinite())
            assertTrue(audio.inferenceMs > 0L)
            val text = classifier.scamScore("Hello, your parcel arrives tomorrow afternoon.")
            assertTrue(text.semanticResult?.available == true)
            val joke = classifier.scamScore("Share your OTP. Just kidding, it's a prank.")
            assertTrue(joke.isJokeOverride)
            assertEquals(0.0, joke.score, 0.0)
        } finally {
            classifier.release()
            spoof.release()
            vad.release()
        }
        var callbacks = 0
        val asr = KrokoLiveTranscriber(context.assets, { _, _ -> callbacks++ })
        try {
            asr.init()
            repeat(8) { asr.acceptWaveform(FloatArray(512)) }
            assertTrue("English ASR did not decode any frames", callbacks > 0)
        } finally { asr.release() }
    }

    @Test fun evaluateTextDataset() {
        val args = InstrumentationRegistry.getArguments()
        val datasetName = args.getString("dataset")
        assumeTrue("No dataset requested; smoke tests only", datasetName != null)
        require(datasetName!!.matches(Regex("[A-Za-z0-9_-]+\\.json")))
        val folder = File(context.filesDir, "evaluation")
        val calls = JSONArray(File(folder, datasetName).readText())
        require(calls.length() > 0) { "Empty dataset" }
        val classifier = ScamClassifier(context)
        try {
            check(classifier.semanticAvailable) { "MiniLM unavailable; refusing to report fallback as full Android evaluation" }
            val rows = JSONArray()
            for (i in 0 until calls.length()) {
                val call = calls.getJSONObject(i)
                val fusion = FusionEngine()
                val agent = SupervisorAgent()
                val turns = call.getJSONArray("turns")
                require(turns.length() > 0) { "Call has no turns" }
                var running = ""
                var flagged = false
                var firstTurn = -1
                var peak = 0f
                var elapsedMs = 0.0
                var semanticScore = 0.0
                for (j in 0 until turns.length()) {
                    running += " " + turns.getJSONObject(j).getString("text")
                    val start = System.nanoTime()
                    val score = classifier.scamScore(running)
                    elapsedMs += (System.nanoTime() - start) / 1_000_000.0
                    semanticScore = score.score
                    if (score.isJokeOverride) fusion.reset() // Runner joke-override reset policy.
                    val result = fusion.combine(StreamSignal(0f), StreamSignal(score.score.toFloat(), score.explain()))
                    peak = maxOf(peak, result.riskScore)
                    val entry = agent.update(result)
                    if (entry != null && entry.recommendation in setOf(Action.WARN, Action.BLOCK)) {
                        if (!flagged) firstTurn = j
                        flagged = true
                    }
                }
                rows.put(JSONObject().put("id", call.getString("id"))
                    .put("label", call.getString("label")).put("language", call.getString("language"))
                    .put("flagged", flagged).put("first_warning_turn", firstTurn)
                    .put("max_text_only_risk", peak.toDouble()).put("last_scam_score", semanticScore)
                    .put("semantic_inference_ms", elapsedMs))
            }
            val output = JSONObject().put("schema_version", 1)
                .put("scope", "android_device_supplied_text_semantic_and_fusion")
                .put("limitations", "No ASR, microphone, acoustic input, or asynchronous PipelineRunner replay")
                .put("minilm_available", true).put("rows", rows)
                .put("device", android.os.Build.MODEL).put("sdk", android.os.Build.VERSION.SDK_INT)
            File(folder, "result.json").writeText(output.toString(2))
        } finally { classifier.release() }
    }
}
