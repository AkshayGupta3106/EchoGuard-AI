package com.echoguard.evaluation

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.echoguard.semantic.KrokoLiveTranscriber
import com.echoguard.semantic.IndicConformerLiveTranscriber
import com.echoguard.semantic.ScamClassifier
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.Assume.assumeTrue
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Offline native ASR replay, no microphone, network, calls, or real-time scheduling. */
@RunWith(AndroidJUnit4::class)
class AudioReplayTest {
    @Test fun replay() {
        val args = InstrumentationRegistry.getArguments()
        val name = args.getString("audio_run")
        assumeTrue("Explicit audio replay run required", name != null)
        require(name!!.matches(Regex("asr-[a-f0-9]+")))
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val folder = File(context.filesDir, "evaluation/$name")
        val manifest = JSONObject(File(folder, "manifest.json").readText())
        val clips = manifest.getJSONArray("clips")
        val rows = JSONArray()
        var partial = ""
        val finals = mutableListOf<String>()
        val callback: (String, Boolean) -> Unit = { text, final ->
            partial = text
            if (final) { if (text.isNotBlank()) finals.add(text); partial = "" }
        }
        val language = manifest.getString("language")
        val en = if (language == "en") KrokoLiveTranscriber(context.assets, callback) else null
        val hi = if (language == "hi") IndicConformerLiveTranscriber(context.assets, callback) else null
        require(en != null || hi != null)
        val classifier = ScamClassifier(context)
        try {
            check(classifier.semanticAvailable) { "MiniLM unavailable" }
            en?.init(); hi?.init()
            for (i in 0 until clips.length()) {
                val clip = clips.getJSONObject(i)
                val filename = clip.getString("pcm_file")
                require(filename.matches(Regex("clip-[0-9]+\\.pcm")))
                val bytes = File(folder, filename).readBytes()
                require(bytes.isNotEmpty() && bytes.size % 4 == 0)
                val floatBuffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                val samples = FloatArray(floatBuffer.remaining()).also { floatBuffer.get(it) }
                require(samples.all { it.isFinite() })
                finals.clear(); partial = ""
                en?.reset(); hi?.reset()
                val start = System.nanoTime()
                for (offset in samples.indices step 512) {
                    val chunk = samples.copyOfRange(offset, minOf(offset + 512, samples.size))
                    en?.acceptWaveform(chunk); hi?.acceptWaveform(chunk)
                }
                // Exercise the runtime's normal pause/endpoint path after end of file.
                repeat(100) { en?.acceptWaveform(FloatArray(512)); hi?.acceptWaveform(FloatArray(512)) }
                check(en?.lastError == null && hi?.lastError == null) { en?.lastError ?: hi?.lastError ?: "ASR failure" }
                val decodeMs = (System.nanoTime() - start) / 1_000_000.0
                val recognized = (finals + listOf(partial).filter { it.isNotBlank() }).joinToString(" ").trim()
                val score = classifier.scamScore(recognized)
                val row = JSONObject().put("id", clip.getString("id")).put("transcript", recognized)
                    .put("asr_ms", decodeMs).put("duration_seconds", samples.size / 16000.0)
                    .put("scam_score", score.score).put("joke_override", score.isJokeOverride)
                rows.put(row)
                // Checkpoint into this unique app-private run; host copies it only after success.
                File(folder, "result.json").writeText(JSONObject().put("scope", "offline_android_asr_replay")
                    .put("language", language).put("rows", rows).put("complete", i + 1 == clips.length()).toString())
            }
        } finally { en?.release(); hi?.release(); classifier.release() }
    }
}
