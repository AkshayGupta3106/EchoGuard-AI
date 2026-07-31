/**
 * SpoofDetector.kt
 * Owner: Person A (acoustic stream)
 *
 * Android port of spoof_detector.py's RollingSpoofScorer, using the ONNX
 * export produced by export_onnx.py (aasist_l.onnx) - verified to produce
 * identical output to the original PyTorch checkpoint before export.
 *
 * Same model contract as the Python version:
 *   input:  "waveform" [1, 64600] float32, 16kHz mono raw audio
 *           (NOT a spectrogram - AASIST-L has its own SincConv front-end)
 *   output: "embedding" [1, 160], "logits" [1, 2]
 *           logits index 0 = spoof, index 1 = bonafide
 *
 * Setup: bundle aasist_l.onnx as an Android asset,
 *   implementation("com.microsoft.onnxruntime:onnxruntime-android:latest.release")
 */

package com.echoguard.acoustic

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.exp

class SpoofDetector(
    assetManager: android.content.res.AssetManager,
    modelAssetPath: String = "models/aasist_l.onnx",
) {
    companion object {
        const val SAMPLE_RATE = 16000
        const val WINDOW_SAMPLES = 64600  // ~4.04s - fixed by the model architecture
    }

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    // Rolling buffer of the most recent audio - a ring buffer would be more
    // efficient than this simple approach for a long-running call, but this
    // is the easiest correct version to start from.
    private val buffer = ArrayDeque<Float>()

    init {
        val modelBytes = assetManager.open(modelAssetPath).readBytes()
        session = env.createSession(modelBytes)
    }

    /** Feed small chunks continuously (post-VAD) from your audio capture loop. */
    fun push(samples: FloatArray) {
        for (s in samples) {
            if (buffer.size >= WINDOW_SAMPLES) buffer.removeFirst()
            buffer.addLast(s)
        }
    }

    data class Result(val spoofScore: Float, val bonafideScore: Float, val inferenceMs: Long)

    /** Call every 1-2 seconds. If less than ~4s has been buffered yet, the
     * buffer is tiled to fill the window - same behavior as the Python
     * reference implementation (matches the original AASIST eval recipe). */
    fun score(): Result {
        if (buffer.isEmpty()) return Result(0f, 1f, 0)

        val raw = buffer.toFloatArray()
        val windowed = if (raw.size >= WINDOW_SAMPLES) {
            raw.copyOfRange(0, WINDOW_SAMPLES)
        } else {
            FloatArray(WINDOW_SAMPLES) { i -> raw[i % raw.size] }  // tile to fill
        }

        val startTime = System.currentTimeMillis()
        val inputTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(windowed), longArrayOf(1, WINDOW_SAMPLES.toLong())
        )
        session.run(mapOf("waveform" to inputTensor)).use { results ->
            val logits = (results.get("logits").get().value as Array<FloatArray>)[0]
            val elapsed = System.currentTimeMillis() - startTime

            // softmax over the 2 logits
            val maxLogit = maxOf(logits[0], logits[1])
            val expSpoof = exp((logits[0] - maxLogit).toDouble())
            val expBona = exp((logits[1] - maxLogit).toDouble())
            val sum = expSpoof + expBona

            return Result(
                spoofScore = (expSpoof / sum).toFloat(),
                bonafideScore = (expBona / sum).toFloat(),
                inferenceMs = elapsed,
            )
        }
    }

    fun release() {
        session.close()
    }
}

/**
 * --- Day-1 test harness ---
 * This is the single most important test in the whole plan: run this
 * against your actual demo clone clip, not synthetic audio.
 *
 * val detector = SpoofDetector(assets)
 * val clipSamples: FloatArray = loadWavAsFloatArray(demoClipFile)  // 16kHz mono
 * detector.push(clipSamples)
 * val result = detector.score()
 * Log.d("AasistTest", "spoof=${result.spoofScore} bonafide=${result.bonafideScore} " +
 *       "took ${result.inferenceMs}ms")
 * // On this dev sandbox's CPU, the equivalent PyTorch inference took
 * // ~750-1000ms per 4s window - expect Android CPU inference to be in a
 * // similar ballpark or slower; confirm it's acceptable for a score that
 * // updates every 1-2s, and profile on your actual target phone.
 */
