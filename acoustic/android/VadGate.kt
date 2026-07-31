/**
 * VadGate.kt
 * Owner: Person A (acoustic stream)
 *
 * Android port of vad_gate.py - same model (silero_vad.onnx), same I/O
 * contract, confirmed against the actual model file:
 *   inputs:  "input" [1, N] float32, "state" [2,1,128] float32,
 *            "sr" scalar int64
 *   outputs: "output" [1,1] float32 speech probability, "stateN" new state
 *
 * The recurrent state MUST be carried between calls (assign the returned
 * stateN back into the next call's state input) - this is the detail
 * that's easy to miss and silently degrades VAD accuracy if skipped.
 *
 * Setup: bundle silero_vad.onnx (from aasist_model/) as an Android asset,
 * add the ONNX Runtime Mobile dependency:
 *   implementation("com.microsoft.onnxruntime:onnxruntime-android:latest.release")
 */

package com.echoguard.acoustic

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer

class VadGate(
    assetManager: android.content.res.AssetManager,
    modelAssetPath: String = "models/silero_vad.onnx",
    private val threshold: Float = 0.5f,
) {
    companion object {
        const val SAMPLE_RATE = 16000L
        const val CHUNK_SAMPLES = 512  // 32ms @ 16kHz - match this model's training chunk size
    }

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private var state = FloatArray(2 * 1 * 128)  // zeroed recurrent state

    init {
        val modelBytes = assetManager.open(modelAssetPath).readBytes()
        session = env.createSession(modelBytes)
    }

    /** Call at the start of each new call/session - stale state from a
     * previous call skews the first few predictions. */
    fun reset() {
        state = FloatArray(2 * 1 * 128)
    }

    data class Result(val speechProb: Float, val isSpeech: Boolean)

    /** samples should be exactly CHUNK_SAMPLES (512) floats, 16kHz mono. */
    fun isSpeech(samples: FloatArray): Result {
        val padded = if (samples.size == CHUNK_SAMPLES) samples
                      else samples.copyOf(CHUNK_SAMPLES)  // zero-pads or truncates

        val inputTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(padded), longArrayOf(1, CHUNK_SAMPLES.toLong())
        )
        val stateTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(state), longArrayOf(2, 1, 128)
        )
        val srTensor = OnnxTensor.createTensor(
            env, LongBuffer.wrap(longArrayOf(SAMPLE_RATE)), longArrayOf()
        )

        val inputs = mapOf("input" to inputTensor, "state" to stateTensor, "sr" to srTensor)
        session.run(inputs).use { results ->
            val prob = (results.get(0).value as Array<FloatArray>)[0][0]
            val newState = results.get(1).value as Array<Array<FloatArray>>
            // Flatten [2,1,128] back into our stored FloatArray for the next call.
            state = FloatArray(2 * 128) { i -> newState[i / 128][0][i % 128] }
            return Result(speechProb = prob, isSpeech = prob >= threshold)
        }
    }

    fun release() {
        session.close()
    }
}
