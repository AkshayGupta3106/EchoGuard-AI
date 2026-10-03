/**
 * Silero ONNX speech-probability inference with carried recurrent state.
 * Model contract:
 *   inputs:  "input" [1, N] float32, "state" [2,1,128] float32,
 *            "sr" scalar int64
 *   outputs: "output" [1,1] float32 speech probability, "stateN" new state
 *
 * Each inference updates state from stateN; reset between audio sessions.
 *
 * Gradle stages models/silero_vad.onnx from the acoustic model directory.
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

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    val available: Boolean get() = session != null && env != null
    private var state = FloatArray(2 * 1 * 128)  // zeroed recurrent state

    init {
        try {
            env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(1)
                setInterOpNumThreads(1)
            }
            opts.use {
                val modelBytes = assetManager.open(modelAssetPath).use { it.readBytes() }
                session = env?.createSession(modelBytes, it)
            }
            android.util.Log.i("VadGate", "VAD ONNX session ready")
        } catch (e: Throwable) {
            android.util.Log.e("VadGate", "Failed to load ONNX session", e)
            env = null
            session = null
        }
    }

    /** Clear recurrent state between independent audio sessions. */
    fun reset() {
        state = FloatArray(2 * 1 * 128)
    }

    data class Result(val speechProb: Float, val isSpeech: Boolean)

    /** samples should be exactly CHUNK_SAMPLES (512) floats, 16kHz mono. */
    fun isSpeech(samples: FloatArray): Result {
        val currentEnv = env ?: return Result(1.0f, true) // Fail-open if VAD is down
        val currentSession = session ?: return Result(1.0f, true)

        return try {
            val padded = if (samples.size == CHUNK_SAMPLES) samples
                          else samples.copyOf(CHUNK_SAMPLES)  // zero-pads or truncates

            val inputTensor = OnnxTensor.createTensor(
                currentEnv, FloatBuffer.wrap(padded), longArrayOf(1, CHUNK_SAMPLES.toLong())
            )
            inputTensor.use {
                val stateTensor = OnnxTensor.createTensor(
                    currentEnv, FloatBuffer.wrap(state), longArrayOf(2, 1, 128)
                )
                stateTensor.use {
                    val srTensor = OnnxTensor.createTensor(
                        currentEnv, LongBuffer.wrap(longArrayOf(SAMPLE_RATE)), longArrayOf()
                    )
                    srTensor.use {
                        val inputs = mapOf("input" to inputTensor, "state" to stateTensor, "sr" to srTensor)
                        currentSession.run(inputs).use { results ->
                            @Suppress("UNCHECKED_CAST")
                            val prob = (results.get(0).value as Array<FloatArray>)[0][0]
                            @Suppress("UNCHECKED_CAST")
                            val newState = results.get(1).value as Array<Array<FloatArray>>
                            // Flatten [2,1,128] back into our stored FloatArray for the next call.
                            state = FloatArray(2 * 128) { i -> newState[i / 128][0][i % 128] }
                            Result(speechProb = prob, isSpeech = prob >= threshold)
                        }
                    }
                }
            }
        } catch (e: Throwable) {
            android.util.Log.e("VadGate", "isSpeech inference failed", e)
            Result(1.0f, true)
        }
    }

    fun release() {
        try {
            session?.close()
        } catch (_: Throwable) {}
        session = null
        env = null
    }
}
