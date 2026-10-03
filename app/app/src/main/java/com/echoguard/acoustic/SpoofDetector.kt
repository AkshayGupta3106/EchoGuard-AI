/**
 * Rolling on-device AASIST-L ONNX inference with the packaged INT8 asset.
 * Model contract:
 *   input:  "waveform" [1, 64600] float32, 16kHz mono raw audio
 *           (NOT a spectrogram - AASIST-L has its own SincConv front-end)
 *   output: "embedding" [1, 160], "logits" [1, 2]
 *           logits index 0 = spoof, index 1 = bonafide
 *
 * Asset: models/aasist_l.onnx. Runtime dependency is configured in build.gradle.kts.
 */

package com.echoguard.acoustic

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.exp

class SpoofDetector(
    context: Context,
    modelAssetPath: String = "models/aasist_l.onnx",
) {
    companion object {
        const val WINDOW_SAMPLES = 64600  // ~4.04s - fixed by the model architecture

        fun clearCache(context: Context) {
            try {
                File(context.cacheDir, "aasist_l.onnx").delete()
                File(context.cacheDir, "aasist_l.onnx.data").delete()
            } catch (_: Throwable) {}
        }
    }

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    val available: Boolean get() = session != null && env != null
    // Memory-efficient ring buffer of raw audio samples
    private val buffer = FloatArray(WINDOW_SAMPLES)
    private var bufferPos = 0
    private var bufferFilled = 0
    private val lock = Any()

    init {
        val modelFile = File(context.cacheDir, "aasist_l.onnx")
        val dataFile = File(context.cacheDir, "aasist_l.onnx.data")
        
        try {
            val runtime = Runtime.getRuntime()
            val availMb = (runtime.maxMemory() - (runtime.totalMemory() - runtime.freeMemory())) / 1024 / 1024
            if (availMb < 80) {
                android.util.Log.w("SpoofDetector", "Low available memory (${availMb}MB). Skipping AASIST-L load to prevent LMK crash.")
                env = null
                session = null
            } else {
                env = OrtEnvironment.getEnvironment()
                val opts = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(1)
                    setInterOpNumThreads(1)
                    setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                }
                opts.use {
                    copyModelAsset(context, modelAssetPath, "aasist_l.onnx")
                    session = env?.createSession(modelFile.absolutePath, it)
                }
                android.util.Log.i("SpoofDetector", "AASIST ONNX session ready")
            }
        } catch (e: Throwable) {
            android.util.Log.e("SpoofDetector", "Failed to load ONNX session", e)
            try {
                modelFile.delete()
                dataFile.delete()
            } catch (_: Throwable) {}
            env = null
            session = null
        }
    }

    /** Append waveform chunks to the rolling acoustic buffer. */
    fun push(samples: FloatArray) = synchronized(lock) {
        for (s in samples) {
            buffer[bufferPos] = s
            bufferPos = (bufferPos + 1) % WINDOW_SAMPLES
            if (bufferFilled < WINDOW_SAMPLES) bufferFilled++
        }
    }

    data class Result(val spoofScore: Float, val bonafideScore: Float, val inferenceMs: Long)

    /** Score the rolling buffer on demand, tiling short input to the model window. */
    fun score(): Result {
        val currentEnv = env ?: return Result(0f, 1f, 0)
        val currentSession = session ?: return Result(0f, 1f, 0)

        // Snapshot under the buffer lock so inference does not block audio appends.
        val raw: FloatArray = synchronized(lock) {
            if (bufferFilled == 0) return Result(0f, 1f, 0)
            
            val snapshot = FloatArray(bufferFilled)
            if (bufferFilled < WINDOW_SAMPLES) {
                // Buffer not yet full, copy from start to current position
                System.arraycopy(buffer, 0, snapshot, 0, bufferFilled)
            } else {
                // Buffer full and wrapped, copy from current position to end, then start to current position
                System.arraycopy(buffer, bufferPos, snapshot, 0, WINDOW_SAMPLES - bufferPos)
                System.arraycopy(buffer, 0, snapshot, WINDOW_SAMPLES - bufferPos, bufferPos)
            }
            snapshot
        }

        val windowed = when {
            raw.size >= WINDOW_SAMPLES -> raw // Should be exactly WINDOW_SAMPLES if filled
            raw.isEmpty() -> return Result(0f, 1f, 0)
            else -> {
                // Tile the short buffer to fill WINDOW_SAMPLES.
                FloatArray(WINDOW_SAMPLES) { i -> raw[i % raw.size] }
            }
        }

        return try {
            val directBuffer = java.nio.ByteBuffer.allocateDirect(windowed.size * 4)
                .order(java.nio.ByteOrder.nativeOrder())
                .asFloatBuffer()
            directBuffer.put(windowed)
            directBuffer.rewind()

            val inputTensor = OnnxTensor.createTensor(
                currentEnv,
                directBuffer,
                longArrayOf(1, WINDOW_SAMPLES.toLong()),
            )
            val startTime = System.currentTimeMillis()
            inputTensor.use { tensor ->
                currentSession.run(mapOf("waveform" to tensor)).use { results ->
                    val logitsValue = results["logits"]
                    if (!logitsValue.isPresent) return Result(0f, 1f, 0)
                val logits = (logitsValue.get().value as Array<*>)[0] as FloatArray
                val elapsed = System.currentTimeMillis() - startTime

                // softmax over the 2 logits
                val maxLogit = maxOf(logits[0], logits[1])
                val expSpoof = exp((logits[0] - maxLogit).toDouble())
                val expBona  = exp((logits[1] - maxLogit).toDouble())
                val sum = expSpoof + expBona

                    Result(
                        spoofScore    = (expSpoof / sum).toFloat(),
                        bonafideScore = (expBona  / sum).toFloat(),
                        inferenceMs   = elapsed,
                    )
                }
            }
        } catch (e: Throwable) {
            android.util.Log.e("SpoofDetector", "ONNX inference failed", e)
            Result(0f, 1f, 0)
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
