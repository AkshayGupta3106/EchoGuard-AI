/**
 * CallMonitorService.kt
 * Owner: together (Day 3)
 *
 * Two separate jobs, deliberately kept independent - see app/README.md for
 * the full reasoning:
 *
 *   1. Detect when a call is active - via TelephonyCallback (API 31+) /
 *      PhoneStateListener (fallback). This part is standard, reliable,
 *      documented Android API, no special tricks.
 *
 *   2. Once a call is active, record via the ordinary MICROPHONE (not any
 *      call-audio API) - this only works well if the user has switched the
 *      call to speaker, which is why MainActivity needs to prompt for that
 *      clearly rather than assuming it silently.
 *
 * NOT run against a real call - the call-state detection code follows
 * documented APIs and should work as-is; the audio capture loop's actual
 * quality against real speakerphone audio hasn't been verified on-device.
 */

package com.echoguard.service

import android.app.*
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import android.telephony.TelephonyCallback
import android.telephony.TelephonyManager
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import com.echoguard.pipeline.PipelineRunner
import kotlinx.coroutines.*

class CallMonitorService : Service() {

    companion object {
        const val CHANNEL_ID = "call_monitor_channel"
        const val NOTIFICATION_ID = 1

        const val SAMPLE_RATE = 16000
        const val CHUNK_SAMPLES = 512  // matches VadGate.CHUNK_SAMPLES (32ms @ 16kHz)
    }

    private lateinit var pipelineRunner: PipelineRunner
    private var audioRecord: AudioRecord? = null
    private var captureJob: Job? = null
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private lateinit var telephonyManager: TelephonyManager

    override fun onCreate() {
        super.onCreate()
        pipelineRunner = PipelineRunner(this)
        createNotificationChannel()
        telephonyManager = getSystemService(TELEPHONY_SERVICE) as TelephonyManager
        registerCallStateListener()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTIFICATION_ID, buildNotification("Watching for calls"), android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
        } else {
            startForeground(NOTIFICATION_ID, buildNotification("Watching for calls"))
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // --- 1. Call-state detection (standard, reliable) --------------------

    // Keep strong references to prevent them from being garbage-collected
    // since TelephonyManager only holds weak references to listeners.
    private var telephonyCallback: Any? = null
    private var phoneStateListener: Any? = null

    private fun registerCallStateListener() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val callback = object : TelephonyCallback(), TelephonyCallback.CallStateListener {
                override fun onCallStateChanged(state: Int) = handleCallState(state)
            }
            telephonyCallback = callback
            telephonyManager.registerTelephonyCallback(mainExecutor, callback)
        } else {
            @Suppress("DEPRECATION")
            val listener = object : android.telephony.PhoneStateListener() {
                @Deprecated("Deprecated in Java")
                override fun onCallStateChanged(state: Int, phoneNumber: String?) = handleCallState(state)
            }
            phoneStateListener = listener
            @Suppress("DEPRECATION")
            telephonyManager.listen(listener, android.telephony.PhoneStateListener.LISTEN_CALL_STATE)
        }
    }

    private fun handleCallState(state: Int) {
        when (state) {
            TelephonyManager.CALL_STATE_OFFHOOK -> onCallActive()
            TelephonyManager.CALL_STATE_IDLE -> onCallEnded()
            // CALL_STATE_RINGING: not acted on here - could show a
            // "get ready" notification, deliberately kept minimal.
        }
    }

    private fun onCallActive() {
        try {
            updateNotification("Call detected \u2014 switch to speaker to enable protection")
            pipelineRunner.start(assets)
            startAudioCapture()
        } catch (e: Throwable) {
            android.util.Log.e("CallMonitorService", "Error starting call active monitoring", e)
        }
    }

    private fun onCallEnded() {
        try {
            stopAudioCapture()
            pipelineRunner.stop()
            updateNotification("Watching for calls")
        } catch (e: Throwable) {
            android.util.Log.e("CallMonitorService", "Error stopping call active monitoring", e)
        }
    }

    // --- 2. Microphone capture loop (the documented speakerphone approach) --

    private fun startAudioCapture() {
        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            return  // permission not granted - MainActivity is responsible
                    // for requesting this before the service is even started
        }

        val minBufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,  // PCM_16BIT is universally supported; PCM_FLOAT is not guaranteed
        )
        if (minBufferSize == AudioRecord.ERROR || minBufferSize == AudioRecord.ERROR_BAD_VALUE) {
            android.util.Log.e("CallMonitorService", "getMinBufferSize returned error $minBufferSize")
            return
        }

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,  // better sharing behavior than MIC during calls
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBufferSize, CHUNK_SAMPLES * 4),
        )

        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            android.util.Log.e("CallMonitorService", "AudioRecord failed to initialize")
            audioRecord?.release()
            audioRecord = null
            return
        }

        audioRecord?.startRecording()

        captureJob = serviceScope.launch {
            val shortBuffer = ShortArray(CHUNK_SAMPLES)
            val floatBuffer = FloatArray(CHUNK_SAMPLES)
            while (isActive) {
                val read = audioRecord?.read(shortBuffer, 0, CHUNK_SAMPLES) ?: 0
                if (read > 0) {
                    // Convert 16-bit PCM to float [-1.0, 1.0] for the pipeline
                    for (i in 0 until read) floatBuffer[i] = shortBuffer[i] / 32768.0f
                    pipelineRunner.onAudioChunk(floatBuffer.copyOf(read))
                }
            }
        }
    }

    private fun stopAudioCapture() {
        captureJob?.cancel()
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }

    override fun onDestroy() {
        stopAudioCapture()
        serviceScope.cancel()
        super.onDestroy()
    }

    // --- Notification plumbing (foreground service requirement) ----------

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID, "Call monitoring", NotificationManager.IMPORTANCE_LOW
        )
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun buildNotification(text: String): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("EchoGuard-AI")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_lock_silent_mode_off)  // placeholder - swap for a real icon
            .setOngoing(true)
            .build()

    private fun updateNotification(text: String) {
        getSystemService(NotificationManager::class.java)
            .notify(NOTIFICATION_ID, buildNotification(text))
    }
}
