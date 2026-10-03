/**
 * Coordinates audio, English Kroko or Hindi IndicConformer ASR, scam scoring,
 * fusion, supervisor recommendations, and observable UI state.
 * Incoming chunks are sent to VAD, AASIST, and the selected ASR engine; the
 * VAD result does not gate downstream audio. Scoring runs every 45 chunks.
 * Accumulated transcript text is supplied to ScamClassifier. Peak risk is
 * Peak risk persists across scoring updates and resets when an intentional joke override fires.
 */

package com.echoguard.pipeline

import android.content.Context
import android.content.res.AssetManager
import com.echoguard.acoustic.SpoofDetector
import com.echoguard.acoustic.VadGate
import com.echoguard.semantic.ScamClassifier
import com.echoguard.semantic.ScamScoreResult
import com.echoguard.semantic.IndicConformerLiveTranscriber
import com.echoguard.fusion.FusionEngine
import com.echoguard.fusion.StreamSignal
import com.echoguard.fusion.SupervisorAgent
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.cancelChildren
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write

class PipelineRunner(
    private val context: Context,
    private val vadThreshold: Float = 0.05f, // VAD probability threshold; not a downstream audio gate.
) {

    companion object {
        fun clearCache(context: Context) {
            try {
                SpoofDetector.clearCache(context)
                com.echoguard.semantic.SemanticScorer.clearCache(context)
            } catch (_: Throwable) {}
        }
    }

    private var vadGate: VadGate? = null
    private var spoofDetector: SpoofDetector? = null
    private var scamClassifier: ScamClassifier? = null
    private var fusionEngine: FusionEngine? = null
    private val supervisorAgent = SupervisorAgent()

    private var krokoTranscriber: com.echoguard.semantic.KrokoLiveTranscriber? = null
    private var transcriber: IndicConformerLiveTranscriber? = null
    private var finalizedTranscript: String = ""
    private var currentPartial: String = ""
    private val transcriptLock = Any()

    private val _uiState = MutableStateFlow(PipelineUiState())
    val uiState: StateFlow<PipelineUiState> = _uiState

    private val scoringScope = kotlinx.coroutines.CoroutineScope(
        kotlinx.coroutines.Dispatchers.Default + kotlinx.coroutines.SupervisorJob(),
    )
    private var isScoring = java.util.concurrent.atomic.AtomicBoolean(false)
    @Volatile private var isRunning = false
    private val lifecycleLock = ReentrantReadWriteLock(true)
    private val lifecycleVersion = java.util.concurrent.atomic.AtomicLong(0)

    fun start(assetManager: AssetManager) = lifecycleLock.write {
        android.util.Log.i("PipelineRunner", "Starting pipeline...")
        // Request scoring cancellation and release the current model sessions.
        stopInternalOnly()
        _uiState.update { PipelineUiState(uiLanguage = it.uiLanguage, isInitializing = true) }

        try {
            android.util.Log.i("PipelineRunner", "Initializing VadGate...")
            vadGate = VadGate(assetManager, threshold = vadThreshold)
            Thread.sleep(200)

            android.util.Log.i("PipelineRunner", "Initializing SpoofDetector...")
            spoofDetector = SpoofDetector(context)
            Thread.sleep(200)

            android.util.Log.i("PipelineRunner", "Initializing ScamClassifier...")
            scamClassifier = ScamClassifier(context)
            Thread.sleep(200)

            android.util.Log.i("PipelineRunner", "Initializing FusionEngine...")
            fusionEngine = FusionEngine()
            Thread.sleep(100)

            // Memory Safeguard: Check available RAM before loading the largest model
            val runtime = Runtime.getRuntime()
            val availableMemoryMb = (runtime.maxMemory() - (runtime.totalMemory() - runtime.freeMemory())) / 1024 / 1024
            android.util.Log.i("PipelineRunner", "Available memory before Transcriber: ${availableMemoryMb}MB")
            
            if (availableMemoryMb < 50) {
                throw IllegalStateException("Low memory ($availableMemoryMb MB available). Please close background apps.")
            }

            vadGate?.reset()
            supervisorAgent.reset()
            finalizedTranscript = ""
            currentPartial = ""
            maxRiskSoFar = 0f
            chunkCount = 0
            synchronized(stateLock) {
                lastSpoofScore = 0f
                lastSpoofExplain = ""
                lastScamScore = 0.0
                lastScamExplain = ""
            }

            android.util.Log.i("PipelineRunner", "Initializing ASR...")
            val normalizeTranscript = { text: String ->
                var t = text
                t = t.replace(Regex("(?i)\\b([A-Za-z])[\\s.]+([A-Za-z])[\\s.]+([A-Za-z])\\b"), "$1$2$3")
                t = t.replace(Regex("(?i)\\b([A-Za-z])[\\s.]+([A-Za-z])\\b"), "$1$2")
                t = t.replace(Regex("(?i)\\bany\\s+desk\\b"), "AnyDesk")
                t = t.replace(Regex("(?i)\\bteam\\s+viewer\\b"), "TeamViewer")
                t
            }
            if (_uiState.value.uiLanguage == AppLanguage.HINDI) {
                try {
                    transcriber = IndicConformerLiveTranscriber(
                        assetManager = assetManager,
                        languageCode = "hi",
                        onPartialResult = { rawText, isFinal ->
                            val text = normalizeTranscript(rawText)
                            synchronized(transcriptLock) {
                                if (isFinal) {
                                    finalizedTranscript += " $text"
                                    currentPartial = ""
                                } else {
                                    currentPartial = text
                                }
                            }
                        },
                        onLatencyMeasured = { ms ->
                            if (ms > 200) android.util.Log.w("PipelineRunner", "Hindi ASR frame latency ${ms}ms")
                        }
                    )
                    transcriber?.init()
                    android.util.Log.i("PipelineRunner", "IndicConformer Hindi ASR initialized successfully!")
                } catch (e: Throwable) {
                    android.util.Log.e("PipelineRunner", "IndicConformer failed", e)
                    throw e
                }
            } else {
                try {
                    krokoTranscriber = com.echoguard.semantic.KrokoLiveTranscriber(
                        assetManager = assetManager,
                        onPartialResult = { rawText, isFinal ->
                            val text = normalizeTranscript(rawText)
                            synchronized(transcriptLock) {
                                if (isFinal) {
                                    finalizedTranscript += " $text"
                                    currentPartial = ""
                                } else {
                                    currentPartial = text
                                }
                            }
                        }
                    )
                    krokoTranscriber?.init()
                    android.util.Log.i("PipelineRunner", "Zipformer English ASR initialized successfully!")
                } catch (e2: Throwable) {
                    android.util.Log.e("PipelineRunner", "Zipformer English ASR failed", e2)
                    throw e2
                }
            }

            isRunning = true
            val unavailable = buildList {
                if (vadGate?.available != true) add("VAD")
                if (spoofDetector?.available != true) add("voice spoof detection")
                if (scamClassifier?.semanticAvailable != true) add("MiniLM (text scoring is rules-only)")
            }
            _uiState.update { it.copy(
                status = MonitorStatus.Monitoring,
                isInitializing = false,
                warningMessage = unavailable.takeIf { models -> models.isNotEmpty() }?.joinToString(prefix = "Unavailable: "),
            ) }
        } catch (e: Throwable) {
            android.util.Log.e("PipelineRunner", "Error during model initialization", e)
            stopInternalOnly()
            currentPartial = "[Error initializing AI models: ${e.localizedMessage}]"
            _uiState.update { it.copy(liveTranscript = "", errorMessage = currentPartial, status = MonitorStatus.Idle, isInitializing = false) }
        }
    }

    fun setLanguage(language: AppLanguage) = lifecycleLock.write {
        _uiState.update { it.copy(uiLanguage = language) }
    }

    fun reportCaptureError(message: String) {
        _uiState.update { it.copy(errorMessage = message) }
    }

    private var chunkCount = 0
    @Volatile private var lastSpoofScore = 0f
    @Volatile private var lastSpoofExplain = ""
    @Volatile private var lastScamScore = 0.0
    @Volatile private var lastScamExplain = ""
    @Volatile private var maxRiskSoFar = 0f
    private val stateLock = Any()

    fun onAudioChunk(samples: FloatArray) = lifecycleLock.read {
        if (!isRunning) return@read
        try {
            onAudioChunkInternal(samples)
        } catch (e: Throwable) {
            android.util.Log.e("PipelineRunner", "Error processing audio chunk", e)
        }
    }

    private fun onAudioChunkInternal(samples: FloatArray) {
        if (!isRunning) return
        vadGate?.isSpeech(samples)

        spoofDetector?.push(samples)
        krokoTranscriber?.acceptWaveform(samples)
        transcriber?.acceptWaveform(samples)

        chunkCount++

        if ((chunkCount % 45 == 0) && isScoring.compareAndSet(false, true)) {
            val version = lifecycleVersion.get()
            val transcriptSnapshot = synchronized(transcriptLock) { 
                "$finalizedTranscript $currentPartial".trim() 
            }
            
            scoringScope.launch {
                try {
                    lifecycleLock.read {
                        if (!isRunning || version != lifecycleVersion.get()) return@read
                        val spoofResult = spoofDetector?.score() ?: SpoofDetector.Result(0f, 1f, 0)
                        val scamResult = scamClassifier?.scamScore(transcriptSnapshot)
                            ?: ScamScoreResult(0.0, emptyList(), null)

                        synchronized(stateLock) {
                            if (scamResult.isJokeOverride) {
                                maxRiskSoFar = 0f
                                fusionEngine?.reset()
                            }
                            lastSpoofScore = spoofResult.spoofScore
                            lastSpoofExplain = if (spoofResult.spoofScore > 0.5f) "likely AI-generated voice" else "no spoof detected"
                            lastScamScore = scamResult.score
                            lastScamExplain = scamResult.explain()
                            if (isRunning) updateUiState(transcriptSnapshot)
                        }
                    }
                } catch (e: Throwable) {
                    android.util.Log.e("PipelineRunner", "Error scoring chunk", e)
                } finally {
                    if (version == lifecycleVersion.get()) isScoring.set(false)
                }
            }
        } else {
            updateUiState(null)
        }
    }

    private fun updateUiState(transcriptOverride: String?) = synchronized(stateLock) {
        if (!isRunning) return@synchronized
        val spoofSignal = StreamSignal(score = lastSpoofScore, explain = lastSpoofExplain)
        val scamSignal = StreamSignal(score = lastScamScore.toFloat(), explain = lastScamExplain)

        val fusionResult = fusionEngine?.combine(spoofSignal, scamSignal) ?: return@synchronized

        maxRiskSoFar = maxOf(maxRiskSoFar, fusionResult.riskScore)
        val stickyResult = fusionResult.copy(riskScore = maxRiskSoFar)

        val language = _uiState.value.uiLanguage
        val entry = supervisorAgent.update(stickyResult, language)

        val fullTranscript = transcriptOverride ?: synchronized(transcriptLock) {
            "$finalizedTranscript $currentPartial".trim()
        }

        _uiState.update { state ->
            val newTimeline = if (entry != null) {
                state.timeline + TimelineUiEntry(
                    time = entry.elapsedStr,
                    text = entry.explanation,
                    riskScorePercent = (entry.riskScore * 100).toInt(),
                    action = entry.recommendation,
                )
            } else {
                state.timeline
            }

            state.copy(
                liveTranscript = fullTranscript,
                currentRiskPercent = (stickyResult.riskScore * 100).toInt(),
                currentAction = entry?.recommendation ?: state.currentAction,
                timeline = newTimeline,
            )
        }
    }

    private fun stopInternalOnly() {
        isRunning = false
        lifecycleVersion.incrementAndGet()
        scoringScope.coroutineContext[kotlinx.coroutines.Job]?.cancelChildren()
        // The lifecycle write lock is held: all native audio/scoring readers have finished.
        isScoring.set(false)

        try { krokoTranscriber?.release() } catch (e: Throwable) { android.util.Log.e("PipelineRunner", "Error releasing krokoTranscriber", e) }
        try { transcriber?.release() } catch (e: Throwable) { android.util.Log.e("PipelineRunner", "Error releasing transcriber", e) }
        try { spoofDetector?.release() } catch (e: Throwable) { android.util.Log.e("PipelineRunner", "Error releasing spoofDetector", e) }
        try { scamClassifier?.release() } catch (e: Throwable) { android.util.Log.e("PipelineRunner", "Error releasing scamClassifier", e) }
        try { vadGate?.release() } catch (e: Throwable) { android.util.Log.e("PipelineRunner", "Error releasing vadGate", e) }
        
        krokoTranscriber = null
        transcriber = null
        spoofDetector = null
        scamClassifier = null
        vadGate = null
        fusionEngine = null

        System.gc()
    }

    fun stop() = lifecycleLock.write {
        stopInternalOnly()
        _uiState.update { 
            it.copy(
                status = MonitorStatus.Idle, 
                isInitializing = false, 
                liveTranscript = "", 
                currentRiskPercent = 0, 
                currentAction = com.echoguard.fusion.Action.MONITOR, 
                timeline = emptyList()
            ) 
        }
    }

    fun release() {
        stop()
        scoringScope.coroutineContext[kotlinx.coroutines.Job]?.cancel()
    }
}
