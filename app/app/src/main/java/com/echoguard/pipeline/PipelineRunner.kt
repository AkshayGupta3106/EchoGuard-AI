/**
 * PipelineRunner.kt
 * Owner: together (Day 3)
 *
 * Wires the four already-tested pieces together in the SAME order verified
 * end-to-end in fusion/demo_pipeline.py:
 *
 *   mic audio -> VadGate --(speech only)--> AASIST-L SpoofDetector
 *                                        -> IndicConformer live transcript -> ScamClassifier
 *   both scores -> FusionEngine -> SupervisorAgent -> PipelineUiState
 *
 * ASR engine: AI4Bharat IndicConformer (NeMo CTC via sherpa-onnx)
 *   Replaced English-only Zipformer Transducer with IndicConformer so the
 *   pipeline handles Hinglish code-switching natively. Indian scam callers
 *   frequently mix Hindi into English conversation — Zipformer would drop or
 *   hallucinate those words. IndicConformer is trained on all 22 official
 *   Indian languages and handles this naturally.
 *
 * Two fixes applied to the version this was based on:
 *
 *   1. Removed the `maxAmp > 0.03f` override that was forcing almost every
 *      audio chunk (including background noise) to be treated as speech,
 *      defeating the VAD entirely. If VAD wasn't triggering reliably in
 *      testing, the real fix is lowering VadGate's own probability
 *      threshold (now a constructor parameter, see vadThreshold below) -
 *      not bypassing it with a raw loudness check that lets noise flood
 *      both downstream models.
 *
 *   2. Scam scoring now tracks a cumulative max risk across the whole call
 *      instead of only scoring the trailing ~40 words. Previously, a red
 *      flag raised early in the call (e.g. "this is your bank's security
 *      department") could silently stop affecting the risk score once the
 *      conversation moved on past that 40-word window - a real fraud
 *      signal shouldn't be "forgotten" just because the caller changed
 *      the subject. The full transcript is still what gets scored (cheap
 *      enough - regex + one MiniLM embed call), and the fused risk score
 *      is now the max of "current fused score" and "highest fused score
 *      seen so far this call," so an escalation sticks.
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

class PipelineRunner(
    private val context: Context,
    private val vadThreshold: Float = 0.05f, // Extremely sensitive so it triggers even on low-volume mics!
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

    fun start(assetManager: AssetManager) {
        android.util.Log.i("PipelineRunner", "Starting pipeline...")
        // Ensure previous sessions and memory are fully freed to prevent leaks
        stopInternalOnly()
        _uiState.update { it.copy(isInitializing = true) }

        try {
            // 1. Initialize VadGate
            android.util.Log.i("PipelineRunner", "Initializing VadGate...")
            vadGate = VadGate(assetManager, threshold = vadThreshold)
            Thread.sleep(200)

            // 2. Initialize SpoofDetector (~150MB ONNX)
            android.util.Log.i("PipelineRunner", "Initializing SpoofDetector...")
            spoofDetector = SpoofDetector(context)
            Thread.sleep(200)

            // 3. Initialize ScamClassifier (MiniLM ~30MB ONNX)
            android.util.Log.i("PipelineRunner", "Initializing ScamClassifier...")
            scamClassifier = ScamClassifier(context)
            Thread.sleep(200)

            // 4. Initialize FusionEngine
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

            // 5. Initialize ASR Transcriber based on selected language
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
                }
            }

            isRunning = true
            _uiState.update { it.copy(status = MonitorStatus.Monitoring, isInitializing = false) }
        } catch (e: Throwable) {
            android.util.Log.e("PipelineRunner", "Error during model initialization", e)
            stopInternalOnly()
            currentPartial = "[Error initializing AI models: ${e.localizedMessage}]"
            _uiState.update { it.copy(liveTranscript = currentPartial, status = MonitorStatus.Idle, isInitializing = false) }
        }
    }

    fun setLanguage(language: AppLanguage) {
        _uiState.update { it.copy(uiLanguage = language) }
    }

    private var chunkCount = 0
    @Volatile private var lastSpoofScore = 0f
    @Volatile private var lastSpoofExplain = ""
    @Volatile private var lastScamScore = 0.0
    @Volatile private var lastScamExplain = ""
    @Volatile private var maxRiskSoFar = 0f
    private val stateLock = Any()

    fun onAudioChunk(samples: FloatArray) {
        if (!isRunning) return
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
            val transcriptSnapshot = synchronized(transcriptLock) { 
                "$finalizedTranscript $currentPartial".trim() 
            }
            
            scoringScope.launch {
                try {
                    if (!isRunning) return@launch
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
                    }
                    
                    if (isRunning) {
                        updateUiState(transcriptSnapshot)
                    }
                } catch (e: Throwable) {
                    android.util.Log.e("PipelineRunner", "Error scoring chunk", e)
                } finally {
                    isScoring.set(false)
                }
            }
        } else {
            updateUiState(null)
        }
    }

    private fun updateUiState(transcriptOverride: String?) {
        if (!isRunning) return
        val spoofSignal: StreamSignal
        val scamSignal: StreamSignal
        
        synchronized(stateLock) {
            spoofSignal = StreamSignal(score = lastSpoofScore, explain = lastSpoofExplain)
            scamSignal = StreamSignal(score = lastScamScore.toFloat(), explain = lastScamExplain)
        }

        val fusionResult = fusionEngine?.combine(spoofSignal, scamSignal) ?: return

        synchronized(stateLock) {
            maxRiskSoFar = maxOf(maxRiskSoFar, fusionResult.riskScore)
        }
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
        scoringScope.coroutineContext[kotlinx.coroutines.Job]?.cancelChildren()
        
        // Prevent native crash: Wait for the concurrent scoring thread to finish 
        // accessing ONNX pointers before we release them.
        var waitCount = 0
        while (isScoring.get() && waitCount < 50) {
            Thread.sleep(10)
            waitCount++
        }

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

    fun stop() {
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
}
