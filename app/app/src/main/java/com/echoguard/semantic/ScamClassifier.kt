/**
 * ScamClassifier.kt
 * Owner: Person B (semantic stream)
 *
 * Kotlin port of scam_classifier.py. Two layers, same as the Python version:
 *
 *   1. Rule layer - regex patterns, fully ported below, deterministic,
 *      no model needed. This is a direct, careful translation of
 *      RULE_CATEGORIES from the Python file - keep the two in sync if you
 *      edit one.
 *
 *   2. Semantic layer - MiniLM embedding similarity. Unlike the acoustic
 *      stream's AASIST-L (which just needed a straight ONNX port), this one
 *      needs a tokenizer too, since MiniLM takes token IDs, not raw text.
 *      Design to minimize on-device work:
 *        - The 14 exemplar sentences NEVER change at runtime, so they're
 *          embedded ONCE on a dev machine (export_exemplar_embeddings.py)
 *          and bundled as a JSON asset - no tokenization/inference needed
 *          for them on-device.
 *        - Only the LIVE transcript gets tokenized + embedded on-device,
 *          via the WordPieceTokenizer below + minilm.onnx.
 *
 * NOT compiled or run - written by hand against the ONNX Runtime Mobile
 * and standard WordPiece algorithm, but this environment has no Android
 * build toolchain to verify it against. Treat as a careful skeleton, test
 * it for real before relying on it.
 */

package com.echoguard.semantic

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.nio.LongBuffer
import kotlin.math.sqrt

// ---------------------------------------------------------------------------
// 1. Rule layer - direct port of RULE_CATEGORIES from scam_classifier.py.
//    Keep these regexes in sync with the Python file if either changes.
// ---------------------------------------------------------------------------

data class RuleCategory(val name: String, val weight: Double, val patterns: List<Regex>)

private val RULE_CATEGORIES: List<RuleCategory> = listOf(
    RuleCategory("otp_request", 0.9, listOf(
        Regex("\\botp\\b", RegexOption.IGNORE_CASE),
        Regex("one[\\s-]?time[\\s-]?password", RegexOption.IGNORE_CASE),
        Regex("share (the )?code", RegexOption.IGNORE_CASE),
        Regex("verification code", RegexOption.IGNORE_CASE),
        Regex("cvv\\b", RegexOption.IGNORE_CASE),
        Regex("ओटीपी", RegexOption.IGNORE_CASE),
        Regex("ओ टी पी", RegexOption.IGNORE_CASE),
        Regex("ओटप", RegexOption.IGNORE_CASE),
        Regex("otp (batao|share|do|bhejo)", RegexOption.IGNORE_CASE),
        Regex("code (batao|share)", RegexOption.IGNORE_CASE),
    )),
    RuleCategory("urgency", 0.5, listOf(
        Regex("act (now|immediately)", RegexOption.IGNORE_CASE),
        Regex("right (away|now)", RegexOption.IGNORE_CASE),
        Regex("within (the next )?\\d+ (minutes|hours)", RegexOption.IGNORE_CASE),
        Regex("immediately or", RegexOption.IGNORE_CASE),
        Regex("last (warning|chance)", RegexOption.IGNORE_CASE),
        Regex("this is urgent", RegexOption.IGNORE_CASE),
        Regex("\\bimmediately\\b", RegexOption.IGNORE_CASE),
        Regex("\\burgent(ly)?\\b", RegexOption.IGNORE_CASE),
        Regex("अभी (करो|भेजो|बताओ)", RegexOption.IGNORE_CASE),
        Regex("तुरंत", RegexOption.IGNORE_CASE),
        Regex("turant", RegexOption.IGNORE_CASE),
        Regex("abhi karo", RegexOption.IGNORE_CASE),
    )),
    RuleCategory("claimed_authority", 0.6, listOf(
        Regex("calling from (your )?bank", RegexOption.IGNORE_CASE),
        Regex("this is (the )?(rbi|sbi|income tax|cyber ?crime|police)", RegexOption.IGNORE_CASE),
        Regex("this is (the )?(cbi|enforcement directorate|\\bed\\b|narcotics|customs)", RegexOption.IGNORE_CASE),
        Regex("security (team|department)", RegexOption.IGNORE_CASE),
        Regex("government (department|office)", RegexOption.IGNORE_CASE),
        Regex("बैंक से (बोल|बात)", RegexOption.IGNORE_CASE),
        Regex("पुलिस (स्टेशन|अधिकारी)", RegexOption.IGNORE_CASE),
        Regex("bank se bol (raha|rahe)", RegexOption.IGNORE_CASE),
        Regex("police officer", RegexOption.IGNORE_CASE),
    )),
    RuleCategory("account_threat", 0.6, listOf(
        Regex("account (will be|has been) (blocked|suspended|frozen)", RegexOption.IGNORE_CASE),
        Regex("legal action", RegexOption.IGNORE_CASE),
        Regex("your card (will be|has been) blocked", RegexOption.IGNORE_CASE),
        Regex("(खाता|अकाउंट|एकाउन्ट|एकाउंट).{0,15}(बंद|ब्लॉक)", RegexOption.IGNORE_CASE),
        Regex("khata (block|band)", RegexOption.IGNORE_CASE),
        Regex("account (block|band)", RegexOption.IGNORE_CASE),
    )),
    RuleCategory("secrecy_pressure", 0.7, listOf(
        Regex("(don'?t|do not) tell (anyone|anybody)", RegexOption.IGNORE_CASE),
        Regex("keep this (confidential|between us)", RegexOption.IGNORE_CASE),
        Regex("do not (hang up|disconnect)", RegexOption.IGNORE_CASE),
        Regex("किसी को (मत|नहीं) बताना", RegexOption.IGNORE_CASE),
        Regex("phone (katna|mat)", RegexOption.IGNORE_CASE),
    )),
    RuleCategory("remote_access", 0.8, listOf(
        Regex("anydesk", RegexOption.IGNORE_CASE),
        Regex("teamviewer", RegexOption.IGNORE_CASE),
        Regex("install (this )?app", RegexOption.IGNORE_CASE),
        Regex("screen[\\s-]?share", RegexOption.IGNORE_CASE),
        Regex("ऐप (डाउनलोड|इंस्टॉल)", RegexOption.IGNORE_CASE),
        Regex("app (download|install) karo", RegexOption.IGNORE_CASE),
    )),
    RuleCategory("payment_demand", 0.75, listOf(
        Regex("send (the )?money", RegexOption.IGNORE_CASE),
        Regex("transfer (the )?(money|amount|funds)", RegexOption.IGNORE_CASE),
        Regex("pay (the )?(fine|fee|amount|penalty)", RegexOption.IGNORE_CASE),
        Regex("(need|require) (you to )?pay", RegexOption.IGNORE_CASE),
        Regex("gift card", RegexOption.IGNORE_CASE),
        Regex("wire (the )?(money|transfer)", RegexOption.IGNORE_CASE),
        Regex("deposit (the )?(money|amount)", RegexOption.IGNORE_CASE),
        Regex("processing fee", RegexOption.IGNORE_CASE),
        Regex("refundable (fee|deposit)", RegexOption.IGNORE_CASE),
        Regex("give (me|us) (the )?money", RegexOption.IGNORE_CASE),
        Regex("hand over (the )?money", RegexOption.IGNORE_CASE),
        Regex("पैसे (भेजो|ट्रांसफर|दो|जमा)", RegexOption.IGNORE_CASE),
        Regex("paise (transfer|bhejo|do|bharo)", RegexOption.IGNORE_CASE),
        Regex("fine (bharo|do)", RegexOption.IGNORE_CASE),
    )),
    RuleCategory("identity_theft", 0.65, listOf(
        Regex("share your aadhaar", RegexOption.IGNORE_CASE),
        Regex("aadhaar (number|card)", RegexOption.IGNORE_CASE),
        Regex("pan card (number|details)", RegexOption.IGNORE_CASE),
        Regex("date of birth and", RegexOption.IGNORE_CASE),
        Regex("mother'?s maiden name", RegexOption.IGNORE_CASE),
        Regex("share your bank details", RegexOption.IGNORE_CASE),
        Regex("आधार (नंबर|कार्ड)", RegexOption.IGNORE_CASE),
        Regex("aadhaar number", RegexOption.IGNORE_CASE),
    )),
    RuleCategory("legal_threat_arrest", 0.85, listOf(
        Regex("arrest warrant", RegexOption.IGNORE_CASE),
        Regex("non[\\s-]?bailable", RegexOption.IGNORE_CASE),
        Regex("fir (has been|will be) filed", RegexOption.IGNORE_CASE),
        Regex("digital arrest", RegexOption.IGNORE_CASE),
        Regex("court notice", RegexOption.IGNORE_CASE),
        Regex("you will be arrested", RegexOption.IGNORE_CASE),
        Regex("गिरफ्तार", RegexOption.IGNORE_CASE),
        Regex("डिजिटल अरेस्ट", RegexOption.IGNORE_CASE),
        Regex("giraftar", RegexOption.IGNORE_CASE),
        Regex("arrest kar", RegexOption.IGNORE_CASE),
    )),
    RuleCategory("lottery_scam", 0.65, listOf(
        Regex("won (a )?lottery", RegexOption.IGNORE_CASE),
        Regex("lucky draw", RegexOption.IGNORE_CASE),
        Regex("prize money", RegexOption.IGNORE_CASE),
        Regex("kbc lottery", RegexOption.IGNORE_CASE),
        Regex("लॉटरी", RegexOption.IGNORE_CASE),
        Regex("इनाम जीते", RegexOption.IGNORE_CASE),
        Regex("lottery lag", RegexOption.IGNORE_CASE),
    )),
    RuleCategory("utility_scam", 0.75, listOf(
        Regex("electricity (will be )?disconnected", RegexOption.IGNORE_CASE),
        Regex("power cut", RegexOption.IGNORE_CASE),
        Regex("electricity bill", RegexOption.IGNORE_CASE),
        Regex("बिजली का बिल", RegexOption.IGNORE_CASE),
        Regex("bijli bill", RegexOption.IGNORE_CASE),
        Regex("light cut", RegexOption.IGNORE_CASE),
    )),
    RuleCategory("reward_scam", 0.6, listOf(
        Regex("cashback", RegexOption.IGNORE_CASE),
        Regex("reward points", RegexOption.IGNORE_CASE),
        Regex("claim (your )?reward", RegexOption.IGNORE_CASE),
        Regex("कैशबैक", RegexOption.IGNORE_CASE),
        Regex("रिवॉर्ड", RegexOption.IGNORE_CASE),
    )),
    RuleCategory("tech_support_scam", 0.7, listOf(
        Regex("computer virus", RegexOption.IGNORE_CASE),
        Regex("microsoft support", RegexOption.IGNORE_CASE),
        Regex("apple support", RegexOption.IGNORE_CASE),
        Regex("refund", RegexOption.IGNORE_CASE),
        Regex("system hacked", RegexOption.IGNORE_CASE),
        Regex("रिफंड", RegexOption.IGNORE_CASE),
        Regex("कंप्यूटर", RegexOption.IGNORE_CASE),
    )),
    RuleCategory("extortion_kidnapping", 0.85, listOf(
        Regex("kidnapped (your|him|her)", RegexOption.IGNORE_CASE),
        Regex("i kidnapped", RegexOption.IGNORE_CASE),
        Regex("your son (is in|has been in)", RegexOption.IGNORE_CASE),
        Regex("your daughter (is in|has been in)", RegexOption.IGNORE_CASE),
        Regex("met with an accident", RegexOption.IGNORE_CASE),
        Regex("hospital admitted", RegexOption.IGNORE_CASE),
        Regex("pay (the )?ransom", RegexOption.IGNORE_CASE),
        Regex("want your son back", RegexOption.IGNORE_CASE),
        Regex("want your daughter back", RegexOption.IGNORE_CASE),
        Regex("want your child back", RegexOption.IGNORE_CASE),
        Regex("अपहरण", RegexOption.IGNORE_CASE),
        Regex("kidnap kar liya", RegexOption.IGNORE_CASE),
        Regex("accident ho gaya", RegexOption.IGNORE_CASE),
        Regex("hospital (me|mein) (hai|admit)", RegexOption.IGNORE_CASE),
    )),
)

data class RuleHit(val category: String, val weight: Double, val matchedText: String)
data class RuleScoreResult(val score: Double, val hits: List<RuleHit>)

/** Standard edit distance - kept simple and obvious rather than optimized,
 * so it stays easy to verify it matches the Python version exactly. */
private fun levenshtein(a: String, b: String): Int {
    if (a == b) return 0
    var prev = IntArray(b.length + 1) { it }
    for (i in 1..a.length) {
        val curr = IntArray(b.length + 1)
        curr[0] = i
        for (j in 1..b.length) {
            val cost = if (a[i - 1] == b[j - 1]) 0 else 1
            curr[j] = minOf(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        }
        prev = curr
    }
    return prev[b.length]
}

/** Undoes two common ASR artifacts before matching - spelled-out acronyms
 * ("o t p" -> "otp") and stray punctuation between letters ("o.t.p" -> "otp").
 * Does NOT fix wrong-word substitutions - see scam_classifier.py's docstring
 * for the same caveat; that's what the sherpa-onnx hotwords fix is for
 * (see generate_hotwords.py). */
private fun normalizeForMatching(transcript: String): String {
    var text = transcript.lowercase()
    
    // Explicit fixes for Zipformer 2023's common phonetic hallucinations
    text = text.replace("or tippy", "otp")
    text = text.replace("old tibby", "otp")
    text = text.replace("scamper", "scammer")
    text = text.replace("any desk", "anydesk")
    text = text.replace("team viewer", "teamviewer")

    text = Regex("\\b([a-z])[\\s.]+([a-z])[\\s.]+([a-z])\\b").replace(text) { it.groupValues[1] + it.groupValues[2] + it.groupValues[3] }
    text = Regex("\\b([a-z])[\\s.]+([a-z])\\b").replace(text) { it.groupValues[1] + it.groupValues[2] }
    return text
}

private data class FuzzyTerm(val term: String, val category: String, val weight: Double, val maxDist: Int)

// Same list as scam_classifier.py's _FUZZY_TERMS - keep these two in sync.
private val FUZZY_TERMS = listOf(
    FuzzyTerm("otp", "otp_request", 0.9, 1),
    FuzzyTerm("cvv", "otp_request", 0.9, 1),
    FuzzyTerm("anydesk", "remote_access", 0.8, 2),
    FuzzyTerm("teamviewer", "remote_access", 0.8, 2),
)

private fun fuzzyHits(normalized: String, alreadyHitCategories: Set<String>): List<RuleHit> {
    val hits = mutableListOf<RuleHit>()
    val words = normalized.split(Regex("\\s+")).filter { it.isNotEmpty() }
    // Single words AND adjacent word-pairs joined - ASR sometimes splits a
    // brand name into two words ("any desk") rather than misspelling it.
    val candidates = words + (0 until words.size - 1).map { words[it] + words[it + 1] }

    for (ft in FUZZY_TERMS) {
        if (ft.category in alreadyHitCategories) continue  // exact match already covered this
        for (w in candidates) {
            if (kotlin.math.abs(w.length - ft.term.length) > ft.maxDist) continue
            if (levenshtein(w, ft.term) <= ft.maxDist) {
                hits.add(RuleHit(ft.category, ft.weight, "$w (fuzzy match for '${ft.term}')"))
                break
            }
        }
    }
    return hits
}

fun ruleScore(transcript: String): RuleScoreResult {
    val normalized = normalizeForMatching(transcript)

    val hits = mutableListOf<RuleHit>()
    for (cat in RULE_CATEGORIES) {
        for (rx in cat.patterns) {
            val m = rx.find(normalized)
            if (m != null) {
                hits.add(RuleHit(cat.name, cat.weight, m.value))
                break  // one hit per category is enough, matches Python version
            }
        }
    }

    val hitCategories = hits.map { it.category }.toSet()
    hits.addAll(fuzzyHits(normalized, hitCategories))

    if (hits.isEmpty()) return RuleScoreResult(0.0, emptyList())

    // score = 1 - product(1 - weight_i), same compounding as the Python version
    var score = 1.0
    for (h in hits) score *= (1.0 - h.weight)
    score = 1.0 - score
    return RuleScoreResult(score.coerceAtMost(1.0), hits)
}

// ---------------------------------------------------------------------------
// 2. Minimal WordPiece tokenizer - the same algorithm BERT/MiniLM tokenizers
//    use: greedy longest-match-first subword matching against a vocab.
//    Loads vocab.txt produced by export_minilm_onnx.py (one token per line,
//    line number == token id).
// ---------------------------------------------------------------------------

class WordPieceTokenizer(vocabLines: List<String>, private val maxSeqLen: Int = 64) {
    private val vocab: Map<String, Int> = vocabLines.withIndex().associate { (i, tok) -> tok to i }
    private val clsId = vocab["[CLS]"] ?: error("vocab.txt missing [CLS]")
    private val sepId = vocab["[SEP]"] ?: error("vocab.txt missing [SEP]")
    private val unkId = vocab["[UNK]"] ?: error("vocab.txt missing [UNK]")
    private val padId = vocab["[PAD]"] ?: 0

    private fun tokenizeWord(word: String): List<Int> {
        // Greedy longest-match, BERT-style: try the longest substring first;
        // subsequent pieces get a "##" prefix.
        val ids = mutableListOf<Int>()
        var start = 0
        var remaining = word
        var isFirst = true
        while (remaining.isNotEmpty()) {
            var end = remaining.length
            var matched: Int? = null
            while (end > 0) {
                val piece = if (isFirst) remaining.substring(0, end) else "##" + remaining.substring(0, end)
                val id = vocab[piece]
                if (id != null) { matched = id; break }
                end--
            }
            if (matched == null) return listOf(unkId)  // whole word -> UNK, standard WordPiece fallback
            ids.add(matched)
            remaining = remaining.substring(end)
            isFirst = false
        }
        return ids
    }

    /** Returns (inputIds, attentionMask), both length maxSeqLen, padded/truncated. */
    fun encode(text: String): Pair<LongArray, LongArray> {
        val words = text.lowercase().trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
        val tokenIds = mutableListOf(clsId)
        for (w in words) {
            val cleaned = w.filter { it.isLetterOrDigit() }
            if (cleaned.isEmpty()) continue
            tokenIds.addAll(tokenizeWord(cleaned))
            if (tokenIds.size >= maxSeqLen - 1) break  // leave room for [SEP]
        }
        tokenIds.add(sepId)

        val inputIds = LongArray(maxSeqLen) { padId.toLong() }
        val attentionMask = LongArray(maxSeqLen) { 0L }
        for (i in tokenIds.indices) {
            if (i >= maxSeqLen) break
            inputIds[i] = tokenIds[i].toLong()
            attentionMask[i] = 1L
        }
        return inputIds to attentionMask
    }
}

// ---------------------------------------------------------------------------
// 3. Semantic scorer - MiniLM ONNX inference + mean pooling + cosine
//    similarity against the precomputed exemplar embeddings.
// ---------------------------------------------------------------------------

data class ExemplarSet(val texts: List<String>, val embeddings: List<FloatArray>)

data class SemanticResult(val score: Float, val closestExemplar: String?, val available: Boolean)

class SemanticScorer(
    private val context: android.content.Context,
    modelAssetPath: String = "models/minilm.onnx",
    vocabAssetPath: String = "models/vocab.txt",
    exemplarsAssetPath: String = "models/exemplar_embeddings.json",
) {
    companion object {
        fun clearCache(context: android.content.Context) {
            try {
                java.io.File(context.cacheDir, "minilm.onnx").delete()
                java.io.File(context.cacheDir, "minilm.onnx.data").delete()
            } catch (_: Throwable) {}
        }
    }

    private var session: OrtSession? = null
    private var tokenizer: WordPieceTokenizer? = null
    private var scamExemplars: ExemplarSet? = null
    private var benignExemplars: ExemplarSet? = null
    private var env: OrtEnvironment? = null

    val available: Boolean get() = session != null && tokenizer != null && scamExemplars != null && benignExemplars != null && env != null

    init {
        val modelFile = java.io.File(context.cacheDir, "minilm.onnx")
        val dataFile = java.io.File(context.cacheDir, "minilm.onnx.data")
        try {
            val runtime = Runtime.getRuntime()
            val availMb = (runtime.maxMemory() - (runtime.totalMemory() - runtime.freeMemory())) / 1024 / 1024
            if (availMb < 100) {
                android.util.Log.w("SemanticScorer", "Low available memory (${availMb}MB). Falling back to rules-only layer to save RAM.")
                session = null
                tokenizer = null
            } else {
                env = OrtEnvironment.getEnvironment()
                val opts = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(1)
                    setInterOpNumThreads(1)
                    setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                }

                if (!modelFile.exists() || !dataFile.exists() || modelFile.length() == 0L || dataFile.length() == 0L) {
                    modelFile.delete()
                    dataFile.delete()
                    context.assets.open(modelAssetPath).use { input ->
                        java.io.FileOutputStream(modelFile).use { output -> input.copyTo(output) }
                    }
                    context.assets.open("$modelAssetPath.data").use { input ->
                        java.io.FileOutputStream(dataFile).use { output -> input.copyTo(output) }
                    }
                }
                session = env?.createSession(modelFile.absolutePath, opts)

                val vocabLines = context.assets.open(vocabAssetPath).bufferedReader().readLines()
                tokenizer = WordPieceTokenizer(vocabLines)

                val json = JSONObject(context.assets.open(exemplarsAssetPath).bufferedReader().readText())
                scamExemplars = parseExemplarSet(json.getJSONArray("scam_exemplars"))
                benignExemplars = parseExemplarSet(json.getJSONArray("benign_exemplars"))
            }
        } catch (e: Throwable) {
            android.util.Log.e("SemanticScorer", "MiniLM failed to load, falling back to rules-only", e)
            try {
                modelFile.delete()
                dataFile.delete()
            } catch (_: Throwable) {}
            session = null
            tokenizer = null
            scamExemplars = null
            benignExemplars = null
        }
    }

    private fun parseExemplarSet(arr: org.json.JSONArray): ExemplarSet {
        val texts = mutableListOf<String>()
        val embeddings = mutableListOf<FloatArray>()
        for (i in 0 until arr.length()) {
            val obj = arr.getJSONObject(i)
            texts.add(obj.getString("text"))
            val embArr = obj.getJSONArray("embedding")
            embeddings.add(FloatArray(embArr.length()) { j -> embArr.getDouble(j).toFloat() })
        }
        return ExemplarSet(texts, embeddings)
    }

    private fun cosineSim(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return dot  // both sides are already L2-normalized, so dot product == cosine similarity
    }

    private fun embed(text: String): FloatArray? {
        val currentEnv = env ?: return null
        val currentSession = session ?: return null
        val currentTokenizer = tokenizer ?: return null
        return try {
            val (inputIds, attentionMask) = currentTokenizer.encode(text)
            val seqLen = inputIds.size

            val inputTensor = OnnxTensor.createTensor(currentEnv, LongBuffer.wrap(inputIds), longArrayOf(1, seqLen.toLong()))
            val maskTensor = OnnxTensor.createTensor(currentEnv, LongBuffer.wrap(attentionMask), longArrayOf(1, seqLen.toLong()))

            currentSession.run(mapOf("input_ids" to inputTensor, "attention_mask" to maskTensor)).use { results ->
                @Suppress("UNCHECKED_CAST")
                val hidden = (results.get("last_hidden_state").get().value as Array<Array<FloatArray>>)[0]  // [seq, hidden]
                val hiddenDim = hidden[0].size

                val pooled = FloatArray(hiddenDim)
                var maskSum = 0f
                for (i in 0 until seqLen) {
                    val m = attentionMask[i].toFloat()
                    maskSum += m
                    for (j in 0 until hiddenDim) pooled[j] += hidden[i][j] * m
                }
                for (j in 0 until hiddenDim) pooled[j] /= maskSum.coerceAtLeast(1e-9f)

                // L2 normalize
                var norm = 0f
                for (v in pooled) norm += v * v
                norm = sqrt(norm).coerceAtLeast(1e-9f)
                for (j in pooled.indices) pooled[j] /= norm

                pooled
            }
        } catch (e: Throwable) {
            android.util.Log.e("SemanticScorer", "Embed failed", e)
            null
        }
    }

    fun score(transcript: String): SemanticResult {
        if (!available || transcript.isBlank()) {
            return SemanticResult(0f, null, available)
        }
        
        // Fix: MiniLM is trained on short sentences. Embedding a massive 
        // multi-minute conversation dilutes the "scam" signal, and our 
        // tokenizer truncates at maxSeqLen (64) anyway, completely ignoring 
        // the end of the call!
        // We only need to embed the most recent trailing context. PipelineRunner 
        // already remembers the max historical risk via `maxRiskSoFar`.
        val words = transcript.split(Regex("\\s+"))
        val trailingContext = words.takeLast(50).joinToString(" ")
        
        val emb = embed(trailingContext) ?: return SemanticResult(0f, null, false)

        val scamEx = scamExemplars ?: return SemanticResult(0f, null, false)
        val benignEx = benignExemplars ?: return SemanticResult(0f, null, false)

        var bestScamSim = -1f
        var bestScamIdx = -1
        scamEx.embeddings.forEachIndexed { i, e ->
            val sim = cosineSim(e, emb)
            if (sim > bestScamSim) { bestScamSim = sim; bestScamIdx = i }
        }
        var bestBenignSim = -1f
        benignEx.embeddings.forEach { e ->
            val sim = cosineSim(e, emb)
            if (sim > bestBenignSim) bestBenignSim = sim
        }

        val margin = bestScamSim - bestBenignSim
        val mapped = (margin / 0.22f).coerceIn(-1f, 1f)

        return SemanticResult(
            score = mapped,
            closestExemplar = if (bestScamIdx >= 0) scamEx.texts[bestScamIdx] else null,
            available = true,
        )
    }

    fun release() {
        try {
            session?.close()
        } catch (_: Throwable) {}
        session = null
        tokenizer = null
        scamExemplars = null
        benignExemplars = null
        env = null
    }
}

// ---------------------------------------------------------------------------
// 4. Combined classifier - same shape as scam_classifier.py's ScamClassifier.
// ---------------------------------------------------------------------------

data class ScamScoreResult(
    val score: Double,
    val ruleHits: List<RuleHit>,
    val semanticResult: SemanticResult?,
    val isJokeOverride: Boolean = false
) {
    fun explain(): String {
        if (isJokeOverride) return "joke/prank detected (overridden)"
        val reasons = mutableListOf<String>()
        for (h in ruleHits) reasons.add("${h.category.replace('_', ' ')} (\"${h.matchedText}\")")
        if (semanticResult != null && semanticResult.available && semanticResult.score > 0.4f) {
            reasons.add("semantically similar to known scam phrasing (closest: \"${semanticResult.closestExemplar}\")")
        }
        return if (reasons.isNotEmpty()) reasons.joinToString("; ") else "no scam signals detected"
    }
}

class ScamClassifier(
    context: android.content.Context,
    private val ruleWeight: Double = 0.5,
    private val semanticWeight: Double = 0.5,
) {
    private val semanticScorer = SemanticScorer(context)

    fun scamScore(transcript: String): ScamScoreResult {
        val r = ruleScore(transcript)
        val s = semanticScorer.score(transcript)

        var combined = r.score as Double
        if (s.available) {
            val sem = s.score.toDouble()
            if (sem > 0) {
                combined = combined + (1.0 - combined) * sem * semanticWeight
            } else {
                combined = combined * (1.0 + sem * semanticWeight)
            }
        }

        val englishJokeRegex = Regex("\\b(kidding|joking|joke|prank|pranking|mazaak|mazak)\\b", RegexOption.IGNORE_CASE)
        val hindiJokeRegex = Regex("(मज़ाक|मजाक|शरारत)")
        
        val englishSeriousRegex = Regex("\\b(not kidding|not joking|not a joke|no joke|serious|seriously)\\b", RegexOption.IGNORE_CASE)
        val hindiSeriousRegex = Regex("(मज़ाक नहीं|मजाक नहीं|सच बोल रहा|गंभीर हूँ|गंभीर)")

        // Find the LAST index of any joke phrase and any serious phrase
        val engJokeEnd = englishJokeRegex.findAll(transcript).lastOrNull()?.range?.last ?: -1
        val hinJokeEnd = hindiJokeRegex.findAll(transcript).lastOrNull()?.range?.last ?: -1
        val lastJokeEnd = maxOf(engJokeEnd, hinJokeEnd)

        val engSeriousEnd = englishSeriousRegex.findAll(transcript).lastOrNull()?.range?.last ?: -1
        val hinSeriousEnd = hindiSeriousRegex.findAll(transcript).lastOrNull()?.range?.last ?: -1
        val lastSeriousEnd = maxOf(engSeriousEnd, hinSeriousEnd)

        // It is a joke IF a joke phrase exists, AND it ends AFTER the most recent serious phrase ends
        val isJoke = lastJokeEnd > -1 && lastJokeEnd > lastSeriousEnd

        if (isJoke) {
            combined = 0.0
        }

        return ScamScoreResult(
            score = Math.round(combined * 1000.0) / 1000.0,
            ruleHits = r.hits,
            semanticResult = if (s.available) s else null,
            isJokeOverride = isJoke
        )
    }

    fun release() {
        semanticScorer.release()
    }
}
