"""Scam-intent scoring from Python rules and optional MiniLM exemplar similarity.

Returns a score in [0, 1] with matched signals. Uses pretrained embeddings
without fine-tuning; falls back to rules if MiniLM cannot load. Callers supply
transcript text and control scoring cadence.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# 1. Rule layer - fast, deterministic, no model required.
#    Each category has a weight (how strong a signal it is on its own) and a
#    list of regex patterns. Keep patterns broad but not so broad they fire
#    on ordinary conversation.
# ---------------------------------------------------------------------------

RULE_CATEGORIES: Dict[str, Dict] = {
    "otp_request": {
        "weight": 0.9,
        "patterns": [
            r"\botp\b",
            r"one[\s-]?time[\s-]?password",
            r"share (the )?code",
            r"verification code",
            r"cvv\b",
            # Hindi/Devanagari OTP and verification-code requests.
            r"ओटीपी",
            r"वन[\s-]?टाइम[\s-]?पासवर्ड",
            r"कोड (बताओ|बताइए|भेजो|भेजिए)",
            r"वेरिफिकेशन कोड",
            r"सीवीवी",
        ],
    },
    "urgency": {
        "weight": 0.5,
        "patterns": [
            r"act (now|immediately)",
            r"right (away|now)",
            r"within (the next )?\d+ (minutes|hours)",
            r"immediately or",
            r"last (warning|chance)",
            r"this is urgent",
            # Standalone urgency terms as well as phrase patterns.
            r"\bimmediately\b",
            r"\burgent(ly)?\b",
            # Hindi/Devanagari
            r"अभी (करो|करिए|करें)",
            r"तुरंत",
            r"जल्दी करो",
        ],
    },
    "claimed_authority": {
        "weight": 0.6,
        "patterns": [
            r"calling from (your )?bank",
            r"this is (the )?(rbi|sbi|income tax|cyber ?crime|police)",
            # Agency impersonation and digital-arrest wording.
            r"this is (the )?(cbi|enforcement directorate|\bed\b|narcotics|customs)",
            r"security (team|department)",
            r"government (department|office)",
            # Hinglish (Romanized Hindi)
            r"cyber crime branch se",
            r"video surveillance mein",
            # Hindi/Devanagari
            r"बैंक से बोल रहा",
            r"पुलिस से बोल रहा",
            r"साइबर क्राइम से बोल रहा",
            r"आयकर विभाग से",
        ],
    },
    "account_threat": {
        "weight": 0.6,
        "patterns": [
            r"account (will be|has been) (blocked|suspended|frozen)",
            r"legal action",
            r"your card (will be|has been) blocked",
            # Hindi/Devanagari
            r"खाता (ब्लॉक|बंद) (हो जाएगा|कर दिया जाएगा)",
            r"कार्ड ब्लॉक हो जाएगा",
            r"कानूनी कार्रवाई",
        ],
    },
    "secrecy_pressure": {
        "weight": 0.7,
        "patterns": [
            r"(don'?t|do not) tell (anyone|anybody)",
            r"keep this (confidential|between us)",
            r"do not (hang up|disconnect)",
            # Hindi/Devanagari
            r"किसी को मत बताना",
            r"यह बात गुप्त रखो",
            r"फोन मत काटो",
        ],
    },
    "remote_access": {
        "weight": 0.8,
        "patterns": [
            r"anydesk",
            r"teamviewer",
            # Allow up to six intervening words between "install" and "app".
            r"\binstall\b(?:\s+\S+){0,6}?\s+app\b",
            r"screen[\s-]?share",
            r"share (the |your )?screen",
            r"read (your |my )?(sms|messages)",
            # Hindi/Devanagari
            r"एनीडेस्क",
            r"टीमव्यूअर",
            r"ऐप इंस्टॉल कर",
            r"स्क्रीन शेयर कर",
        ],
    },
    # Direct money/payment requests.
    "payment_demand": {
        "weight": 0.75,
        "patterns": [
            r"send (the )?money",
            r"transfer (the )?(money|amount|funds)",
            r"pay (the )?(fine|fee|amount|penalty)",
            r"(need|require) (you to )?pay",
            r"gift card",
            r"wire (the )?(money|transfer)",
            r"deposit (the )?(money|amount)",
            r"processing (fee|charge)",
            r"refundable (fee|deposit)",
            r"give (me|us) (the )?money",
            r"hand over (the )?money",
            # Match payment verbs with up to six words before rupees/account.
            r"\b(pay|deposit|transfer|wire)\b(?:\s+\S+){0,6}?\s+rupees\b",
            r"\bsend\b(?:\s+\S+){0,6}?\s+(rupees|to this account|to the account)\b",
            # "move your savings into a(n) ... account" - the digital-arrest
            # money-transfer step, distinct wording from send/transfer/pay.
            r"move (all )?your (savings|money) (in)?to",
            r"(secure|safe) government account",
            # Hindi/Devanagari payment verbs and amount anchors.
            r"पैसे (दो|दीजिए|भेजो|भेजिए)",
            r"रकम (भेजो|भेजिए|ट्रांसफर कर)",
            r"पैसे ट्रांसफर कर",
            r"फीस (जमा|भरनी|जमा करनी)",
            # Require a payment imperative, not an ordinary mention of rupees.
            r"रुपये\s*(जमा (कर|कीजिए|करें|कर दीजिए)|भेज (दो|दीजिए|दीजिये)|भरनी होगी|देने होंगे)",
            r"(जमा|ट्रांसफर) कर (दीजिए|दो)",
            r"गिफ्ट कार्ड",
            # Romanized Hindi amount/UPI terms with payment verbs.
            r"\b(rupaye|rupiye|rupye)\b.{0,25}\b(jama|bhej|de dijiye|dijiye|chahiye|karo)\b",
            r"\b(jama|bhej|transfer)\b.{0,25}\b(rupaye|rupiye|rupye|paise)\b",
            r"\bupi\b.{0,15}\b(kar dijiye|kardo|kar do)\b",
        ],
    },
    # Guaranteed-return investment and trading-group signals.
    "investment_fraud": {
        "weight": 0.6,
        "patterns": [
            r"guaranteed (profit|return|income)",
            r"zero risk",
            r"double your money",
            r"trading group",
            r"only (a )?(few|two|couple of) slots? (left|remaining)",
            # Hindi/Devanagari
            r"पक्का (मुनाफा|फायदा|रिटर्न)",
            r"बिल्कुल जोखिम नहीं",
            r"ट्रेडिंग ग्रुप",
        ],
    },
    # Recording/blackmail extortion signals.
    "sextortion_blackmail": {
        "weight": 0.8,
        "patterns": [
            r"recording of (your|you)",
            r"video call",
            r"embarrassing (content|video)",
            r"sent? to all your contacts",
            r"delete (everything|the video|the recording)",
            # Hindi/Devanagari
            r"(वीडियो|रिकॉर्डिंग).{0,15}(शर्मनाक|भेज)",
            r"सब कॉन्टैक्ट्स को भेजी",
        ],
    },
    # Task/reward and level-unlocking signals.
    "task_reward_fraud": {
        "weight": 0.6,
        "patterns": [
            r"(like and review|review short videos)",
            r"unlock (the )?(higher|next) level",
            r"earn .{0,15}(per task|rupees per)",
            # Hindi/Devanagari
            r"टास्क.{0,15}(पैसे|रुपये)",
            r"अगले (स्तर|लेवल)",
        ],
    },
    # Identity-document numbers and personal-data requests.
    "identity_theft": {
        "weight": 0.65,
        "patterns": [
            r"share your aadhaar",
            r"aadhaar (number|card)",
            r"pan card (number|details)",
            r"date of birth and",
            r"mother'?s maiden name",
            r"share your bank details",
            r"\bcard number\b(?:\s+\S+){0,6}?\bbataiye\b",
            r"\bcvv\b|\bexpiry date\b.{0,20}bataiye",
            # Hindi/Devanagari
            r"आधार (नंबर|कार्ड) (दो|दीजिए|बताओ|बताइए)",
            r"पैन कार्ड नंबर",
            r"जन्म तारीख बताओ",
        ],
    },
    # Arrest, warrant, and court-notice threats.
    "legal_threat_arrest": {
        "weight": 0.85,
        "patterns": [
            r"arrest warrant",
            r"non[\s-]?bailable",
            r"fir (has been|will be) filed",
            r"digital arrest",
            r"court notice",
            r"you will be arrested",
            # Hindi/Devanagari
            r"गिरफ्तार (किया जाएगा|कर लिया जाएगा)",
            r"डिजिटल अरेस्ट",
            r"गैर[\s-]?जमानती वारंट",
            r"अदालत का नोटिस",
        ],
    },
    "lottery_scam": {
        "weight": 0.65,
        "patterns": [
            r"won (a )?lottery",
            r"lucky draw",
            r"prize money",
            r"kbc lottery",
            r"लॉटरी",
            r"इनाम जीते",
            r"lottery lag",
        ],
    },
    "utility_scam": {
        "weight": 0.75,
        "patterns": [
            r"electricity (will be )?disconnected",
            r"power cut",
            r"electricity bill",
            r"बिजली का बिल",
            r"bijli bill",
            r"bijli vibhag",
            r"connection kat",
            r"light cut",
        ],
    },
    "reward_scam": {
        "weight": 0.6,
        "patterns": [
            r"cashback",
            r"reward points",
            r"claim (your )?reward",
            r"कैशबैक",
            r"रिवॉर्ड",
        ],
    },
    "tech_support_scam": {
        "weight": 0.7,
        "patterns": [
            r"computer virus",
            r"microsoft support",
            r"apple support",
            r"refund",
            r"system hacked",
            r"रिफंड",
            r"कंप्यूटर",
        ],
    },
    "extortion_kidnapping": {
        "weight": 0.85,
        "patterns": [
            r"kidnapped (your|him|her)",
            r"i kidnapped",
            r"your son (is in|has been in)",
            r"your daughter (is in|has been in)",
            r"met with an accident",
            r"hospital admitted",
            r"pay (the )?ransom",
            r"want your son back",
            r"want your daughter back",
            r"want your child back",
            r"i will kill (you|your child|your family)",
            r"अपहरण",
            r"kidnap kar liya",
            r"accident ho gaya",
            r"hospital (me|mein) (hai|admit)",
            # Hindi/Devanagari death threats.
            r"मार दूंगा",
            r"जान से मार",
            r"तुम्हें मार",
        ],
    },
}

_COMPILED_RULES = {
    name: {"weight": cfg["weight"], "regexes": [re.compile(p, re.IGNORECASE) for p in cfg["patterns"]]}
    for name, cfg in RULE_CATEGORIES.items()
}


def _levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance using dynamic programming."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def _normalize_for_matching(transcript: str) -> str:
    """Join spelled-out/inter-letter-punctuated acronyms; not arbitrary word substitutions."""
    text = transcript.lower()
    # collapse "x y z" or "x.y.z" -> "xyz" for short runs of single letters
    text = re.sub(r'\b([a-z])[\s.]+([a-z])[\s.]+([a-z])\b', r'\1\2\3', text)
    text = re.sub(r'\b([a-z])[\s.]+([a-z])\b', r'\1\2', text)
    return text


# Restrict fuzzy matching to high-value terms; longer phrases use exact regexes.
_FUZZY_TERMS = [
    ("otp", "otp_request", 0.9, 1),
    ("cvv", "otp_request", 0.9, 1),
    ("anydesk", "remote_access", 0.8, 2),
    ("teamviewer", "remote_access", 0.8, 2),
]


def _fuzzy_hits(normalized_transcript: str, already_hit_categories: set) -> list:
    hits = []
    words = normalized_transcript.split()
    # Joined adjacent words cover split brand names such as "any desk".
    candidates = list(words) + [words[i] + words[i + 1] for i in range(len(words) - 1)]

    for term, category, weight, max_dist in _FUZZY_TERMS:
        if category in already_hit_categories:
            continue  # exact match already covered this category, don't double-count
        for w in candidates:
            if abs(len(w) - len(term)) > max_dist:
                continue  # cheap length check before the more expensive DP
            if _levenshtein(w, term) <= max_dist:
                hits.append({"category": category, "weight": weight,
                             "matched_text": f"{w} (fuzzy match for '{term}')"})
                break
    return hits


def rule_score(transcript: str) -> Dict:
    """Deterministic keyword/pattern score, plus a fuzzy layer for short
    high-value terms an ASR system is likely to mishear. Returns score +
    which categories fired."""
    normalized = _normalize_for_matching(transcript)

    hits = []
    for name, cfg in _COMPILED_RULES.items():
        for rx in cfg["regexes"]:
            m = rx.search(normalized)
            if m:
                hits.append({"category": name, "weight": cfg["weight"], "matched_text": m.group(0)})
                break  # one hit per category is enough

    hit_categories = {h["category"] for h in hits}
    hits.extend(_fuzzy_hits(normalized, hit_categories))

    if not hits:
        return {"score": 0.0, "hits": []}

    # Compound rule weights: 1 - product(1 - weight_i).
    score = 1.0
    for h in hits:
        score *= (1.0 - h["weight"])
    score = 1.0 - score
    return {"score": round(min(score, 1.0), 3), "hits": hits}


# ---------------------------------------------------------------------------
# 2. Zero-shot semantic layer - MiniLM embeddings, no fine-tuning.
#    Exemplars are short, hand-written lines representative of common scam
#    scripts, loaded from echoguard.semantic.exemplars.
# ---------------------------------------------------------------------------

from echoguard.semantic.exemplars import SCAM_EXEMPLARS, BENIGN_EXEMPLARS


class SemanticScorer:
    """Wraps MiniLM embeddings. Degrades gracefully if the model isn't available."""

    # The English MiniLM model is skipped for any Devanagari codepoint.
    # This script heuristic is not language identification: Latin-script
    # Hinglish still receives semantic scoring, without a Hindi-accuracy guarantee.
    _DEVANAGARI_RX = re.compile(r"[\u0900-\u097F]")

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        self.scam_embeddings = None
        self.benign_embeddings = None
        self._load(model_name)

    def _load(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer  # noqa: import kept local
            self.model = SentenceTransformer(model_name)
            self.scam_embeddings = self.model.encode(SCAM_EXEMPLARS, normalize_embeddings=True)
            self.benign_embeddings = self.model.encode(BENIGN_EXEMPLARS, normalize_embeddings=True)
        except Exception as e:  # noqa: broad - any load failure should degrade, not crash
            print(f"[semantic_scorer] MiniLM unavailable ({e.__class__.__name__}: {e}). "
                  f"Falling back to rules-only scoring.")
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def score(self, transcript: str) -> Dict:
        if not self.available or not transcript.strip():
            return {"score": 0.0, "closest_exemplar": None, "available": self.available}

        if self._DEVANAGARI_RX.search(transcript):
            # Distinguish a loaded but inapplicable English model from a load failure.
            return {"score": 0.0, "closest_exemplar": None, "available": True,
                    "skipped_reason": "devanagari_script_english_only_model"}

        import numpy as np
        emb = self.model.encode([transcript], normalize_embeddings=True)[0]

        scam_sims = self.scam_embeddings @ emb
        benign_sims = self.benign_embeddings @ emb

        best_scam_idx = int(np.argmax(scam_sims))
        best_scam_sim = float(scam_sims[best_scam_idx])
        best_benign_sim = float(np.max(benign_sims))

        # Margin over the closest benign exemplar avoids flagging ordinary
        # sentences that happen to share some vocabulary with scam scripts.
        margin = best_scam_sim - best_benign_sim
        # Map margin -> [-1.0, 1.0] score.
        # Positive means scam-like. Negative means benign-like.
        # A margin near 0 means the model is unsure.
        mapped = max(-1.0, min(1.0, margin / 0.22))

        return {
            "score": round(mapped, 3),
            "closest_exemplar": SCAM_EXEMPLARS[best_scam_idx],
            "similarity": round(best_scam_sim, 3),
            "available": True,
        }


# ---------------------------------------------------------------------------
# 3. Combined scam_score() - what the fusion engine actually calls.
# ---------------------------------------------------------------------------

@dataclass
class ScamScoreResult:
    score: float
    rule_hits: List[Dict] = field(default_factory=list)
    semantic_hit: Optional[Dict] = None
    is_joke_override: bool = False
    timestamp: float = field(default_factory=time.time)

    def explain(self) -> str:
        """Human-readable reasons, for the fraud timeline UI."""
        if self.is_joke_override:
            return "joke/prank detected (overridden)"
        reasons = [f"{h['category'].replace('_', ' ')} (\"{h['matched_text']}\")" for h in self.rule_hits]
        if self.semantic_hit and self.semantic_hit.get("score", 0) > 0.4:
            reasons.append(f"semantically similar to known scam phrasing "
                            f"(closest: \"{self.semantic_hit['closest_exemplar']}\")")
        return "; ".join(reasons) if reasons else "no scam signals detected"


class ScamClassifier:
    def __init__(self, rule_weight: float = 0.5, semantic_weight: float = 0.5):
        self.rule_weight = rule_weight
        self.semantic_weight = semantic_weight
        self.semantic_scorer = SemanticScorer()

    def scam_score(self, transcript: str) -> ScamScoreResult:
        r = rule_score(transcript)
        s = self.semantic_scorer.score(transcript)

        if s["available"]:
            sem = s["score"]
            if sem > 0:
                combined = r["score"] + (1.0 - r["score"]) * sem * self.semantic_weight
            else:
                combined = r["score"] * (1.0 + sem * self.semantic_weight)
        else:
            combined = r["score"]

        # English phrases use word boundaries. Hindi phrases use substring
        # matching because dependent vowel signs are not Python \w characters.
        joke_rx = re.compile(
            r"\b(just kidding|was kidding|i'm kidding|im kidding|i am kidding|only joking|it's a prank|its a prank|mazaak tha|mazak tha)\b"
            r"|(?:मज़ाक कर रहा|मजाक कर रहा|मज़ाक था|मजाक था|मजाक कर रही|मज़ाक कर रही)",
            re.IGNORECASE)
        serious_rx = re.compile(
            r"\b(not kidding|i am serious|seriously|not a joke)\b"
            r"|(?:मज़ाक नहीं|मजाक नहीं|सच बोल रहा|गंभीर हूँ)",
            re.IGNORECASE)

        last_joke_match = list(joke_rx.finditer(transcript))
        last_serious_match = list(serious_rx.finditer(transcript))

        last_joke_idx = last_joke_match[-1].start() if last_joke_match else -1
        last_serious_idx = last_serious_match[-1].start() if last_serious_match else -1

        is_joke = last_joke_idx > -1 and last_joke_idx > last_serious_idx

        if is_joke:
            combined = 0.0

        return ScamScoreResult(
            score=round(combined, 3),
            rule_hits=r["hits"],
            semantic_hit=s if s["available"] else None,
            is_joke_override=is_joke,
        )


# ---------------------------------------------------------------------------
# Self-test: python -m echoguard.semantic.scam_classifier
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    classifier = ScamClassifier()

    test_calls = [
        "Hi, this is calling from your bank's security department. "
        "Your account will be blocked, please share the OTP immediately.",

        "Hey, just checking if we're still on for lunch tomorrow at noon.",

        "Do not tell anyone about this call. Install AnyDesk so I can "
        "verify your account right now, this is urgent.",

        "Your order has shipped and should arrive by Friday, thanks for your patience.",
    ]

    for i, call in enumerate(test_calls, 1):
        result = classifier.scam_score(call)
        print(f"\n--- Test call {i} ---")
        print(f"Transcript: {call[:70]}...")
        print(f"Score: {result.score}")
        print(f"Why: {result.explain()}")
