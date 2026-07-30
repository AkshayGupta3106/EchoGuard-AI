"""
scam_classifier.py
------------------
Owner: Person B (semantic stream)

Produces scam_score(transcript) -> a score in [0, 1] plus a breakdown of
*which* signals fired, so the agent decision layer / fraud timeline can
explain itself instead of showing a bare number.

Design decisions (see architecture discussion):
  - No fine-tuning. We don't have labeled scam-call data, and fine-tuning a
    classifier on a handful of examples tends to do worse than a simple
    zero-shot + rules approach on small conversational datasets.
  - Two independent signal sources, combined:
      1. rule_score()      - keyword/pattern layer (OTP requests, urgency,
                              claimed authority, secrecy pressure, etc.)
      2. semantic_score()  - MiniLM embedding similarity to a small
                              hand-written set of exemplar scam lines.
  - If the MiniLM model can't be loaded (no network / no weights on device
    yet), the module degrades to rules-only rather than crashing. This
    matters for the live demo: a missing model file should never take the
    whole pipeline down.

Call this incrementally: feed it the *running* transcript (or just the
newest chunk) as the live ASR stream produces text, every 1-2 seconds,
same cadence as the acoustic stream's spoof_score().
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
            # Bare "immediately"/"urgent" were previously only matched
            # inside specific phrases like "act immediately" - a plain
            # demand like "give me money immediately" hit nothing.
            r"\bimmediately\b",
            r"\burgent(ly)?\b",
        ],
    },
    "claimed_authority": {
        "weight": 0.6,
        "patterns": [
            r"calling from (your )?bank",
            r"this is (the )?(rbi|sbi|income tax|cyber ?crime|police)",
            # Added: CBI/Enforcement Directorate/customs/narcotics
            # impersonation is one of the most common current Indian
            # phone-scam patterns (the "digital arrest" script) and had no
            # coverage at all before.
            r"this is (the )?(cbi|enforcement directorate|\bed\b|narcotics|customs)",
            r"security (team|department)",
            r"government (department|office)",
        ],
    },
    "account_threat": {
        "weight": 0.6,
        "patterns": [
            r"account (will be|has been) (blocked|suspended|frozen)",
            r"legal action",
            r"your card (will be|has been) blocked",
        ],
    },
    "secrecy_pressure": {
        "weight": 0.7,
        "patterns": [
            r"(don'?t|do not) tell (anyone|anybody)",
            r"keep this (confidential|between us)",
            r"do not (hang up|disconnect)",
        ],
    },
    "remote_access": {
        "weight": 0.8,
        "patterns": [
            r"anydesk",
            r"teamviewer",
            r"install (this )?app",
            r"screen[\s-]?share",
        ],
    },
    # Direct requests for money/payment - previously had NO coverage at
    # all despite being one of the most common scam signals there is.
    # "Send money immediately" only ever matched "urgency" (0.5) if
    # anything, and "give me money" or "transfer the amount" matched
    # nothing whatsoever.
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
            r"processing fee",
            r"refundable (fee|deposit)",
            r"give (me|us) (the )?money",
            r"hand over (the )?money",
        ],
    },
    # New: requests for identity-document numbers/personal data - the
    # "KYC update" / identity-harvesting scam pattern. Previously only
    # showed up incidentally in one SCAM_EXEMPLARS sentence, with no rule
    # coverage of its own.
    "identity_theft": {
        "weight": 0.65,
        "patterns": [
            r"share your aadhaar",
            r"aadhaar (number|card)",
            r"pan card (number|details)",
            r"date of birth and",
            r"mother'?s maiden name",
            r"share your bank details",
        ],
    },
    # New: the specific "digital arrest" / fake-warrant escalation script.
    # Distinct from and more severe than the generic "legal action" phrase
    # already in account_threat - this is a currently very common and
    # specific Indian phone-scam pattern with previously zero coverage.
    "legal_threat_arrest": {
        "weight": 0.85,
        "patterns": [
            r"arrest warrant",
            r"non[\s-]?bailable",
            r"fir (has been|will be) filed",
            r"digital arrest",
            r"court notice",
            r"you will be arrested",
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
            r"अपहरण",
            r"kidnap kar liya",
            r"accident ho gaya",
            r"hospital (me|mein) (hai|admit)",
        ],
    },
}

_COMPILED_RULES = {
    name: {"weight": cfg["weight"], "regexes": [re.compile(p, re.IGNORECASE) for p in cfg["patterns"]]}
    for name, cfg in RULE_CATEGORIES.items()
}


def _levenshtein(a: str, b: str) -> int:
    """Standard edit distance, small hand-rolled DP - no extra dependency
    needed for something this small, and it needs to match the Kotlin port
    exactly so keep it simple and obvious rather than clever."""
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
    """Undoes two common ASR artifacts before matching, on top of whatever
    the sherpa-onnx hotwords fix (see generate_hotwords.py) already
    prevents at the source:
      - spelled-out acronyms: "o t p" -> "otp" (ASR sometimes emits short
        acronyms as separate letters instead of one token)
      - stray punctuation between letters: "o.t.p" -> "otp"
    This does NOT fix wrong-word substitutions (e.g. "OTP" heard as
    "old TV") - no downstream text processing can recover information the
    ASR simply didn't produce. That's what the hotwords fix is for.
    """
    text = transcript.lower()
    # collapse "x y z" or "x.y.z" -> "xyz" for short runs of single letters
    text = re.sub(r'\b([a-z])[\s.]+([a-z])[\s.]+([a-z])\b', r'\1\2\3', text)
    text = re.sub(r'\b([a-z])[\s.]+([a-z])\b', r'\1\2', text)
    return text


# High-value short/acronym terms worth fuzzy-matching even after the rule
# regexes run - the regexes above already catch exact matches; this catches
# near-misses within a small edit distance. Deliberately NOT applied to the
# longer phrase patterns (urgency, authority, etc.) - those rely on ordinary
# English words a general ASR model already recognizes reliably, so fuzzy
# matching there would mostly just create false positives.
_FUZZY_TERMS = [
    ("otp", "otp_request", 0.9, 1),
    ("cvv", "otp_request", 0.9, 1),
    ("anydesk", "remote_access", 0.8, 2),
    ("teamviewer", "remote_access", 0.8, 2),
]


def _fuzzy_hits(normalized_transcript: str, already_hit_categories: set) -> list:
    hits = []
    words = normalized_transcript.split()
    # Check single words AND adjacent word-pairs joined together - ASR
    # sometimes splits a brand name into two words ("any desk") rather than
    # misspelling it as one ("anidesk"). Both are real, observed failure modes.
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

    # Combine like independent probabilities of "this is a genuine signal":
    # score = 1 - product(1 - weight_i). Multiple weak signals compound.
    score = 1.0
    for h in hits:
        score *= (1.0 - h["weight"])
    score = 1.0 - score
    return {"score": round(min(score, 1.0), 3), "hits": hits}


# ---------------------------------------------------------------------------
# 2. Zero-shot semantic layer - MiniLM embeddings, no fine-tuning.
#    Exemplars are short, hand-written lines representative of common scam
#    scripts. Add to this list as you observe more patterns during testing -
#    no retraining needed, just re-embed (cheap).
# ---------------------------------------------------------------------------

SCAM_EXEMPLARS: List[str] = [
    "Your account will be blocked if you do not act immediately.",
    "Please share the OTP that was just sent to your phone.",
    "This is an urgent call from your bank's security department.",
    "Do not tell anyone about this call, it is confidential.",
    "We need to verify your card details right now.",
    "Install this app so I can access your screen remotely.",
    "There is a legal case against you, pay now to resolve it.",
    "Your KYC has expired, share your Aadhaar and bank details.",
    "You have won a prize, just pay a small processing fee first.",
    "This is the income tax department, you must pay immediately or face arrest.",
    "I have kidnapped your son, if you want him back give me some amount.",
    "Your child met with an accident and is admitted in the hospital, send money for treatment.",
    "Congratulations, you have won a KBC lottery of 25 lakhs. Pay the tax amount to claim it.",
    "Your electricity bill is pending, power will be disconnected at 9 PM tonight.",
    "You have received a cashback reward, click the link to claim it into your account.",
    "This is Microsoft technical support, your computer has a virus.",
]

BENIGN_EXEMPLARS: List[str] = [
    "Hi mom, I am running late, save some dinner for me.",
    "Could you please tell me my account balance?",
    "I need to book a flight to Mumbai for tomorrow evening.",
    "Yeah that sounds good, let's meet at the coffee shop around 4 PM.",
    "I kidnapped your child, give me some amount. Just kidding, I was joking.",
    "Share your OTP and bank details. Just kidding, it's a prank.",
    "I am calling from the police, you will be arrested. Relax, I am just kidding.",
    "Hello, I kidnaped your child. Okay, I was kidding. But I said I was kidding",
    "Hi, just calling to check how you're doing.",
    "Can we reschedule our meeting to next week?",
    "Your order has been shipped and will arrive on Friday.",
    "This is a reminder about your appointment tomorrow.",
]


class SemanticScorer:
    """Wraps MiniLM embeddings. Degrades gracefully if the model isn't available."""

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

        joke_rx = re.compile(r"\b(just kidding|was kidding|i'm kidding|im kidding|i am kidding|only joking|it's a prank|its a prank|mazaak tha|mazak tha|मज़ाक कर रहा|मजाक कर रहा|मज़ाक था|मजाक था|मजाक कर रही|मज़ाक कर रही)\b", re.IGNORECASE)
        serious_rx = re.compile(r"\b(not kidding|i am serious|seriously|not a joke|मज़ाक नहीं|मजाक नहीं|सच बोल रहा|गंभीर हूँ)\b", re.IGNORECASE)

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
# Quick self-test / demo. Run directly: python scam_classifier.py
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
