"""
Unit tests for the EchoGuard-AI Python modules (fusion, supervisor, scam classifier rules,
ONNX spoof detector, VAD gate, web demo API, script syntax).

    python -m pytest tools/tests -q

Expected-failure tests document known detection limitations.
"""
import itertools
import py_compile
import random
import time

import numpy as np
import pytest

from echoguard.fusion.fusion_engine import FusionEngine, StreamSignal
from echoguard.fusion.supervisor_agent import Action, SupervisorAgent
from echoguard.semantic import scam_classifier as sc


@pytest.fixture(scope="module")
def clf():
    c = sc.ScamClassifier()
    c.semantic_scorer.model = None          # Force rules-only behavior for these tests.
    return c


# ------------------------------------------------------------------ fusion
class TestFusion:
    def test_scam_sets_baseline(self):
        r = FusionEngine().combine(StreamSignal(0.0), StreamSignal(0.9))
        assert r.risk_score == pytest.approx(0.9)

    def test_spoof_boost_is_bounded(self):
        r = FusionEngine().combine(StreamSignal(1.0), StreamSignal(0.0))
        assert r.risk_score == pytest.approx(0.45)

    def test_bounds_and_monotonic(self):
        f = FusionEngine()
        grid = [i / 10 for i in range(11)]
        for sp, sc_ in itertools.product(grid, grid):
            r = f.combine(StreamSignal(sp), StreamSignal(sc_)).risk_score
            assert 0.0 <= r <= 1.0
            assert r >= f.combine(StreamSignal(0.0), StreamSignal(sc_)).risk_score - 1e-9
            assert r >= f.combine(StreamSignal(sp), StreamSignal(0.0)).risk_score - 1e-9

    @pytest.mark.xfail(reason="KNOWN: spoof prob saturates near 1.0 on real speech, so risk = 0.45 >= 0.35 WARN with a silent text layer", strict=False)
    def test_high_spoof_alone_should_not_warn(self):
        r = FusionEngine().combine(StreamSignal(1.0), StreamSignal(0.0))
        assert r.risk_score < 0.35


# ------------------------------------------------------------------ supervisor
class TestSupervisor:
    def _fr(self, risk, spoof="no spoof detected", scam="no scam signals detected"):
        f = FusionEngine()
        r = f.combine(StreamSignal(0.0, spoof), StreamSignal(risk, scam))
        return r

    @pytest.mark.parametrize("score,expected", [(0.0, Action.MONITOR), (0.34, Action.MONITOR), (0.35, Action.WARN),
                                                 (0.64, Action.WARN), (0.65, Action.BLOCK), (1.0, Action.BLOCK)])
    def test_thresholds(self, score, expected):
        e = SupervisorAgent().update(self._fr(score))
        assert e is not None and e.recommendation == expected

    def test_quiet_when_nothing_changes(self):
        a = SupervisorAgent()
        assert a.update(self._fr(0.5)) is not None
        assert a.update(self._fr(0.5)) is None
        assert len(a.timeline) == 1

    def test_new_entry_when_level_changes(self):
        a = SupervisorAgent()
        a.update(self._fr(0.1))
        e = a.update(self._fr(0.7, scam="urgency"))
        assert e is not None and e.recommendation == Action.BLOCK


# ------------------------------------------------------------------ scam classifier (rules only)
class TestRules:
    BENIGN = ["Hi, your package arrives tomorrow between two and four.", "See you at dinner on Saturday.",
              "आपका अपॉइंटमेंट गुरुवार को ग्यारह बजे है।", ""]

    @pytest.mark.parametrize("text", BENIGN)
    def test_benign_scores_zero(self, clf, text):
        assert clf.scam_score(text).score == 0.0

    def test_otp_request_is_high(self, clf):
        assert clf.scam_score("Please share your OTP right now").score >= 0.9

    def test_rules_compound_like_noisy_or(self):
        r = sc.rule_score("share your otp right now")
        assert len(r["hits"]) >= 2
        expected = 1.0
        for h in r["hits"]:
            expected *= (1 - h["weight"])
        assert r["score"] == pytest.approx(1 - expected, abs=1e-3)

    @pytest.mark.parametrize("text", ["o t p batao", "o.t.p please", "send the OTQ now", "install any desk"])
    def test_asr_artifacts_are_recovered(self, clf, text):
        assert clf.scam_score(text).score > 0

    def test_hindi_devanagari_rule(self, clf):
        assert clf.scam_score("आपको ओटीपी बताना होगा").score > 0.5

    def test_joke_override_zeroes(self, clf):
        assert clf.scam_score("Share your OTP. Just kidding, it's a prank.").score == 0.0

    def test_serious_after_joke_restores(self, clf):
        assert clf.scam_score("Just kidding. Actually I am serious, share your OTP.").score > 0.5

    def test_score_always_bounded_and_never_crashes(self, clf):
        rnd = random.Random(1)
        alphabet = "abc xyz otp cvv अभी 123 .,;'\"\n\t🙂"
        for _ in range(300):
            s = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 200)))
            v = clf.scam_score(s).score
            assert 0.0 <= v <= 1.0

    def test_long_transcript_is_fast(self, clf):
        text = ("hello how are you today " * 2500) + " share your otp"
        t0 = time.perf_counter()
        clf.scam_score(text)
        assert time.perf_counter() - t0 < 2.0

    @pytest.mark.xfail(reason="KNOWN: no rule for reworded fraud, e.g. sextortion / task fraud / investment (see held-out sets)", strict=False)
    def test_reworded_fraud_is_caught(self, clf):
        assert clf.scam_score("We recorded your video call. Pay twenty thousand within one hour or we send it to your contacts.").score >= 0.35


# ------------------------------------------------------------------ spoof detector (ONNX)
@pytest.fixture(scope="module")
def spoof(repo):
    if not (repo / "assets" / "acoustic" / "aasist_l.onnx").exists():
        pytest.skip("aasist_l.onnx not present")
    from echoguard.acoustic import spoof_detector_onnx as m
    return m


class TestSpoofDetector:
    def test_output_contract(self, spoof):
        d = spoof.SpoofDetector()
        r = d.predict_array(np.random.randn(64600).astype(np.float32) * 0.05)
        assert set(["spoof_score", "bonafide_score", "inference_ms"]) <= set(r)
        assert r["spoof_score"] + r["bonafide_score"] == pytest.approx(1.0, abs=2e-4)

    @pytest.mark.parametrize("n", [1, 100, 16000, 64600, 200000])
    def test_any_length_is_padded_or_tiled(self, spoof, n):
        d = spoof.SpoofDetector()
        r = d.predict_array(np.random.randn(n).astype(np.float32) * 0.05)
        assert 0.0 <= r["spoof_score"] <= 1.0

    def test_rolling_scorer_empty_buffer(self, spoof):
        r = spoof.RollingSpoofScorer().score()
        assert r["spoof_score"] == 0.0 and "note" in r

    def test_rolling_scorer_keeps_last_window(self, spoof):
        s = spoof.RollingSpoofScorer()
        s.push(np.zeros(100000, dtype=np.float32))
        assert len(s._buffer) == 64600

    def test_wrong_sample_rate_is_rejected(self, spoof, tmp_path):
        import soundfile as sf
        p = tmp_path / "x.wav"
        sf.write(p, np.zeros(8000, dtype=np.float32), 8000)
        with pytest.raises(ValueError):
            spoof.SpoofDetector().predict_file(str(p))

    def test_margin_is_exposed(self, spoof):
        r = spoof.SpoofDetector().predict_array(np.random.randn(64600).astype(np.float32) * 0.05)
        assert "spoof_margin" in r
        assert np.isfinite(r["spoof_margin"])


# ------------------------------------------------------------------ VAD gate
class TestVad:
    def _gate(self, repo):
        vad = repo / "assets" / "acoustic" / "silero_vad.onnx"
        if not vad.exists():
            pytest.skip("silero_vad.onnx not downloaded (see README)")
        from echoguard.acoustic import vad_gate
        return vad_gate.VadGate()

    def test_silence_is_not_speech(self, repo):
        g = self._gate(repo)
        assert not g.is_speech(np.zeros(512, dtype=np.float32))["is_speech"]

    @pytest.mark.xfail(reason="KNOWN: the model needs a 64-sample context prepended to each chunk; without it real speech scores < 0.5", strict=False)
    def test_real_speech_is_detected(self, repo, kit):
        import soundfile as sf
        g = self._gate(repo)
        x, _ = sf.read(str(kit / "data/audio/bonafide" / "jfk_whisper_cpp_sample.wav"), dtype="float32")
        flags = [g.is_speech(x[i:i + 512])["is_speech"] for i in range(0, len(x) - 511, 512)]
        assert np.mean(flags) > 0.3


# ------------------------------------------------------------------ web demo API
@pytest.fixture(scope="module")
def client(repo):
    from fastapi.testclient import TestClient
    import importlib
    if not (repo / "assets" / "acoustic" / "aasist_l.onnx").exists():
        pytest.skip("aasist_l.onnx not present")
    app_mod = importlib.import_module("echoguard.web.application")
    return TestClient(app_mod.app)


class TestWebDemo:
    def test_health(self, client):
        assert client.get("/api/health").json() == {"status": "ok"}

    def test_analyze_scam_and_benign(self, client):
        s = client.post("/api/analyze", data={"transcript": "This is your bank. Share your OTP right now or your account will be blocked."}).json()
        b = client.post("/api/analyze", data={"transcript": "Hello, your parcel arrives tomorrow afternoon."}).json()
        assert s["risk_score"] >= 0.65 and s["action"] in ("block", "warn")
        assert b["risk_score"] < 0.35

    def test_live_session_flow(self, client):
        sid = client.post("/api/live/start").json()["session_id"]
        r = client.post("/api/live/turn", data={"session_id": sid, "text": "Please share your OTP right now"}).json()
        assert r["risk_score"] >= 0.65
        assert client.post("/api/live/end", data={"session_id": sid}).json()["ended"] is True

    def test_unknown_session_404(self, client):
        assert client.post("/api/live/turn", data={"session_id": "nope", "text": "hi"}).status_code == 404

    def test_empty_turn_is_skipped(self, client):
        sid = client.post("/api/live/start").json()["session_id"]
        assert client.post("/api/live/turn", data={"session_id": sid, "text": "   "}).json() == {"skipped": True}

    def test_bad_audio_chunk_is_400(self, client):
        sid = client.post("/api/live/start").json()["session_id"]
        r = client.post("/api/live/audio-chunk", data={"session_id": sid}, files={"audio": ("x.wav", b"not audio", "audio/wav")})
        assert r.status_code == 400

    @pytest.mark.xfail(reason="KNOWN: benign speech-like audio raises WARN from the acoustic layer alone", strict=False)
    def test_benign_audio_should_not_warn(self, client, kit):
        import io
        import soundfile as sf
        x, sr = sf.read(str(kit / "data/audio/bonafide" / "jfk_whisper_cpp_sample.wav"), dtype="float32")
        sid = client.post("/api/live/start").json()["session_id"]
        b = io.BytesIO()
        sf.write(b, x[:sr], sr, format="WAV", subtype="PCM_16")
        r = client.post("/api/live/audio-chunk", data={"session_id": sid}, files={"audio": ("c.wav", b.getvalue(), "audio/wav")}).json()
        assert r["risk_score"] < 0.35


# ------------------------------------------------------------------ scripts
class TestScripts:
    @pytest.mark.parametrize("name", ["tools/models/download_kroko.py", "tools/models/download_indicconformer.py",
                                        "tools/models/export_aasist.py", "tools/models/export_minilm.py"])
    def test_script_compiles(self, repo, name):
        py_compile.compile(str(repo / name), doraise=True)
