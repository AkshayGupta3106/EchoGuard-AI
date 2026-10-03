"""Evaluate supplied text over the web session API, with optional audio false alarms.

Reports per-language classification, confidence intervals, cold start, RTT,
turn latency, and retries. The configuration canary is heuristic. Does not
evaluate browser ASR or Android inference. Results remain in unique local runs.
Run: python tools/evaluation/run_eval_e2e.py --help
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from pathlib import Path

import requests

from echoguard.paths import (REPO_ROOT, DATASETS_DIR, new_evaluation_run,
                              check_evaluation_output, check_dataset_names, normalize_evaluation_paths)
from echoguard.evaluation import metrics as st

HERE = REPO_ROOT

CANARY_ID = "h_en_scam_tech_support_03"   # Heuristic rules/MiniLM configuration probe.
WARN_ACTIONS = ("warn", "block")


def post(url, retries=2, timeout=30, **kw):
    """POST with a small retry for transient errors; returns (response, retries_used)."""
    used = 0
    while True:
        try:
            r = requests.post(url, timeout=timeout, **kw)
            if r.status_code in (502, 503, 504) and used < retries:
                used += 1
                time.sleep(1.5 * used)
                continue
            r.raise_for_status()
            return r, used
        except (requests.Timeout, requests.ConnectionError):
            if used < retries:
                used += 1
                time.sleep(1.5 * used)
                continue
            raise


def wait_until_up(base, max_wait=120):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < max_wait:
        try:
            if requests.get(f"{base}/api/health", timeout=10).ok:
                return round(time.perf_counter() - t0, 1)
        except Exception:
            pass
        time.sleep(2)
    return None


def measure_rtt(base, n=10):
    ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            requests.get(f"{base}/api/health", timeout=10)
            ms.append((time.perf_counter() - t0) * 1000)
        except Exception:
            pass
    return round(statistics.median(ms), 1) if ms else None


def run_call(base, item, stats):
    r, u = post(f"{base}/api/live/start")
    stats["retries"] += u
    sid = r.json()["session_id"]
    flagged, mx, det = False, 0.0, None
    lat = []
    try:
        for i, turn in enumerate(item["turns"]):
            t0 = time.perf_counter()
            r, u = post(f"{base}/api/live/turn", data={"session_id": sid, "text": turn["text"], "turn_id": f"eval-{i}"})
            lat.append((time.perf_counter() - t0) * 1000)
            stats["retries"] += u
            resp = r.json()
            mx = max(mx, resp.get("risk_score", 0.0))
            ne = resp.get("new_entry")
            if ne and str(ne.get("action", "")).lower() in WARN_ACTIONS and not flagged:
                flagged, det = True, i
    finally:
        try:
            requests.post(f"{base}/api/live/end", data={"session_id": sid}, timeout=15)
        except Exception:
            pass
    return flagged, mx, det, lat


def eval_dataset(base, data):
    rows, all_lat, first_lat, errors = [], [], [], []
    stats = {"retries": 0}
    for item in data:
        try:
            flagged, mx, det, lat = run_call(base, item, stats)
        except Exception as e:
            errors.append({"id": item["id"], "error": f"{e.__class__.__name__}: {e}"})
            continue
        all_lat += lat
        first_lat += lat[1:]
        rows.append({"id": item["id"], "language": item.get("language"), "label": item["label"],
                     "predicted_scam": flagged, "max_risk_score": round(mx, 4), "detection_turn": det,
                     "critical_turn_index": item.get("critical_turn_index")})
    y = [r["label"] == "scam" for r in rows]
    metrics = {"overall": st.confusion(y, [r["predicted_scam"] for r in rows])}
    for lang in sorted({r["language"] for r in rows}):
        sub = [r for r in rows if r["language"] == lang]
        metrics[lang] = st.confusion([r["label"] == "scam" for r in sub], [r["predicted_scam"] for r in sub])
    return rows, metrics, all_lat, first_lat, errors, stats["retries"]


def lat_block(xs):
    if not xs:
        return None
    return {"p50": round(st.pct(xs, 50), 1), "p95": round(st.pct(xs, 95), 1), "p99": round(st.pct(xs, 99), 1), "n": len(xs)}


def wav_bytes(x, sr=16000):
    import soundfile as sf
    b = io.BytesIO()
    sf.write(b, x, sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def audio_false_alarm(base, folder, seconds=20):
    """Stream benign real recordings in 1 s chunks (text layer silent); count calls that raise WARN/BLOCK."""
    import soundfile as sf
    res, first_times, peaks, lat = 0, [], [], []
    files = sorted(Path(folder).glob("*.wav"))
    for f in files:
        x, sr = sf.read(str(f), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        r, _ = post(f"{base}/api/live/start")
        sid = r.json()["session_id"]
        first, peak = None, 0.0
        try:
            for k in range(0, min(len(x), seconds * sr), sr):
                t0 = time.perf_counter()
                r, _ = post(f"{base}/api/live/audio-chunk", data={"session_id": sid},
                            files={"audio": ("chunk.wav", wav_bytes(x[k:k + sr], sr), "audio/wav")})
                lat.append((time.perf_counter() - t0) * 1000)
                resp = r.json()
                peak = max(peak, resp.get("risk_score", 0.0))
                ne = resp.get("new_entry")
                if ne and str(ne.get("action", "")).lower() in WARN_ACTIONS and first is None:
                    first = k // sr + 1
        finally:
            try:
                requests.post(f"{base}/api/live/end", data={"session_id": sid}, timeout=15)
            except Exception:
                pass
        peaks.append(peak)
        if first is not None:
            res += 1
            first_times.append(first)
    n = len(files)
    return {"recordings": n, "raised_warn_or_block_from_audio_alone": res,
            "fraction": round(res / n, 3) if n else None,
            "median_seconds_to_first_warning": statistics.median(first_times) if first_times else None,
            "median_peak_risk": round(statistics.median(peaks), 3) if peaks else None,
            "audio_chunk_latency_ms": lat_block(lat)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--dataset", nargs="+", required=True)
    ap.add_argument("--bonafide-dir", default=None, help="folder of real 16 kHz mono wavs for the audio false-alarm test")
    ap.add_argument("--canary-from", default=str(DATASETS_DIR / "test_heldout.json"))
    ap.add_argument("--out", default=str(new_evaluation_run("web") / "eval_results_e2e.json"))
    a = normalize_evaluation_paths(ap.parse_args())
    try:
        check_dataset_names(a.dataset)
        check_evaluation_output(Path(a.out).parent, (Path(a.out).name,))
    except ValueError as exc:
        ap.error(str(exc))
    base = a.url.rstrip("/")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"waiting for {base} ...")
    cold = wait_until_up(base)
    if cold is None:
        sys.exit("Service did not answer /api/health within 120 s. Check --url; no valid evaluation ran.")
    rtt = measure_rtt(base)
    out = {"schema_version": 4, "url": base, "cold_start_wait_s": cold, "health_rtt_ms_median": rtt, "datasets": {}}
    print(f"up after {cold}s | health RTT median {rtt} ms")

    try:
        canary = next(x for x in json.load(open(a.canary_from, encoding="utf-8")) if x["id"] == CANARY_ID)
        stats = {"retries": 0}
        _, mx, _, _ = run_call(base, canary, stats)
        out["deployed_config_fingerprint"] = {
            "canary": CANARY_ID, "max_risk": round(mx, 3),
            "guess": "rules+MiniLM (semantic layer active)" if mx >= 0.35 else "rules-only (semantic layer NOT active)",
            "note": "heuristic: rules-only scores this scam 0.0; MiniLM scores it about 0.5",
        }
        print("fingerprint:", out["deployed_config_fingerprint"]["guess"])
    except Exception as e:
        out["deployed_config_fingerprint"] = {"error": str(e)}

    for path in a.dataset:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows, m, lat, lat_ex, errors, retries = eval_dataset(base, data)
        out["datasets"][Path(path).name] = {
            "n_items": len(data), "n_completed": len(rows), "n_errors": len(errors), "retries_used": retries,
            "errors": errors, "metrics": m,
            "per_turn_latency_ms": lat_block(lat),
            "per_turn_latency_ms_excluding_first_turn": lat_block(lat_ex),
            "per_turn_latency_minus_rtt_ms_p50": round(st.pct(lat, 50) - (rtt or 0), 1) if lat else None,
            "per_call": rows,
        }
        o = m["overall"]
        print(f"[e2e] {Path(path).name}: n={o['n']} P={o['precision']} R={o['recall']} F1={o['f1']} "
               f"CI={o['f1_ci95_bootstrap']} | p50={(out['datasets'][Path(path).name]['per_turn_latency_ms'] or {}).get('p50', 'n/a')} ms "
              f"| errors={len(errors)} retries={retries}")

    if a.bonafide_dir:
        out["audio_false_alarm"] = audio_false_alarm(base, a.bonafide_dir)
        f = out["audio_false_alarm"]
        print(f"[e2e audio] {f['raised_warn_or_block_from_audio_alone']}/{f['recordings']} real recordings raised "
              f"WARN/BLOCK from audio alone (median {f['median_seconds_to_first_warning']} s)")

    out["evaluation_complete"] = all(value["n_errors"] == 0 and value["n_completed"] == value["n_items"]
                                      for value in out["datasets"].values())
    with Path(a.out).open("x", encoding="utf-8") as report:
        report.write(json.dumps(st.json_safe(out), ensure_ascii=False, indent=1, allow_nan=False))
    print("wrote", a.out)
    if not out["evaluation_complete"]:
        sys.exit("Some calls failed; saved partial diagnostics, not a complete benchmark. Fix errors and retry.")


if __name__ == "__main__":
    main()
