"""Evaluate Python text scoring, acoustic behavior, and VAD coverage.

Compares rules/MiniLM with per-language/category metrics and confidence intervals.
Optional acoustic checks measure ROC-AUC/EER, false alarms, channel simulation,
and latency; VAD diagnostics compare context-free/context-aware coverage.
--sweep selects a threshold on DEV only. Reports stay in unique local run folders.
Run: python tools/evaluation/run_eval.py --help
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from echoguard.paths import (REPO_ROOT, DATASETS_DIR, SYNTHETIC_AUDIO_DIR,
                              new_evaluation_run, check_evaluation_output, check_dataset_names, normalize_evaluation_paths)

HERE = REPO_ROOT
from echoguard.evaluation import metrics as st

WINDOW = 64600            # AASIST-L input, samples at 16 kHz
WINDOW_MS = WINDOW / 16000 * 1000
WARN_T, FUSION_BOOST = 0.35, 0.45           # SupervisorAgent MEDIUM threshold, FusionEngine max_spoof_boost
WARN_EQUIV_SPOOF = WARN_T / FUSION_BOOST    # spoof prob at which spoof alone (scam=0) reaches WARN


# ----------------------------------------------------------------------------- loading
def load_repo(repo: Path):
    if repo.resolve() != REPO_ROOT:
        sys.exit(
            f"--repo {repo} differs from the installed checkout {REPO_ROOT}. "
            "Install the target checkout with pip install --no-deps -e . first."
        )
    from echoguard.semantic.scam_classifier import ScamClassifier
    from echoguard.fusion.fusion_engine import FusionEngine, StreamSignal
    from echoguard.fusion.supervisor_agent import SupervisorAgent, Action
    return ScamClassifier, FusionEngine, StreamSignal, SupervisorAgent, Action


def build_classifiers(ScamClassifier, want: str):
    with contextlib.redirect_stdout(io.StringIO()):
        full = ScamClassifier()
    minilm_active = bool(full.semantic_scorer.available)
    rules = copy.copy(full)
    rules.semantic_scorer = copy.copy(full.semantic_scorer)
    rules.semantic_scorer.model = None            # force rules-only
    out = {}
    if want in ("rules", "both"):
        out["rules"] = rules
    if want in ("minilm", "both"):
        if minilm_active:
            out["minilm"] = full
        else:
            raise ValueError("Requested MiniLM configuration could not be loaded; refusing a rules-only benchmark")
    return out, minilm_active


# ----------------------------------------------------------------------------- text layers
def eval_text(dataset, clf, threshold, FusionEngine, StreamSignal, SupervisorAgent, Action):
    rows, lat = [], []
    for item in dataset:
        full = " ".join(t["text"] for t in item["turns"])
        t0 = time.perf_counter()
        res = clf.scam_score(full)
        lat.append((time.perf_counter() - t0) * 1000)
        fusion, agent = FusionEngine(), SupervisorAgent()
        running, first, mx, flagged = "", None, 0.0, False
        for i, turn in enumerate(item["turns"]):
            running += " " + turn["text"]
            s = clf.scam_score(running)
            fr = fusion.combine(StreamSignal(0.0, "text-only"), StreamSignal(s.score, s.explain()))
            mx = max(mx, fr.risk_score)
            e = agent.update(fr)
            if e is not None and e.recommendation in (Action.WARN, Action.BLOCK) and first is None:
                first, flagged = i, True
        rows.append({
            "id": item["id"], "language": item["language"], "label": item["label"], "category": item["category"],
            "difficulty": item.get("difficulty", "dev"), "critical_turn_index": item.get("critical_turn_index"),
            "semantic_score": res.score, "semantic_pred": res.score >= threshold,
            "rules_fired": [h["category"] for h in res.rule_hits],
            "textfusion_max_risk": round(mx, 4), "textfusion_pred": flagged, "detection_turn": first,
        })
    return rows, lat


def summarize_text(rows):
    out = {}
    for key, name in (("semantic_pred", "semantic_layer"), ("textfusion_pred", "text_only_fusion_agent")):
        y = [r["label"] == "scam" for r in rows]
        blk = {"overall": st.confusion(y, [r[key] for r in rows])}
        for lang in sorted({r["language"] for r in rows}):
            sub = [r for r in rows if r["language"] == lang]
            blk[lang] = st.confusion([r["label"] == "scam" for r in sub], [r[key] for r in sub])
        sc = [r["semantic_score"] if key == "semantic_pred" else r["textfusion_max_risk"] for r in rows]
        blk["auc_pr"] = round(st.average_precision(y, sc), 4)
        out[name] = blk
    cats = {}
    for r in rows:
        if r["label"] == "scam":
            c = cats.setdefault(f'{r["language"]}/{r["category"]}', {"n": 0, "caught": 0})
            c["n"] += 1
            c["caught"] += int(r["semantic_pred"])
    out["per_category_scam_recall"] = {k: {**v, "recall": round(v["caught"] / v["n"], 3)} for k, v in sorted(cats.items())}
    diffs = {}
    for r in rows:
        d = diffs.setdefault(r["difficulty"], {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += int(r["semantic_pred"] == (r["label"] == "scam"))
    out["accuracy_by_difficulty"] = {k: {**v, "accuracy": round(v["correct"] / v["n"], 3)} for k, v in diffs.items()}
    out["false_positives"] = [r["id"] for r in rows if r["label"] == "benign" and r["semantic_pred"]]
    out["false_negatives"] = [r["id"] for r in rows if r["label"] == "scam" and not r["semantic_pred"]]
    deltas = [r["detection_turn"] - r["critical_turn_index"] for r in rows
              if r["label"] == "scam" and r["detection_turn"] is not None and r["critical_turn_index"] is not None]
    out["time_to_detection_text_only"] = {
        "scam_calls": sum(1 for r in rows if r["label"] == "scam"),
        "never_flagged": sum(1 for r in rows if r["label"] == "scam" and r["detection_turn"] is None),
        "median_turns_vs_critical": statistics.median(deltas) if deltas else None,
        "note": "negative = flagged before the labelled critical turn",
    }
    return out


def sweep_threshold(dev_rows, others: dict):
    """Pick the threshold on DEV only; report the same threshold, unchanged, on every other set."""
    grid = [round(0.2 + 0.05 * i, 2) for i in range(0, 14)]
    table = []
    for t in grid:
        y = [r["label"] == "scam" for r in dev_rows]
        p = [r["semantic_score"] >= t for r in dev_rows]
        table.append((t, st.confusion(y, p)["f1"]))
    best_t = max(table, key=lambda x: (x[1], -abs(x[0] - 0.35)))[0]
    res = {"dev_f1_by_threshold": dict(table), "chosen_on_dev": best_t, "applied_unchanged": {}}
    for name, rows in others.items():
        y = [r["label"] == "scam" for r in rows]
        p = [r["semantic_score"] >= best_t for r in rows]
        res["applied_unchanged"][name] = st.confusion(y, p)
    return res


# ----------------------------------------------------------------------------- audio helpers
def read_wav(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Audio must contain finite, nonempty samples")
    if sr != 16000:
        import scipy.signal as ss
        x = ss.resample_poly(x, 16000, sr).astype(np.float32)
    return x


def windows_of(x, vad, min_speech=0.5, first_only=False, max_windows=None):
    """Non-overlapping 4.04 s windows containing enough speech (VAD) or energy (fallback)."""
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("Audio windows require finite, nonempty mono samples")
    out = []
    starts = range(0, max(1, len(x) - WINDOW + 1), WINDOW)
    for s in starts:
        seg = x[s:s + WINDOW]
        if len(seg) < WINDOW:
            seg = np.tile(seg, int(WINDOW / max(len(seg), 1)) + 1)[:WINDOW]
        if vad is not None:
            ok = vad.speech_rate(seg)[0] >= min_speech
        else:
            ok = float(np.sqrt((seg ** 2).mean())) > 0.005
        if ok:
            out.append(seg)
            if first_only:
                break
        if max_windows and len(out) >= max_windows:
            break
    return out


class Aasist:
    def __init__(self, mod):
        self.det = mod.SpoofDetector()
        self.roll_cls = mod.RollingSpoofScorer

    def logits(self, seg):
        _, lg = self.det.session.run(None, {self.det._input_name: seg[None, :].astype(np.float32)})
        return lg[0]

    def score(self, seg):
        l = self.logits(seg)
        e = np.exp(l - l.max())
        p = e / e.sum()
        return float(p[0]), float(l[0] - l[1])          # (spoof probability, logit margin)


def dist(a):
    a = np.array(a, dtype=float)
    if not len(a):
        return {"n": 0}
    q = np.percentile(a, [5, 25, 50, 75, 95])
    return {"n": int(len(a)), "min": round(float(a.min()), 2), "p5": round(float(q[0]), 2), "p25": round(float(q[1]), 2),
            "median": round(float(q[2]), 2), "p75": round(float(q[3]), 2), "p95": round(float(q[4]), 2), "max": round(float(a.max()), 2)}


def live_loop(aasist, StreamSignal, FusionEngine, SupervisorAgent, Action, x, seconds=60):
    """Simulate webdemo /api/live/audio-chunk on a benign recording (text layer silent)."""
    roll, fusion, agent = aasist.roll_cls(aasist.det), FusionEngine(), SupervisorAgent()
    silent = StreamSignal(0.0, "no scam signals detected")
    first, peak = None, 0.0
    for i in range(0, min(len(x), seconds * 16000), 16000):
        roll.push(x[i:i + 16000])
        r = roll.score()
        fr = fusion.combine(StreamSignal(r["spoof_score"]), silent)
        peak = max(peak, fr.risk_score)
        e = agent.update(fr)
        if e is not None and e.recommendation in (Action.WARN, Action.BLOCK) and first is None:
            first = i // 16000 + 1
    return first, peak


# ----------------------------------------------------------------------------- acoustic
def eval_acoustic(args, repo, mods, vad):
    ScamClassifier, FusionEngine, StreamSignal, SupervisorAgent, Action = mods
    from echoguard.evaluation.tools.phone_channel import phone_channel
    from echoguard.acoustic import spoof_detector_onnx
    aas = Aasist(spoof_detector_onnx)
    res = {"window_ms": round(WINDOW_MS, 1), "vad_used": vad is not None}

    groups = {}         # name -> list of windows (float32 arrays)
    live = {}           # name -> list of raw recordings for live-loop
    if args.audio_dir and Path(args.audio_dir).exists():
        wl = []
        for f in sorted(Path(args.audio_dir).glob("*.wav"))[: args.max_synthetic]:
            wl += windows_of(read_wav(f), vad, 0.3, first_only=True)
        groups["synthetic_tts"] = wl
        groups["synthetic_tts_phone_sim"] = [phone_channel(w) for w in wl]
    for k, d in enumerate(args.bonafide_dir or []):
        d = Path(d)
        if not d.is_dir():
            raise ValueError(f"Requested bonafide directory not found: {d}")
        files = sorted(d.glob("*.wav"))
        name = f"real_speech_{k + 1}"
        wl, recs = [], []
        for f in files:
            x = read_wav(f)
            recs.append(x)
            wl += windows_of(x, vad, 0.5 if vad else 0.0, max_windows=args.max_windows_per_file)
        groups[name] = wl
        live[name] = recs
        res.setdefault("bonafide_dirs", {})[name] = {"n_files": len(files), "n_windows": len(wl),
                                                       "total_minutes": round(sum(len(r) for r in recs) / 16000 / 60, 1)}
        # clean speech through the phone channel (only meaningful for clean studio folders)
        groups[name + "_phone_sim"] = [phone_channel(w) for w in wl[: args.max_synthetic]]

    if not groups or any(not windows for windows in groups.values()):
        raise ValueError("Requested acoustic groups contain no usable audio windows; no complete acoustic benchmark ran")

    scores = {}
    for name, wl in groups.items():
        pr, mg = zip(*[aas.score(w) for w in wl]) if wl else ([], [])
        scores[name] = {"prob": list(pr), "margin": list(mg)}
        res.setdefault("distributions", {})[name] = {"spoof_prob": dist(pr), "logit_margin": dist(mg)}

    real_names = [n for n in scores if n.startswith("real_speech") and not n.endswith("_phone_sim")]
    fa = {}
    for n in real_names:
        pr = np.array(scores[n]["prob"])
        if len(pr):
            fa[n] = {"n_windows": int(len(pr)),
                     "frac_spoof_prob_gt_0.5": round(float((pr > 0.5).mean()), 3),
                     "frac_would_WARN_via_fusion_alone": round(float((pr >= WARN_EQUIV_SPOOF).mean()), 3)}
    res["false_alarm_on_real_speech"] = fa

    auc = {}
    if "synthetic_tts" in scores:
        for kind, tts_key in (("clean_tts", "synthetic_tts"), ("phone_sim_tts", "synthetic_tts_phone_sim")):
            for n in real_names:
                if not scores[n]["prob"]:
                    continue
                auc[f"{kind}_vs_{n}"] = {
                    "auc_softmax_prob": round(st.roc_auc(scores[tts_key]["prob"], scores[n]["prob"]), 3),
                    "auc_logit_margin": round(st.roc_auc(scores[tts_key]["margin"], scores[n]["margin"]), 3),
                    "eer_logit_margin": round(st.eer(scores[tts_key]["margin"], scores[n]["margin"]), 3),
                    "n_tts": len(scores[tts_key]["prob"]), "n_real": len(scores[n]["prob"]),
                }
    res["separability"] = auc

    ll = {}
    for name, recs in live.items():
        firsts, peaks = [], []
        for x in recs:
            if len(x) < 16000:
                continue
            f, p = live_loop(aas, StreamSignal, FusionEngine, SupervisorAgent, Action, x)
            firsts.append(f)
            peaks.append(p)
        warned = [f for f in firsts if f is not None]
        ll[name] = {"recordings": len(firsts), "raised_warn_or_block_from_audio_alone": len(warned),
                    "fraction": round(len(warned) / len(firsts), 3) if firsts else None,
                    "median_seconds_to_first_warning": statistics.median(warned) if warned else None,
                    "median_peak_risk": round(float(np.median(peaks)), 3) if peaks else None}
    res["live_loop_simulation"] = ll

    # latency: repeated, warmed-up, with spread
    probe = (groups.get("synthetic_tts") or next((v for v in groups.values() if v), []))[:100]
    if probe:
        for w in probe[:3]:
            aas.logits(w)
        runs = []
        for _ in range(args.repeat):
            ms = []
            for w in probe:
                t0 = time.perf_counter()
                aas.logits(w)
                ms.append((time.perf_counter() - t0) * 1000)
            runs.append({"p50": round(st.pct(ms, 50), 1), "p99": round(st.pct(ms, 99), 1)})
        res["latency"] = {
            "windows_per_run": len(probe), "runs": runs,
            "p50_ms_median_of_runs": round(statistics.median(r["p50"] for r in runs), 1),
            "p99_ms_median_of_runs": round(statistics.median(r["p99"] for r in runs), 1),
            "p50_ms_range": [min(r["p50"] for r in runs), max(r["p50"] for r in runs)],
            "rtf_p50": round(statistics.median(r["p50"] for r in runs) / WINDOW_MS, 4),
            "one_window_is_ms_of_audio": round(WINDOW_MS, 1),
            "note": "single ONNX thread, one 4.04 s window per call; the whole pipeline is not included",
        }
    return res, scores


# ----------------------------------------------------------------------------- VAD
def eval_vad(args, vad_fixed_cls, repo):
    model = Path(args.vad_model)
    with_context, without_context = vad_fixed_cls(model, use_context=True), vad_fixed_cls(model, use_context=False)
    rng = np.random.default_rng(0)
    sets = {"silence": [np.zeros(16000 * 3, np.float32)],
            "low_level_noise": [(rng.standard_normal(16000 * 3) * 0.01).astype(np.float32)]}
    if args.audio_dir and Path(args.audio_dir).exists():
        sets["synthetic_tts"] = [read_wav(f) for f in sorted(Path(args.audio_dir).glob("*.wav"))[:20]]
    for k, d in enumerate(args.bonafide_dir or []):
        if Path(d).exists():
            sets[f"real_speech_{k + 1}"] = [read_wav(f) for f in sorted(Path(d).glob("*.wav"))]
    out = {}
    for name, clips in sets.items():
        r_plain = [without_context.speech_rate(c) for c in clips]
        r_ctx = [with_context.speech_rate(c) for c in clips]
        tot = sum(r[2] for r in r_ctx) or 1
        out[name] = {"clips": len(clips), "chunks": tot,
                     "speech_rate_repo_behaviour_no_context": round(sum(r[0] * r[2] for r in r_plain) / tot, 3),
                     "speech_rate_fixed_with_context": round(sum(r[0] * r[2] for r in r_ctx) / tot, 3),
                     "max_prob_repo_behaviour": round(max(r[1] for r in r_plain), 3),
                     "max_prob_fixed": round(max(r[1] for r in r_ctx), 3)}
    return out


# ----------------------------------------------------------------------------- report
def cpu_info():
    try:
        import onnxruntime
        ort_v = onnxruntime.__version__
    except Exception:
        ort_v = None
    return {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "numpy": np.__version__, "onnxruntime": ort_v}


def write_report(path, R):
    L = ["# EchoGuard-AI evaluation results\n",
         f"- Evaluation complete: **{R.get('evaluation_complete', False)}**; incomplete reports are diagnostics, not a complete benchmark.",
         f"- date {R['meta']['date']} | repo commit `{R['meta']['repo_head']}` | MiniLM loadable here: **{R['meta']['minilm_loadable']}**",
         f"- machine: {R['meta']['machine']['platform']} | cpus {R['meta']['machine']['cpu_count']} | onnxruntime {R['meta']['machine']['onnxruntime']}",
         f"- semantic threshold {R['meta']['threshold']}",
         "- The lightweight web image uses rules-only text scoring. "
         "This local evaluation does not verify a deployed server's configuration.\n"]
    for set_name, per_cfg in R["text"].items():
        L.append(f"## {set_name}\n")
        L += ["| config | layer | scope | n | precision | recall | F1 | F1 95% CI |", "|---|---|---|---|---|---|---|---|"]
        for cfg, blk in per_cfg.items():
            if cfg == "paired_test":
                continue
            for layer in ("semantic_layer", "text_only_fusion_agent"):
                for scope, m in blk[layer].items():
                    if scope == "auc_pr":
                        continue
                    L.append(f"| {cfg} | {layer} | {scope} | {m['n']} | {m['precision']} | {m['recall']} | {m['f1']} | {m['f1_ci95_bootstrap']} |")
        for cfg, blk in per_cfg.items():
            if cfg == "paired_test":
                continue
            L.append(f"\n**{cfg}** false positives: {blk['false_positives']}  \n**{cfg}** false negatives: {blk['false_negatives']}")
        if "paired_test" in per_cfg:
            L.append(f"\nMcNemar rules vs minilm (exact): {per_cfg['paired_test']}")
        L.append("")
    if "threshold_sweep" in R:
        L += ["## Threshold chosen on DEV only, applied unchanged", json.dumps(R["threshold_sweep"], indent=1), ""]
    if "acoustic" in R:
        L += ["## Acoustic layer", "```", json.dumps(R["acoustic"], indent=1), "```", ""]
    if "vad" in R:
        L += ["## VAD coverage: context-free vs context-aware", "```", json.dumps(R["vad"], indent=1), "```", ""]
    with Path(path).open("x", encoding="utf-8") as report:
        report.write("\n".join(L))


def load_dataset(path):
    """Validate every requested dataset before inference or report writes."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Dataset must be a nonempty list: {path}")
    ids = set()
    for call in data:
        if not isinstance(call, dict) or not all(isinstance(call.get(key), str) for key in ("id", "language", "category")):
            raise ValueError(f"Calls require string id/language/category fields: {path}")
        if call["id"] in ids:
            raise ValueError(f"Duplicate call ID in {path}: {call['id']}")
        ids.add(call["id"])
        if call.get("label") not in ("scam", "benign") or not isinstance(call.get("turns"), list) or not call["turns"]:
            raise ValueError(f"Calls require a scam/benign label and nonempty turns: {path}")
        if any(not isinstance(turn, dict) or not isinstance(turn.get("text"), str) for turn in call["turns"]):
            raise ValueError(f"Turn text must be a string: {path}")
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dev", default=str(DATASETS_DIR / "dataset.json"), help="DEV set (rules were tuned on it)")
    ap.add_argument("--heldout", nargs="*", default=[str(DATASETS_DIR / "test_heldout.json"), str(DATASETS_DIR / "test_heldout2.json")],
                    help="held-out sets; never tune on them")
    ap.add_argument("--config", choices=["rules", "minilm", "both"], default="both")
    ap.add_argument("--threshold", type=float, default=WARN_T)
    ap.add_argument("--sweep", action="store_true", help="choose the threshold on DEV only")
    ap.add_argument("--audio-dir", default=str(SYNTHETIC_AUDIO_DIR), help="synthetic TTS clips (dev set audio)")
    ap.add_argument("--bonafide-dir", action="append", help="folder of real 16 kHz mono wavs; repeat for several")
    ap.add_argument("--vad-model", default=None, help="path to silero_vad.onnx (enables VAD-filtered windows + VAD test)")
    ap.add_argument("--no-acoustic", action="store_true")
    ap.add_argument("--repeat", type=int, default=5, help="latency repeats")
    ap.add_argument("--max-synthetic", type=int, default=60)
    ap.add_argument("--max-windows-per-file", type=int, default=None)
    ap.add_argument("--outdir", default=str(new_evaluation_run("python")))
    a = normalize_evaluation_paths(ap.parse_args())
    try:
        check_dataset_names([a.dev, *a.heldout])
        datasets = {path: load_dataset(path) for path in [a.dev, *a.heldout]}
        if not math.isfinite(a.threshold) or not 0 <= a.threshold <= 1:
            raise ValueError("Threshold must be finite and between 0 and 1")
        if a.repeat < 1 or a.max_synthetic < 1 or (a.max_windows_per_file is not None and a.max_windows_per_file < 1):
            raise ValueError("Repeat/window limits must be positive")
        if not a.no_acoustic:
            for directory in [a.audio_dir, *(a.bonafide_dir or [])]:
                if directory and (not Path(directory).is_dir() or not any(Path(directory).glob("*.wav"))):
                    raise ValueError(f"Requested audio directory is missing or contains no WAV files: {directory}")
            if a.vad_model and not Path(a.vad_model).is_file():
                raise ValueError(f"Requested VAD model not found: {a.vad_model}")
        outputs = ["RESULTS.md", "eval_results.json"]
        outputs.extend(f"per_call_{Path(path).stem}.json" for path in [a.dev, *a.heldout])
        check_evaluation_output(a.outdir, outputs)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))

    repo = Path(a.repo).resolve()
    outdir = Path(a.outdir)
    mods = load_repo(repo)
    ScamClassifier, FusionEngine, StreamSignal, SupervisorAgent, Action = mods
    try:
        clfs, minilm_active = build_classifiers(ScamClassifier, a.config)
    except ValueError as exc:
        ap.error(str(exc))
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"MiniLM loadable: {minilm_active} | configurations run: {list(clfs)}")

    head = "unknown"
    try:
        h = (repo / ".git" / "HEAD").read_text().strip()
        if h.startswith("ref:"):
            ref = repo / ".git" / h.split(" ", 1)[1]
            h = ref.read_text().strip() if ref.exists() else h
        head = h[:12]
    except Exception:
        pass
    R = {"schema_version": 4, "evaluation_complete": True, "meta": {"date": time.strftime("%Y-%m-%d"), "repo_head": head, "threshold": a.threshold,
                                       "minilm_loadable": minilm_active, "machine": cpu_info(), "configs": list(clfs)}}

    sets = {"DEV (in-sample, rules were tuned on it)": a.dev}
    sets.update({f"HELD-OUT {Path(path).stem}": path for path in a.heldout})

    R["text"], all_rows, lat = {}, {}, []
    for label, path in sets.items():
        data = datasets[path]
        per_cfg = {}
        for cfg, clf in clfs.items():
            rows, l = eval_text(data, clf, a.threshold, FusionEngine, StreamSignal, SupervisorAgent, Action)
            lat += l
            all_rows[(label, cfg)] = rows
            per_cfg[cfg] = {"n_calls": len(rows), **summarize_text(rows)}
            m = per_cfg[cfg]["semantic_layer"]["overall"]
            print(f"[{label}] {cfg:6s} n={m['n']} P={m['precision']} R={m['recall']} F1={m['f1']}")
        if "rules" in per_cfg and "minilm" in per_cfg:
            ra, rb = all_rows[(label, "rules")], all_rows[(label, "minilm")]
            ca = [x["semantic_pred"] == (x["label"] == "scam") for x in ra]
            cb = [x["semantic_pred"] == (x["label"] == "scam") for x in rb]
            per_cfg["paired_test"] = st.mcnemar_exact(ca, cb)
        R["text"][label] = per_cfg
        with (outdir / f"per_call_{Path(path).stem}.json").open("x", encoding="utf-8") as report:
            report.write(json.dumps(st.json_safe({cfg: all_rows[(label, cfg)] for cfg in clfs}), ensure_ascii=False, indent=1, allow_nan=False))
    # pooled held-out (all held-out sets together, per configuration): the most stable single number
    held = [k for k in sets if k.startswith("HELD-OUT")]
    if len(held) > 1:
        pooled = {}
        for cfg in clfs:
            rows = [r for k in held for r in all_rows[(k, cfg)]]
            pooled[cfg] = {"n_calls": len(rows), **summarize_text(rows)}
            m = pooled[cfg]["semantic_layer"]["overall"]
            print(f"[HELD-OUT pooled] {cfg:6s} n={m['n']} P={m['precision']} R={m['recall']} F1={m['f1']} CI={m['f1_ci95_bootstrap']}")
        R["text"]["HELD-OUT pooled (all held-out sets)"] = pooled
    if lat:
        R["semantic_latency_ms"] = {"n": len(lat), "p50": round(st.pct(lat, 50), 2), "p95": round(st.pct(lat, 95), 2),
                                    "p99": round(st.pct(lat, 99), 2), "note": "full-transcript scoring, hardware dependent"}

    if a.sweep and clfs:
        cfg = "rules" if "rules" in clfs else next(iter(clfs))
        dev_key = next((k for k in sets if k.startswith("DEV")), None)
        if dev_key:
            R["threshold_sweep"] = {"config": cfg, **sweep_threshold(
                all_rows[(dev_key, cfg)], {k: all_rows[(k, cfg)] for k in sets if k != dev_key})}

    if not a.no_acoustic:
        vad_cls = None
        vad = None
        phase = "acoustic"
        try:
            if a.vad_model:
                from echoguard.evaluation.tools.vad_fixed import VadGateFixed
                vad_cls = VadGateFixed
                vad = VadGateFixed(a.vad_model)
            R["acoustic"], _ = eval_acoustic(a, repo, mods, vad)
            if vad_cls is not None:
                phase = "vad"
                R["vad"] = eval_vad(a, vad_cls, repo)
        except Exception as e:                                # keep the text results even if audio deps are missing
            R[phase] = {"error": f"{e.__class__.__name__}: {e}"}
            R["evaluation_complete"] = False
            print(f"{phase} evaluation failed:", e)

    R = st.json_safe(R)
    with (outdir / "eval_results.json").open("x", encoding="utf-8") as report:
        report.write(json.dumps(R, ensure_ascii=False, indent=1, allow_nan=False))
    write_report(outdir / "RESULTS.md", R)
    missing = [k for k in sets if k.startswith("HELD-OUT")]
    if not missing:
        print("\n*** No held-out set ran: there is NO fair F1 in this output. ***")
    print(f"\nwrote {outdir / 'eval_results.json'} and {outdir / 'RESULTS.md'}")
    if not R["evaluation_complete"]:
        sys.exit("Requested evaluation failed; saved partial diagnostics, not a complete benchmark.")


if __name__ == "__main__":
    main()
