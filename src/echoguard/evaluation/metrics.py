"""Dependency-light classification metrics and confidence intervals."""
from __future__ import annotations

import math
import random
from typing import Sequence


def json_safe(value):
    """Use JSON null for undefined metrics rather than non-standard NaN/Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def bootstrap_f1(y_true: Sequence[bool], y_pred: Sequence[bool], iters: int = 2000, seed: int = 7):
    rng = random.Random(seed)
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        tp = sum(1 for i in idx if y_true[i] and y_pred[i])
        fp = sum(1 for i in idx if (not y_true[i]) and y_pred[i])
        fn = sum(1 for i in idx if y_true[i] and (not y_pred[i]))
        vals.append(prf(tp, fp, fn)[2])
    vals.sort()
    return (vals[int(0.025 * iters)], vals[int(0.975 * iters)])


def confusion(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a and b)
    fp = sum(1 for a, b in zip(y_true, y_pred) if (not a) and b)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a and (not b))
    tn = sum(1 for a, b in zip(y_true, y_pred) if (not a) and (not b))
    pr, rc, f1 = prf(tp, fp, fn)
    n = len(y_true)
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": round((tp + tn) / n, 4) if n else None,
        "precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4),
        "precision_ci95": [round(x, 3) for x in wilson(tp, tp + fp)],
        "recall_ci95": [round(x, 3) for x in wilson(tp, tp + fn)],
        "f1_ci95_bootstrap": [round(x, 3) for x in bootstrap_f1(list(y_true), list(y_pred))],
    }


def average_precision(y_true: Sequence[bool], scores: Sequence[float]) -> float:
    pos = sum(1 for y in y_true if y)
    if pos == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp, ap = 0, 0.0
    for rank, i in enumerate(order, 1):
        if y_true[i]:
            tp += 1
            ap += tp / rank
    return ap / pos


def roc_auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score(pos) > score(neg)) with ties counted as 0.5 (Mann-Whitney)."""
    if not len(pos) or not len(neg):
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks.setdefault(k, avg)
        i = j + 1
    rsum = sum(ranks[k] for k, (_, lab) in enumerate(allv) if lab == 1)
    P, N = len(pos), len(neg)
    return (rsum - P * (P + 1) / 2) / (P * N)


def eer(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Equal error rate; 'pos' = spoof scores, 'neg' = bonafide scores (higher = more spoof)."""
    if not len(pos) or not len(neg):
        return float("nan")
    best, val = 1e9, 0.5
    for t in sorted(set(list(pos) + list(neg))):
        far = sum(1 for v in neg if v >= t) / len(neg)
        frr = sum(1 for v in pos if v < t) / len(pos)
        if abs(far - frr) < best:
            best, val = abs(far - frr), (far + frr) / 2
    return val


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """Exact two-sided McNemar test on paired correctness (A vs B)."""
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)   # A right, B wrong
    c = sum(1 for x, y in zip(correct_a, correct_b) if (not x) and y)  # A wrong, B right
    n = b + c
    if n == 0:
        return {"a_only_correct": b, "b_only_correct": c, "p_value": 1.0}
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return {"a_only_correct": b, "b_only_correct": c, "p_value": round(min(1.0, p), 4)}


def pct(vals: Sequence[float], q: float) -> float:
    if not len(vals):
        return float("nan")
    xs = sorted(vals)
    k = (len(xs) - 1) * q / 100
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
