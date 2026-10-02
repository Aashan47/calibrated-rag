"""SQuAD-style scoring + calibration metrics (stdlib only)."""

from __future__ import annotations

import re
import string
from collections import Counter


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> int:
    return int(any(_normalize(pred) == _normalize(g) for g in golds))


def f1(pred: str, golds: list[str]) -> float:
    best = 0.0
    pt = _normalize(pred).split()
    for g in golds:
        gt = _normalize(g).split()
        if not pt or not gt:
            best = max(best, float(pt == gt))
            continue
        common = Counter(pt) & Counter(gt)
        same = sum(common.values())
        if same == 0:
            continue
        prec, rec = same / len(pt), same / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def is_correct(pred: str, golds: list[str], thresh: float = 0.5) -> bool:
    """A prediction counts as correct if it exact-matches or clears an F1 threshold."""
    return exact_match(pred, golds) == 1 or f1(pred, golds) >= thresh


def ece(confidences: list[float], correct: list[bool], bins: int = 10) -> float:
    """Expected Calibration Error over attempted answers."""
    n = len(confidences)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi or (b == 0 and c == 0))]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        total += (len(idx) / n) * abs(acc - conf)
    return total


def reliability_bins(confidences: list[float], correct: list[bool], bins: int = 10):
    """Per-bin (mid, mean_conf, accuracy, count) for the reliability diagram."""
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi or (b == 0 and c == 0))]
        if not idx:
            out.append(((lo + hi) / 2, None, None, 0))
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        out.append(((lo + hi) / 2, conf, acc, len(idx)))
    return out
