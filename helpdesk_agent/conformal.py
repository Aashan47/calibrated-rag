"""Split-conformal selective prediction: calibrate an abstention threshold with a
risk target.

Goal: when the agent *does* answer, it should be wrong at most a target fraction `alpha`
of the time. We pick a confidence threshold on a held-out calibration split so the empirical
error among accepted (answered) predictions is <= alpha, then verify the guarantee holds on
a disjoint test split. This is the selective-prediction / risk-control idea: trade a little
coverage for a reliability guarantee on what you do answer.

Each calibration record is (confidence, model_answerable, correct). A record can only be
"answered" if the model judged it answerable AND confidence >= threshold.
"""

from __future__ import annotations


def calibrate(records: list[tuple[float, bool, bool]], alpha: float = 0.15,
              min_coverage: float = 0.0) -> float:
    """Return the lowest threshold whose selective error on calibration is <= alpha
    (maximizing coverage). Falls back to the most conservative threshold if none qualifies."""
    # candidate thresholds = the observed confidences (plus 0 and just-above-max)
    confs = sorted({c for c, ans, _ in records if ans} | {0.0, 1.01})
    best_t = 1.01
    total = len(records)
    for t in confs:
        answered = [(corr) for c, ans, corr in records if ans and c >= t]
        if not answered:
            continue
        risk = 1.0 - (sum(answered) / len(answered))
        coverage = len(answered) / total if total else 0.0
        if risk <= alpha and coverage >= min_coverage:
            best_t = t          # confs ascending → first qualifying = lowest = max coverage
            break
    return best_t


def selective_report(records: list[tuple[float, bool, bool]], threshold: float) -> dict:
    """Coverage + selective error among answered, for a given threshold."""
    answered = [corr for c, ans, corr in records if ans and c >= threshold]
    total = len(records)
    return {
        "threshold": threshold,
        "coverage": (len(answered) / total) if total else 0.0,
        "answered": len(answered),
        "selective_accuracy": (sum(answered) / len(answered)) if answered else 0.0,
        "selective_error": (1 - sum(answered) / len(answered)) if answered else 0.0,
    }
