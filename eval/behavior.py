"""Agent-behaviour, groundedness, and calibration analysis — fully offline.

Runs on the predictions already produced by evaluate.py (results/preds.json), so it costs no
API calls and is deterministic. It answers the questions a point accuracy number can't:

  * Behaviour   — how many searches does the agent run, how often does it reformulate, and
                  how does it decide to abstain (agent judgment vs. the confidence gate)?
  * Retrieval   — does reformulation actually recover the gold passage the first query missed?
  * Groundedness— when it answers, does its citation point to the passage with the answer?
  * Calibration — headline metrics on the held-out test split with 95% bootstrap CIs, and a
                  check that the conformal guarantee holds with margin.

    python -m eval.behavior            # writes results/behavior.json
"""

from __future__ import annotations

import json
import os

from calibrated_rag import data, metrics
from eval.stats import bootstrap_ci, fmt_pct, proportion

HERE = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(HERE, "results")


def _load():
    contexts, items = data.load()
    preds_path = os.path.join(RESULTS, "preds.json")
    if not os.path.exists(preds_path):
        raise SystemExit("results/preds.json not found — run `python evaluate.py` first.")
    preds = json.load(open(preds_path))
    res = json.load(open(os.path.join(RESULTS, "results.json")))
    aligned = [(it, preds[it["question"]]) for it in items if it["question"] in preds]
    return contexts, aligned, res


def _searches(pred):
    return [s for s in pred.get("steps", []) if s.get("action") == "search"]


def behaviour(aligned):
    n = len(aligned)
    n_searches = [len(_searches(p)) for _, p in aligned]
    reformulated = sum(1 for c in n_searches if c > 1)
    agent_abstain = sum(1 for _, p in aligned
                        if any(s.get("action") == "abstain" for s in p.get("steps", [])))
    dist = {}
    for c in n_searches:
        dist[c] = dist.get(c, 0) + 1
    return {
        "queries": n,
        "mean_searches_per_query": round(sum(n_searches) / n, 3),
        "max_searches": max(n_searches),
        "reformulated_pct": round(reformulated / n, 4),
        "agent_abstained_pct": round(agent_abstain / n, 4),
        "search_count_distribution": {str(k): dist[k] for k in sorted(dist)},
    }


def retrieval_recovery(aligned):
    """Does reformulation recover the gold passage the first search missed? (answerable only)"""
    seed_hit = final_hit = recovered = seed_miss = answerable = 0
    for it, p in aligned:
        if it["is_impossible"]:
            continue
        answerable += 1
        gold = it.get("gold_ctx")
        searches = _searches(p)
        seed_ids = searches[0].get("ids", []) if searches else []
        final_ids = p.get("retrieved", [])
        sh = gold in seed_ids
        fh = gold in final_ids
        seed_hit += sh
        final_hit += fh
        if not sh:
            seed_miss += 1
            if fh:
                recovered += 1
    return {
        "answerable_questions": answerable,
        "seed_recall": round(seed_hit / answerable, 4) if answerable else 0.0,
        "final_recall": round(final_hit / answerable, 4) if answerable else 0.0,
        "seed_misses": seed_miss,
        "recovered_by_reformulation": recovered,
        "recovery_rate_on_misses": round(recovered / seed_miss, 4) if seed_miss else None,
    }


def groundedness(aligned, tau):
    """When it answers an answerable question correctly, does the citation point to the
    gold passage? (citation accuracy / faithfulness proxy)"""
    answered_correct = cite_gold = cite_present = 0
    for it, p in aligned:
        if it["is_impossible"]:
            continue
        did_answer = bool(p.get("answerable")) and p.get("confidence", 0.0) >= tau
        correct = metrics.is_correct(p.get("answer", ""), it["answers"])
        if did_answer and correct:
            answered_correct += 1
            cite = p.get("citation")
            if cite:
                cite_present += 1
                if cite.get("corpus_id") == it.get("gold_ctx"):
                    cite_gold += 1
    return {
        "answered_correct": answered_correct,
        "citation_present_pct": round(cite_present / answered_correct, 4) if answered_correct else None,
        "citation_is_gold_passage_pct": round(cite_gold / answered_correct, 4) if answered_correct else None,
    }


def calibration_with_ci(aligned, tau, alpha):
    """Headline metrics on the held-out TEST split with 95% bootstrap CIs (same split as
    evaluate.py: Random(13), 40% calibrate / 60% test)."""
    import random
    rows = []
    for it, p in aligned:
        imp = it["is_impossible"]
        did = bool(p.get("answerable")) and p.get("confidence", 0.0) >= tau
        corr = (not imp) and metrics.is_correct(p.get("answer", ""), it["answers"])
        rows.append({"imp": imp, "answered": did, "correct": corr})
    idx = list(range(len(rows)))
    random.Random(13).shuffle(idx)
    test = [rows[i] for i in idx[int(0.4 * len(idx)):]]

    def selective_acc(rs):
        a = [r for r in rs if r["answered"]]
        return (sum(r["correct"] for r in a) / len(a)) if a else 0.0

    def coverage(rs):
        return sum(r["answered"] for r in rs) / len(rs) if rs else 0.0

    def halluc(rs):
        imp = [r for r in rs if r["imp"]]
        return (sum(1 for r in imp if r["answered"]) / len(imp)) if imp else 0.0

    def sel_error(rs):
        a = [r for r in rs if r["answered"]]
        return (sum(1 for r in a if not r["correct"]) / len(a)) if a else 0.0

    sel_err_ci = bootstrap_ci(test, sel_error, seed=1)
    return {
        "test_n": len(test),
        "selective_accuracy": bootstrap_ci(test, selective_acc, seed=1),
        "coverage": bootstrap_ci(test, coverage, seed=2),
        "hallucination_unanswerable": bootstrap_ci(test, halluc, seed=3),
        "selective_error": sel_err_ci,
        "alpha_target": alpha,
        "guarantee_holds_at_point": sel_err_ci["point"] <= alpha,
        "guarantee_upper_ci_within_margin": sel_err_ci["hi"] <= alpha + 0.05,
    }


def main():
    contexts, aligned, res = _load()
    tau = float(res["calibrated_threshold"])
    alpha = float(res.get("alpha_target_error", 0.20))
    report = {
        "n_predictions": len(aligned),
        "calibrated_threshold": tau,
        "behaviour": behaviour(aligned),
        "retrieval_recovery": retrieval_recovery(aligned),
        "groundedness": groundedness(aligned, tau),
        "calibration_test_split": calibration_with_ci(aligned, tau, alpha),
    }
    out = os.path.join(RESULTS, "behavior.json")
    json.dump(report, open(out, "w"), indent=2)

    b, r, g, c = (report["behaviour"], report["retrieval_recovery"],
                  report["groundedness"], report["calibration_test_split"])
    print("\n============  AGENT BEHAVIOUR  ============")
    print(f"mean searches/query = {b['mean_searches_per_query']}  (max {b['max_searches']})")
    print(f"reformulated (>1 search): {b['reformulated_pct']*100:.1f}%    "
          f"agent-initiated abstain: {b['agent_abstained_pct']*100:.1f}%")
    print(f"search-count distribution: {b['search_count_distribution']}")
    print("\n============  RETRIEVAL / REFORMULATION  ============")
    print(f"gold-passage recall:  seed={r['seed_recall']*100:.1f}%  ->  "
          f"after loop={r['final_recall']*100:.1f}%")
    rr = r["recovery_rate_on_misses"]
    print(f"reformulation recovered {r['recovered_by_reformulation']}/{r['seed_misses']} "
          f"seed-misses" + (f" ({rr*100:.1f}%)" if rr is not None else ""))
    print("\n============  GROUNDEDNESS (citation)  ============")
    cg = g["citation_is_gold_passage_pct"]
    print(f"of correctly-answered Qs, citation points to the gold passage: "
          + (f"{cg*100:.1f}%" if cg is not None else "n/a"))
    print("\n============  CALIBRATION (test split, 95% bootstrap CI)  ============")
    print(f"selective accuracy      {fmt_pct(c['selective_accuracy'])}")
    print(f"coverage                {fmt_pct(c['coverage'])}")
    print(f"hallucination (unansw.) {fmt_pct(c['hallucination_unanswerable'])}")
    print(f"selective error         {fmt_pct(c['selective_error'])}   (target <= {alpha*100:.0f}%)")
    print(f"guarantee holds at point estimate: {c['guarantee_holds_at_point']}   "
          f"upper-CI within margin: {c['guarantee_upper_ci_within_margin']}")
    print(f"\nwrote {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
