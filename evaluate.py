"""End-to-end evaluation: run the agent over a SQuAD 2.0 slice, calibrate the abstention
threshold with split-conformal selective prediction, and report the numbers + charts.

Raw model predictions are cached (results/preds.json) so re-running the analysis is free.

Usage:
    GEMINI_API_KEY=... python evaluate.py            # full run
    GEMINI_API_KEY=... python evaluate.py --alpha 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import random

from calibrated_rag import charts, conformal, data, metrics
from calibrated_rag.agent import Agent

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")


def run_predictions(agent: Agent, items: list[dict], cache: str) -> list[dict]:
    preds = json.load(open(cache)) if os.path.exists(cache) else {}
    out = []
    for n, it in enumerate(items, 1):
        q = it["question"]
        if q not in preds:
            preds[q] = agent.predict(q)
            if n % 20 == 0:
                print(f"  ...{n}/{len(items)} queried")
                json.dump(preds, open(cache, "w"))
        out.append(preds[q])
    json.dump(preds, open(cache, "w"))
    return out


def per_item(items, preds):
    """Attach correctness to each (question, prediction)."""
    rows = []
    for it, pr in zip(items, preds):
        imp = it["is_impossible"]
        ans_correct = (not imp) and metrics.is_correct(pr.get("answer", ""), it["answers"])
        rows.append({
            "imp": imp,
            "flag": bool(pr.get("answerable")),      # model said "answerable"
            "conf": float(pr.get("confidence", 0.0)),              # self-consistency agreement
            "sr": float(pr.get("selfreport", pr.get("confidence", 0.0))),  # raw self-reported
            "correct": bool(ans_correct),            # correct IF answered (False for unanswerable)
        })
    return rows


def system_stats(rows, use_threshold):
    """Metrics for a policy. use_threshold=None -> trust the model (no confidence gate)."""
    n = len(rows)
    n_imp = sum(r["imp"] for r in rows)
    answered = [r for r in rows if r["flag"] and (use_threshold is None or r["conf"] >= use_threshold)]
    sel_correct = sum(r["correct"] for r in answered)
    halluc = sum(1 for r in answered if r["imp"])              # answered an unanswerable
    # task accuracy: answerable->answered correctly; unanswerable->abstained
    task = 0
    for r in rows:
        did_answer = r["flag"] and (use_threshold is None or r["conf"] >= use_threshold)
        task += (not did_answer) if r["imp"] else (did_answer and r["correct"])
    return {
        "coverage": len(answered) / n if n else 0.0,
        "selective_accuracy": sel_correct / len(answered) if answered else 0.0,
        "hallucination_rate_unanswerable": halluc / n_imp if n_imp else 0.0,
        "task_accuracy": task / n if n else 0.0,
        "answered": len(answered),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.15, help="target error among answered")
    ap.add_argument("--k", type=int, default=3, help="passages retrieved per question")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    print("Loading SQuAD 2.0 slice...")
    contexts, items = data.load()
    n_imp = sum(i["is_impossible"] for i in items)
    print(f"  {len(items)} questions ({len(items)-n_imp} answerable, {n_imp} unanswerable) "
          f"over {len(contexts)} passages")

    print("Running the agent (retrieve -> answer-or-abstain)...")
    agent = Agent(contexts, k=args.k)
    preds = run_predictions(agent, items, os.path.join(RESULTS, "preds.json"))
    rows = per_item(items, preds)

    # split: calibration / test
    idx = list(range(len(rows)))
    random.Random(13).shuffle(idx)
    cut = int(0.4 * len(idx))
    cal = [rows[i] for i in idx[:cut]]
    test = [rows[i] for i in idx[cut:]]

    # calibrate threshold on calibration split
    cal_records = [(r["conf"], r["flag"], r["correct"]) for r in cal]
    tau = conformal.calibrate(cal_records, alpha=args.alpha)
    test_records = [(r["conf"], r["flag"], r["correct"]) for r in test]
    cal_check = conformal.selective_report(test_records, tau)

    uncal = system_stats(test, use_threshold=None)
    calib = system_stats(test, use_threshold=tau)
    # Naive baseline: a vanilla RAG that always answers (no abstention at all).
    tc = sum(r["correct"] for r in test)
    naive = {"coverage": 1.0, "selective_accuracy": tc / len(test),
             "hallucination_rate_unanswerable": 1.0, "task_accuracy": tc / len(test),
             "answered": len(test)}

    # calibration quality (ECE) over attempted answers on the full set
    attempted = [r for r in rows if r["flag"]]
    confs = [r["conf"] for r in attempted]
    corr = [r["correct"] for r in attempted]
    ece = metrics.ece(confs, corr)
    ece_sr = metrics.ece([r["sr"] for r in attempted], corr)   # ablation: raw self-reported
    bins = metrics.reliability_bins(confs, corr)

    # accuracy-vs-coverage curve on test
    curve = []
    for t in [i / 20 for i in range(21)]:
        st = system_stats(test, use_threshold=t)
        if st["answered"] >= 3:
            curve.append((round(st["coverage"], 4), round(st["selective_accuracy"], 4)))

    results = {
        "model": os.environ.get("CRAG_MODEL", "gemini-2.5-flash"),
        "dataset": "SQuAD 2.0 (dev slice)",
        "n_questions": len(items), "n_unanswerable": n_imp,
        "n_passages": len(contexts), "retrieval_k": args.k,
        "alpha_target_error": args.alpha,
        "calibrated_threshold": round(tau, 4),
        "ece": round(ece, 4),
        "ece_selfconsistency": round(ece, 4),
        "ece_selfreported": round(ece_sr, 4),
        "naive_always_answer": naive,
        "uncalibrated_trust_model": uncal,
        "calibrated_abstention": calib,
        "test_selective_error_at_threshold": round(cal_check["selective_error"], 4),
        "guarantee_held": cal_check["selective_error"] <= args.alpha + 0.05,
    }
    json.dump(results, open(os.path.join(RESULTS, "results.json"), "w"), indent=2)
    open(os.path.join(RESULTS, "reliability.svg"), "w").write(charts.reliability_svg(bins, ece))
    chosen = (calib["coverage"], calib["selective_accuracy"])
    open(os.path.join(RESULTS, "coverage.svg"), "w").write(charts.coverage_svg(curve, chosen))

    print("\n================  RESULTS  ================")
    print(f"model={results['model']}  dataset={results['dataset']}  k={args.k}")
    print(f"questions={len(items)} (unanswerable={n_imp})  passages={len(contexts)}")
    print(f"target error alpha={args.alpha}  ->  calibrated confidence threshold = {tau:.2f}")
    print(f"confidence calibration (ECE):  self-consistency={ece:.3f}   raw self-reported={ece_sr:.3f}")
    print("\n                           naive-all   trust-model   calibrated")
    print(f"selective accuracy         {naive['selective_accuracy']*100:6.1f}%     {uncal['selective_accuracy']*100:6.1f}%      {calib['selective_accuracy']*100:6.1f}%")
    print(f"hallucination on unanswer  {naive['hallucination_rate_unanswerable']*100:6.1f}%     {uncal['hallucination_rate_unanswerable']*100:6.1f}%      {calib['hallucination_rate_unanswerable']*100:6.1f}%")
    print(f"task accuracy (ans+absta)  {naive['task_accuracy']*100:6.1f}%     {uncal['task_accuracy']*100:6.1f}%      {calib['task_accuracy']*100:6.1f}%")
    print(f"coverage (fraction answ.)  {naive['coverage']*100:6.1f}%     {uncal['coverage']*100:6.1f}%      {calib['coverage']*100:6.1f}%")
    print(f"\ntest-split selective error at threshold = {cal_check['selective_error']*100:.1f}% "
          f"(target <= {args.alpha*100:.0f}%)  guarantee_held={results['guarantee_held']}")
    print("wrote results/results.json, results/reliability.svg, results/coverage.svg")


if __name__ == "__main__":
    main()
