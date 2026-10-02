"""End-to-end evaluation: run the agent over a labelled ticket set, calibrate the escalation
threshold with split-conformal selective prediction, and report the numbers + charts.

    GEMINI_API_KEY=... python evaluate.py                       # helpdesk tickets (the product)
    GEMINI_API_KEY=... python evaluate.py --dataset squad       # public benchmark
    GEMINI_API_KEY=... python evaluate.py --alpha 0.10          # stricter error target

Raw predictions are cached (results/<dataset>/preds.json) so re-running the analysis is free.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from helpdesk_agent import charts, conformal, datasets, metrics
from helpdesk_agent.agent import Agent

HERE = os.path.dirname(__file__)


def results_dir(dataset: str) -> str:
    d = os.path.join(HERE, "results", dataset)
    os.makedirs(d, exist_ok=True)
    return d


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
    """Metrics for a policy. use_threshold=None -> trust the model (no confidence gate).

    In support terms: coverage = auto-resolution rate; selective_accuracy = accuracy of replies
    actually sent; hallucination_rate_unanswerable = wrong replies sent to tickets the help
    centre doesn't cover; task_accuracy = right reply OR correct escalation."""
    n = len(rows)
    n_imp = sum(r["imp"] for r in rows)
    answered = [r for r in rows if r["flag"] and (use_threshold is None or r["conf"] >= use_threshold)]
    sel_correct = sum(r["correct"] for r in answered)
    halluc = sum(1 for r in answered if r["imp"])              # answered an unanswerable
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


def split(rows, frac_cal: float = 0.4, seed: int = 13):
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    cut = int(frac_cal * len(idx))
    return [rows[i] for i in idx[:cut]], [rows[i] for i in idx[cut:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="helpdesk", choices=["helpdesk", "squad"])
    ap.add_argument("--alpha", type=float, default=0.15, help="target error among answered")
    ap.add_argument("--k", type=int, default=3, help="passages retrieved per question")
    args = ap.parse_args()
    out_dir = results_dir(args.dataset)

    print(f"Loading {args.dataset} ...")
    contexts, items = datasets.load(args.dataset)
    n_imp = sum(i["is_impossible"] for i in items)
    print(f"  {len(items)} questions ({len(items)-n_imp} answerable, {n_imp} unanswerable) "
          f"over {len(contexts)} passages")

    print("Running the agent ...")
    agent = Agent(contexts, k=args.k)
    preds = run_predictions(agent, items, os.path.join(out_dir, "preds.json"))
    rows = per_item(items, preds)

    cal, test = split(rows)
    cal_records = [(r["conf"], r["flag"], r["correct"]) for r in cal]
    tau = conformal.calibrate(cal_records, alpha=args.alpha)
    test_records = [(r["conf"], r["flag"], r["correct"]) for r in test]
    cal_check = conformal.selective_report(test_records, tau)

    uncal = system_stats(test, use_threshold=None)
    calib = system_stats(test, use_threshold=tau)
    tc = sum(r["correct"] for r in test)
    naive = {"coverage": 1.0, "selective_accuracy": tc / len(test),
             "hallucination_rate_unanswerable": 1.0, "task_accuracy": tc / len(test),
             "answered": len(test)}

    attempted = [r for r in rows if r["flag"]]
    confs = [r["conf"] for r in attempted]
    corr = [r["correct"] for r in attempted]
    ece = metrics.ece(confs, corr)
    ece_sr = metrics.ece([r["sr"] for r in attempted], corr)
    bins = metrics.reliability_bins(confs, corr)

    curve = []
    for t in [i / 20 for i in range(21)]:
        st = system_stats(test, use_threshold=t)
        if st["answered"] >= 3:
            curve.append((round(st["coverage"], 4), round(st["selective_accuracy"], 4)))

    results = {
        "model": os.environ.get("HDA_MODEL", "gemini-2.5-flash"),
        "dataset": {"helpdesk": "Northwind help centre tickets",
                    "squad": "SQuAD 2.0 (dev slice)"}[args.dataset],
        "n_questions": len(items), "n_unanswerable": n_imp,
        "n_passages": len(contexts), "retrieval_k": args.k,
        "n_calibration": len(cal), "n_test": len(test),
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
    json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    open(os.path.join(out_dir, "reliability.svg"), "w").write(charts.reliability_svg(bins, ece))
    chosen = (calib["coverage"], calib["selective_accuracy"])
    open(os.path.join(out_dir, "coverage.svg"), "w").write(charts.coverage_svg(curve, chosen))

    print("\n================  RESULTS  ================")
    print(f"model={results['model']}  dataset={results['dataset']}  k={args.k}")
    print(f"questions={len(items)} (unanswerable={n_imp})  passages={len(contexts)}  "
          f"calibrate/test = {len(cal)}/{len(test)}")
    print(f"target error alpha={args.alpha}  ->  calibrated confidence threshold = {tau:.2f}")
    print(f"confidence calibration (ECE):  self-consistency={ece:.3f}   raw self-reported={ece_sr:.3f}")
    print("\n                                 always-reply  trust-model   calibrated")
    print(f"accuracy of replies sent         {naive['selective_accuracy']*100:6.1f}%     {uncal['selective_accuracy']*100:6.1f}%      {calib['selective_accuracy']*100:6.1f}%")
    print(f"wrong replies to uncovered Qs    {naive['hallucination_rate_unanswerable']*100:6.1f}%     {uncal['hallucination_rate_unanswerable']*100:6.1f}%      {calib['hallucination_rate_unanswerable']*100:6.1f}%")
    print(f"correct outcome (reply/escalate) {naive['task_accuracy']*100:6.1f}%     {uncal['task_accuracy']*100:6.1f}%      {calib['task_accuracy']*100:6.1f}%")
    print(f"auto-resolved (coverage)         {naive['coverage']*100:6.1f}%     {uncal['coverage']*100:6.1f}%      {calib['coverage']*100:6.1f}%")
    print(f"\ntest-split selective error at threshold = {cal_check['selective_error']*100:.1f}% "
          f"(target <= {args.alpha*100:.0f}%)  guarantee_held={results['guarantee_held']}")
    print(f"wrote results/{args.dataset}/results.json, reliability.svg, coverage.svg")


if __name__ == "__main__":
    main()
