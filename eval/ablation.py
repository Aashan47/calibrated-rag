"""Ablation: does the agent's decision loop actually help, versus a single-shot baseline?

The honest test of "is this really an agent, or just a renamed pipeline" is whether the loop
earns its extra LLM calls. We run the identical stack two ways on the same SQuAD slice:

  * single-shot  (HDA_MAX_STEPS=0): retrieve once, answer. The classic RAG pipeline.
  * agent        (default loop):     judge sufficiency, reformulate + re-search, then answer.

and compare gold-passage recall, end-to-end task accuracy / hallucination, and the cost in
LLM calls per query. Predictions are cached (results/preds.json for the agent, written by
evaluate.py; results/preds_single.json here) so re-analysis is free.

    GEMINI_API_KEY=... python -m eval.ablation [--dataset helpdesk|squad] --alpha 0.20
"""

from __future__ import annotations

import argparse
import json
import os
import random

from helpdesk_agent import conformal, datasets
from helpdesk_agent.agent import Agent
from evaluate import per_item, system_stats

HERE = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(HERE, "results")   # set per dataset in main()


def run_predictions(contexts, items, max_steps, cache):
    preds = json.load(open(cache)) if os.path.exists(cache) else {}
    agent = Agent(contexts, k=3, max_steps=max_steps)
    todo = [it for it in items if it["question"] not in preds]
    for n, it in enumerate(todo, 1):
        preds[it["question"]] = agent.predict(it["question"])
        if n % 20 == 0:
            print(f"  ...{n}/{len(todo)} queried (max_steps={max_steps})")
            json.dump(preds, open(cache, "w"))
    json.dump(preds, open(cache, "w"))
    return [preds[it["question"]] for it in items]


def gold_recall(items, preds):
    ans = [(it, p) for it, p in zip(items, preds) if not it["is_impossible"]]
    hit = sum(1 for it, p in ans if it.get("gold_ctx") in p.get("retrieved", []))
    return hit / len(ans) if ans else 0.0


def llm_calls_per_query(preds, n_samples=5):
    """Decision calls (one per non-seed step) + the N answer samples, averaged."""
    total = 0
    for p in preds:
        steps = p.get("steps", [])
        decisions = sum(1 for s in steps if s.get("action") in ("decide", "abstain"))
        answered = any(s.get("action") == "decide" and s.get("next") == "answer" for s in steps) \
            or not any(s.get("action") == "abstain" for s in steps)
        total += decisions + (n_samples if answered else 0)
    return round(total / len(preds), 2) if preds else 0.0


def evaluate_system(items, preds, alpha):
    rows = per_item(items, preds)
    idx = list(range(len(rows)))
    random.Random(13).shuffle(idx)
    cut = int(0.4 * len(idx))
    cal, test = [rows[i] for i in idx[:cut]], [rows[i] for i in idx[cut:]]
    tau = conformal.calibrate([(r["conf"], r["flag"], r["correct"]) for r in cal], alpha=alpha)
    trust = system_stats(test, use_threshold=None)
    calib = system_stats(test, use_threshold=tau)
    return {
        "gold_recall": round(gold_recall(items, preds), 4),
        "llm_calls_per_query": llm_calls_per_query(preds),
        "calibrated_threshold": round(tau, 4),
        "trust_model": {k: round(v, 4) for k, v in trust.items()},
        "calibrated": {k: round(v, 4) for k, v in calib.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.20)
    ap.add_argument("--dataset", default="helpdesk", choices=["helpdesk", "squad"])
    args = ap.parse_args()
    global RESULTS
    RESULTS = os.path.join(HERE, "results", args.dataset)
    os.makedirs(RESULTS, exist_ok=True)

    contexts, items = datasets.load(args.dataset)
    print(f"Loaded {len(items)} questions over {len(contexts)} passages.\n")

    print("Agent (decision loop) — reusing results/preds.json ...")
    agent_preds = run_predictions(contexts, items, max_steps=3,
                                  cache=os.path.join(RESULTS, "preds.json"))
    print("Single-shot baseline (max_steps=0) ...")
    single_preds = run_predictions(contexts, items, max_steps=0,
                                   cache=os.path.join(RESULTS, "preds_single.json"))

    report = {
        "dataset": args.dataset,
        "alpha_target_error": args.alpha,
        "single_shot": evaluate_system(items, single_preds, args.alpha),
        "agent_loop": evaluate_system(items, agent_preds, args.alpha),
    }
    out = os.path.join(RESULTS, "ablation.json")
    json.dump(report, open(out, "w"), indent=2)

    s, a = report["single_shot"], report["agent_loop"]
    print("\n==================  AGENT LOOP vs SINGLE-SHOT  ==================")
    print(f"{'':28}{'single-shot':>14}{'agent-loop':>14}")
    print(f"{'gold-passage recall':28}{s['gold_recall']*100:>13.1f}%{a['gold_recall']*100:>13.1f}%")
    print(f"{'task acc (trust-model)':28}{s['trust_model']['task_accuracy']*100:>13.1f}%"
          f"{a['trust_model']['task_accuracy']*100:>13.1f}%")
    print(f"{'selective acc (calibrated)':28}{s['calibrated']['selective_accuracy']*100:>13.1f}%"
          f"{a['calibrated']['selective_accuracy']*100:>13.1f}%")
    print(f"{'halluc (trust-model)':28}{s['trust_model']['hallucination_rate_unanswerable']*100:>13.1f}%"
          f"{a['trust_model']['hallucination_rate_unanswerable']*100:>13.1f}%")
    print(f"{'LLM calls / query':28}{s['llm_calls_per_query']:>14}{a['llm_calls_per_query']:>14}")
    print(f"\nwrote {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
