"""Handle a support ticket from the command line: a cited reply draft, or an escalation.

Uses your ingested help centre if present (data/index.json), else the Northwind demo help
centre, with the escalation threshold calibrated on its labelled tickets.

    GEMINI_API_KEY=... python ask.py
    GEMINI_API_KEY=... python ask.py "Can I pay by bank transfer?"
"""

from __future__ import annotations

import sys

from helpdesk_agent import agent as agent_mod
from helpdesk_agent import corpus


def handle(agent, articles, tau, message):
    pred = agent.predict(message)
    d = agent_mod.decide(pred, tau)
    searches = [s for s in pred.get("steps", []) if s.get("action") == "search"]
    if len(searches) > 1:
        print(f"  (agent ran {len(searches)} searches: "
              + " | ".join(s.get("query", "") for s in searches) + ")")
    if d["answered"]:
        a = articles[d["citation"]["corpus_id"]] if d.get("citation") else None
        print(f"\n  REPLY DRAFT: {d.get('reply') or d['answer']}")
        print(f"  key fact: {d['answer']}   confidence {d['confidence']:.2f} "
              f"(threshold {tau:.2f})")
        if a:
            print(f"  source: #{d['citation']['corpus_id']} {a['title']} ({a['category_name']})")
    elif pred.get("error"):
        print(f"\n  MODEL UNAVAILABLE: {pred['error']}")
        print("  (an API failure, not an escalation; retry or check the key/quota)")
    else:
        if any(s.get("action") == "abstain" for s in pred.get("steps", [])):
            why = "the help centre does not cover this"
        elif not pred.get("answerable"):
            why = f"only {pred.get('votes',0)} of {agent.n} samples found a supported answer"
        else:
            why = f"confidence {pred.get('confidence',0):.2f} < threshold {tau:.2f}"
        closest = ", ".join(f"#{i} {articles[i]['title']}" for i in pred.get("retrieved", [])[:3])
        print("\n  ESCALATED to a human.")
        print(f"  reason: {why}")
        print(f"  closest articles: {closest or 'none'}")


def main():
    print("Loading help centre + agent...")
    c = corpus.load_corpus()
    tau = corpus.abstention_threshold(c["is_custom"])
    agent = agent_mod.Agent(c["contexts"])
    print(f"Ready: {len(c['contexts'])} articles from {c['name']}. "
          f"Escalation threshold = {tau:.2f}\n")
    if len(sys.argv) > 1:
        handle(agent, c["articles"], tau, " ".join(sys.argv[1:]))
        return
    print("Paste a customer message (or 'q' to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("q", "quit", "exit"):
            break
        if q:
            handle(agent, c["articles"], tau, q)


if __name__ == "__main__":
    main()
