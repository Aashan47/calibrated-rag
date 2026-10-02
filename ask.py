"""Interactive CLI: ask the agent a question and watch it answer (with a citation) or abstain.

Uses the ingested corpus if present (data/index.json), else the SQuAD 2.0 demo slice, and the
abstention threshold calibrated by evaluate.py (results/results.json).

    GEMINI_API_KEY=... python ask.py
    GEMINI_API_KEY=... python ask.py "Who was yersinia pestis named for?"
"""

from __future__ import annotations

import sys

from calibrated_rag import agent as agent_mod
from calibrated_rag import corpus


def answer(agent, tau, question):
    pred = agent.predict(question)
    d = agent_mod.decide(pred, tau)
    searches = [s for s in pred.get("steps", []) if s.get("action") == "search"]
    if len(searches) > 1:   # the agent reformulated and searched again
        print(f"  (agent ran {len(searches)} searches: "
              + " | ".join(s.get("query", "") for s in searches) + ")")
    if d["answered"]:
        print(f"\n  ANSWER: {d['answer']}")
        print(f"  confidence: {d['confidence']:.2f}  (>= calibrated threshold {tau:.2f})")
        if d.get("citation"):
            snippet = d["citation"]["text"][:240].replace("\n", " ")
            print(f"  cited passage: {snippet}...")
    elif pred.get("error"):
        print(f"\n  MODEL UNAVAILABLE: {pred['error']}")
        print("  (this is an API failure, not an abstention; retry or check the key/quota)")
    else:
        if any(s.get("action") == "abstain" for s in pred.get("steps", [])):
            why = "the agent judged the answer is not in these documents"
        elif not pred.get("answerable"):
            why = f"only {pred.get('votes',0)} of {agent.n} samples found a supported answer"
        else:
            why = f"confidence {pred.get('confidence',0):.2f} < calibrated threshold {tau:.2f}"
        print("\n  ABSTAINED: I can't answer this reliably from the documents.")
        print(f"  reason: {why}")


def main():
    print("Loading corpus + agent...")
    contexts, source, is_custom = corpus.load_contexts()
    tau = corpus.abstention_threshold(is_custom)
    agent = agent_mod.Agent(contexts)
    print(f"Ready: {len(contexts)} passages from {source}. Abstention threshold = {tau:.2f}\n")

    if len(sys.argv) > 1:
        answer(agent, tau, " ".join(sys.argv[1:]))
        return
    print("Type a question (or 'q' to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("q", "quit", "exit"):
            break
        if q:
            answer(agent, tau, q)


if __name__ == "__main__":
    main()
