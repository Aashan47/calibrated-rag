"""Retrieval ablation: does the RIGHT passage get retrieved? Compares lexical (TF-IDF) vs
hybrid (TF-IDF + embeddings, RRF) by recall@k on the SQuAD slice — each question has a known
gold passage. Cheap: no LLM calls, just retrieval (embeddings are cached).

    GEMINI_API_KEY=... python retrieval_eval.py
"""

from __future__ import annotations

import json
import os

from calibrated_rag import data
from calibrated_rag.retriever import HybridRetriever, TfidfRetriever

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def recall_at(retriever, items, k):
    hits = sum(it["gold_ctx"] in retriever.search(it["question"], k) for it in items)
    return hits / len(items)


def main():
    contexts, items = data.load()
    print(f"{len(items)} questions over {len(contexts)} passages")
    print("Building retrievers (hybrid embeds the corpus once; cached)...")
    tf = TfidfRetriever(contexts)
    hy = HybridRetriever(contexts)
    print(f"hybrid mode: {hy.mode}\n")

    rows = {"lexical_tfidf": {}, "hybrid": {}}
    print(f"{'recall@k':>10} {'lexical':>10} {'hybrid':>10}")
    for k in (1, 3, 5):
        rt, rh = recall_at(tf, items, k), recall_at(hy, items, k)
        rows["lexical_tfidf"][f"recall@{k}"] = round(rt, 4)
        rows["hybrid"][f"recall@{k}"] = round(rh, 4)
        print(f"{k:>10} {rt*100:9.1f}% {rh*100:9.1f}%")

    os.makedirs(RESULTS, exist_ok=True)
    json.dump({"n_questions": len(items), "n_passages": len(contexts),
               "hybrid_mode": hy.mode, **rows},
              open(os.path.join(RESULTS, "retrieval.json"), "w"), indent=2)
    print("\nwrote results/retrieval.json")


if __name__ == "__main__":
    main()
