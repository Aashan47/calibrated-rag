"""The agent: retrieve relevant passages, then answer-or-abstain with a *self-consistency*
confidence and a citation to the supporting passage.

A single LLM self-reported confidence is poorly calibrated (the model says "100%" even when
it's confidently wrong). Instead we sample the answer several times and use agreement as the
confidence signal: genuinely answerable questions produce the same answer repeatedly, while
hallucinations tend to disagree across samples. The samples run concurrently so this stays
fast.

The raw model output is a *candidate* answer + confidence + citation. The decision to actually
answer or abstain is made downstream by the conformal threshold (conformal.py) — the model
proposes, the calibrated policy disposes.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from . import llm
from .retriever import HybridRetriever, TfidfRetriever

_NORM = re.compile(r"[^a-z0-9 ]")


def _norm(s: str) -> str:
    return _NORM.sub("", (s or "").lower()).strip()


class Agent:
    def __init__(self, contexts: list[str], k: int = 3, n_samples: int = 5,
                 retriever: str = "") -> None:
        self.contexts = contexts
        # default: hybrid (lexical + embeddings), which self-falls-back to lexical if no
        # embeddings are available. Set CRAG_RETRIEVER=tfidf to force lexical-only.
        choice = retriever or os.environ.get("CRAG_RETRIEVER", "hybrid")
        if choice == "tfidf":
            self.retriever = TfidfRetriever(contexts)
            self.retrieval_mode = "lexical"
        else:
            hr = HybridRetriever(contexts)
            self.retriever = hr
            self.retrieval_mode = hr.mode
        self.k = k
        self.n = n_samples

    def predict(self, question: str) -> dict:
        idx = self.retriever.search(question, k=self.k)
        numbered = "\n\n".join(f"[{j+1}] {self.contexts[i]}" for j, i in enumerate(idx))
        with ThreadPoolExecutor(max_workers=self.n) as ex:
            samples = list(ex.map(
                lambda t: llm.answer_or_abstain(question, numbered, temperature=t),
                [0.0] + [0.7] * (self.n - 1)))
        agg = _aggregate(samples, self.n)
        # map the cited passage (1-based, within retrieved) back to the corpus
        cite_local = agg.pop("cite_local")
        citation = None
        if 1 <= cite_local <= len(idx):
            cid = idx[cite_local - 1]
            citation = {"corpus_id": cid, "text": self.contexts[cid]}
        return {**agg, "retrieved": idx, "citation": citation}


def _aggregate(samples: list[dict], n: int) -> dict:
    """Self-consistency: confidence = fraction of samples that agree on the top answer; the
    citation is the passage most of those agreeing samples pointed to."""
    answerable_votes = sum(s["answerable"] for s in samples)
    votes: dict[str, list] = defaultdict(lambda: [0, "", []])
    for s in samples:
        if s["answerable"] and s["answer"]:
            key = _norm(s["answer"])
            votes[key][0] += 1
            votes[key][1] = s["answer"]
            votes[key][2].append(s.get("cite", 0))
    if votes:
        top = max(votes.values(), key=lambda v: v[0])
        confidence, answer = top[0] / n, top[1]
        cite_local = Counter(c for c in top[2] if c).most_common(1)
        cite_local = cite_local[0][0] if cite_local else 0
    else:
        confidence, answer, cite_local = 0.0, "", 0
    # the model's raw self-reported confidence (for the calibration ablation)
    sr = [s["confidence"] for s in samples if s["answerable"]]
    selfreport = sum(sr) / len(sr) if sr else 0.0
    return {"answerable": answerable_votes / n >= 0.5, "answer": answer,
            "confidence": confidence, "selfreport": selfreport, "cite_local": cite_local}


def decide(pred: dict, threshold: float) -> dict:
    """Apply the calibrated abstention policy to a raw prediction.

    Abstain if the model flagged it unanswerable OR its confidence is below the calibrated
    threshold. Otherwise answer."""
    answered = bool(pred.get("answerable")) and pred.get("confidence", 0.0) >= threshold
    return {"answered": answered,
            "answer": pred.get("answer", "") if answered else "",
            "confidence": pred.get("confidence", 0.0),
            "citation": pred.get("citation") if answered else None}
