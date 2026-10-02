"""The agent: a bounded decision loop over a search tool, ending in a *calibrated*
answer-or-abstain.

This is an agent, not a fixed pipeline. Each step the LLM looks at what it has retrieved so
far and **chooses its next action** — search again with a reformulated query, answer now, or
give up (abstain) — instead of running a hard-coded retrieve→answer path. State (the passages
gathered, the queries already tried) carries across steps, and the loop is bounded so it
always terminates.

When the agent decides to answer, the *answer itself* is produced with a **self-consistency**
confidence: a single LLM self-reported confidence is badly calibrated ("100%" even when wrong),
so we sample the answer several times and use agreement as the signal — genuine answers repeat,
hallucinations scatter. That confidence then faces the calibrated conformal threshold
(conformal.py) downstream: the agent proposes, the calibrated policy disposes. Two independent
ways to abstain — the agent judging the corpus can't support an answer, and the confidence
falling below the risk-controlled threshold — which is exactly the reliability we're after.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from . import llm
from .retriever import HybridRetriever, TfidfRetriever
from .tools import SearchTool

_NORM = re.compile(r"[^a-z0-9 ]")


def _norm(s: str) -> str:
    return _NORM.sub("", (s or "").lower()).strip()


_DECIDE_PROMPT = """You are a retrieval agent answering a QUESTION strictly from a document \
corpus. You can call a search tool to pull passages. Decide the single best NEXT ACTION.

QUESTION: {question}

PASSAGES GATHERED SO FAR:
{passages}

SEARCH QUERIES ALREADY TRIED: {queries}

Choose one:
- "answer"  — the gathered passages clearly contain the answer. Answer now.
- "search"  — they don't yet, but a BETTER, DIFFERENT query might find it. Provide "query" \
(reworded, not one already tried — try synonyms, names, or more specific terms).
- "abstain" — the answer is very unlikely to be anywhere in this corpus; stop.

Return ONLY JSON, no prose:
{{"action": "answer|search|abstain", "query": "<only if action=search>", "reason": "<brief>"}}
JSON:"""


class Agent:
    def __init__(self, contexts: list[str], k: int = 3, n_samples: int = 5,
                 max_steps: int = 3, retriever: str = "") -> None:
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
        self.max_steps = max(1, int(os.environ.get("CRAG_MAX_STEPS", max_steps)))
        self.search = SearchTool(self.retriever, contexts, k=k)

    # -- the agent loop -----------------------------------------------------------------
    def predict(self, question: str) -> dict:
        """Run the decision loop and return a raw prediction (the conformal threshold is
        applied later by `decide`). The returned dict also carries `steps` — the actions the
        agent took — so callers can show the agent's reasoning trace."""
        gathered: list[int] = []        # corpus ids, in discovery order
        seen: set[int] = set()
        queries: list[str] = []
        steps: list[dict] = []

        def run_search(q: str) -> int:
            queries.append(q)
            hits = self.search(q)
            added = 0
            for h in hits:
                cid = h["corpus_id"]
                if cid not in seen:
                    seen.add(cid)
                    gathered.append(cid)
                    added += 1
            steps.append({"action": "search", "query": q, "found": len(hits), "new": added})
            return added

        # Seed: always start by searching the question as asked.
        run_search(question)

        for _ in range(self.max_steps):
            d = self._decide(question, gathered, queries)
            action = d.get("action", "answer")
            if action == "abstain":
                steps.append({"action": "abstain", "reason": d.get("reason", "")})
                return self._abstained(gathered, steps)
            if action == "search":
                q = (d.get("query") or "").strip()
                steps.append({"action": "decide", "next": "search", "query": q,
                              "reason": d.get("reason", "")})
                if not q or _norm(q) in {_norm(x) for x in queries}:
                    break           # no genuinely new query to try → stop and answer
                run_search(q)
                continue
            # action == "answer" (default / fall-through)
            steps.append({"action": "decide", "next": "answer", "reason": d.get("reason", "")})
            break

        return self._finish(question, gathered, steps)

    def _decide(self, question: str, gathered: list[int], queries: list[str]) -> dict:
        """Ask the LLM what to do next. Defaults to 'answer' on any parse failure so a flaky
        decision never strands the loop (the confidence gate still protects correctness)."""
        if not gathered:
            return {"action": "search", "query": question, "reason": "no passages yet"}
        passages = "\n\n".join(f"[{j+1}] {self.contexts[i]}" for j, i in enumerate(gathered))
        tried = "; ".join(queries) if queries else "(none)"
        raw = llm.complete(_DECIDE_PROMPT.format(
            question=question, passages=passages[:8000], queries=tried), max_tokens=200)
        try:
            m = re.search(r"\{.*\}", raw, re.S)
            obj = json.loads(m.group(0)) if m else {}
            action = str(obj.get("action", "")).lower().strip()
            if action not in {"answer", "search", "abstain"}:
                action = "answer"
            return {"action": action, "query": obj.get("query", ""),
                    "reason": str(obj.get("reason", ""))[:200]}
        except Exception:  # noqa: BLE001
            return {"action": "answer", "reason": "decision parse failed"}

    def _finish(self, question: str, gathered: list[int], steps: list[dict]) -> dict:
        """The agent chose to answer: produce a self-consistency answer over everything it
        gathered, and map the citation back to the corpus."""
        if not gathered:
            return self._abstained(gathered, steps)
        numbered = "\n\n".join(f"[{j+1}] {self.contexts[i]}" for j, i in enumerate(gathered))
        with ThreadPoolExecutor(max_workers=self.n) as ex:
            samples = list(ex.map(
                lambda t: llm.answer_or_abstain(question, numbered, temperature=t),
                [0.0] + [0.7] * (self.n - 1)))
        agg = _aggregate(samples, self.n)
        cite_local = agg.pop("cite_local")
        citation = None
        if 1 <= cite_local <= len(gathered):
            cid = gathered[cite_local - 1]
            citation = {"corpus_id": cid, "text": self.contexts[cid]}
        return {**agg, "retrieved": gathered, "citation": citation, "steps": steps}

    @staticmethod
    def _abstained(gathered: list[int], steps: list[dict]) -> dict:
        return {"answerable": False, "answer": "", "confidence": 0.0, "selfreport": 0.0,
                "retrieved": gathered, "citation": None, "steps": steps}


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
