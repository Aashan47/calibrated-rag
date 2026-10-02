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

A third outcome is kept strictly separate from both: the model being *unavailable*. A failed
API call is reported as `error`, never as "unanswerable" — an agent that blames the question
when the API was down is misreporting, and that is the one thing a reliability layer must not do.
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
        # CRAG_SAMPLES lets a rate-limited deployment trade a little calibration quality
        # for fewer calls per question (each question costs ~1 decision + N samples).
        self.n = max(1, int(os.environ.get("CRAG_SAMPLES", n_samples)))
        # max_steps = 0 disables the decision loop (single retrieve→answer) — used as the
        # ablation baseline in eval/ablation.py to measure what the agent loop actually buys.
        self.max_steps = max(0, int(os.environ.get("CRAG_MAX_STEPS", max_steps)))
        self.search = SearchTool(self.retriever, contexts, k=k)

    # -- the agent loop -----------------------------------------------------------------
    def predict(self, question: str) -> dict:
        """Run the decision loop and return a raw prediction (the conformal threshold is
        applied later by `decide`). The returned dict also carries `steps` — the actions the
        agent took — so callers can show the agent's reasoning trace, `llm_calls`, and an
        `error` field when the model itself was unavailable."""
        gathered: list[int] = []        # corpus ids, in discovery order
        seen: set[int] = set()
        queries: list[str] = []
        steps: list[dict] = []
        calls = [0]                     # LLM calls made (boxed so closures can bump it)

        def run_search(q: str) -> int:
            queries.append(q)
            hits = self.search(q)
            ids = [h["corpus_id"] for h in hits]
            added = 0
            for cid in ids:
                if cid not in seen:
                    seen.add(cid)
                    gathered.append(cid)
                    added += 1
            # record the ids so offline trajectory analysis (eval/behavior.py) can measure
            # retrieval recall before/after reformulation without re-querying.
            steps.append({"action": "search", "query": q, "ids": ids, "new": added})
            return added

        # Seed: always start by searching the question as asked.
        run_search(question)

        for _ in range(self.max_steps):
            d = self._decide(question, gathered, queries)
            calls[0] += 1
            action = d.get("action", "answer")
            if action == "abstain":
                steps.append({"action": "abstain", "reason": d.get("reason", "")})
                return self._abstained(gathered, steps, calls[0])
            if action == "search":
                q = (d.get("query") or "").strip()
                steps.append({"action": "decide", "next": "search", "query": q,
                              "reason": d.get("reason", "")})
                if not q or _norm(q) in {_norm(x) for x in queries}:
                    break           # no genuinely new query to try → stop and answer
                run_search(q)
                continue
            # action == "answer" (default / fall-through)
            step = {"action": "decide", "next": "answer", "reason": d.get("reason", "")}
            if d.get("error"):
                step["error"] = d["error"]   # the decision was never made — the API failed
            steps.append(step)
            break

        return self._finish(question, gathered, steps, calls)

    def _decide(self, question: str, gathered: list[int], queries: list[str]) -> dict:
        """Ask the LLM what to do next.

        If the model is unreachable we fall through to an answer attempt (which will surface
        the same outage) but we *say so* via `error` — the trace must never claim the agent
        "judged the passages sufficient" when no judgement happened."""
        if not gathered:
            return {"action": "search", "query": question, "reason": "no passages yet"}
        passages = "\n\n".join(f"[{j+1}] {self.contexts[i]}" for j, i in enumerate(gathered))
        tried = "; ".join(queries) if queries else "(none)"
        try:
            raw = llm.complete(_DECIDE_PROMPT.format(
                question=question, passages=passages[:8000], queries=tried), max_tokens=200)
        except llm.LLMError as exc:
            return {"action": "answer", "reason": "decision call failed", "error": str(exc)}
        try:
            m = re.search(r"\{.*\}", raw, re.S)
            obj = json.loads(m.group(0)) if m else {}
            action = str(obj.get("action", "")).lower().strip()
            if action not in {"answer", "search", "abstain"}:
                action = "answer"
            return {"action": action, "query": obj.get("query", ""),
                    "reason": str(obj.get("reason", ""))[:200]}
        except Exception:  # noqa: BLE001
            return {"action": "answer", "reason": "decision response was not parseable"}

    def _finish(self, question: str, gathered: list[int], steps: list[dict],
                calls: list[int]) -> dict:
        """The agent chose to answer: produce a self-consistency answer over everything it
        gathered, and map the citation back to the corpus."""
        if not gathered:
            return self._abstained(gathered, steps, calls[0])
        numbered = "\n\n".join(f"[{j+1}] {self.contexts[i]}" for j, i in enumerate(gathered))
        with ThreadPoolExecutor(max_workers=self.n) as ex:
            samples = list(ex.map(
                lambda t: llm.answer_or_abstain(question, numbered, temperature=t),
                [0.0] + [0.7] * (self.n - 1)))
        calls[0] += self.n
        agg = _aggregate(samples, self.n)
        cite_local = agg.pop("cite_local")
        citation = None
        if 1 <= cite_local <= len(gathered):
            cid = gathered[cite_local - 1]
            citation = {"corpus_id": cid, "text": self.contexts[cid]}
        return {**agg, "retrieved": gathered, "citation": citation, "steps": steps,
                "llm_calls": calls[0]}

    @staticmethod
    def _abstained(gathered: list[int], steps: list[dict], calls: int) -> dict:
        return {"answerable": False, "answer": "", "confidence": 0.0, "selfreport": 0.0,
                "votes": 0, "errors": 0, "retrieved": gathered, "citation": None,
                "steps": steps, "llm_calls": calls}


def _aggregate(samples: list[dict], n: int) -> dict:
    """Self-consistency: confidence = fraction of samples that agree on the top answer; the
    citation is the passage most of those agreeing samples pointed to.

    Samples that *failed* (API error) are counted separately. If every sample failed, the
    result carries `error` and must be reported as an outage, not an abstention."""
    errors = [s["error"] for s in samples if s.get("error")]
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
    out = {"answerable": answerable_votes / n >= 0.5, "answer": answer,
           "confidence": confidence, "selfreport": selfreport, "cite_local": cite_local,
           "votes": answerable_votes, "errors": len(errors)}
    if errors and len(errors) == len(samples):
        out["error"] = errors[0]
    return out


def decide(pred: dict, threshold: float) -> dict:
    """Apply the calibrated abstention policy to a raw prediction.

    Abstain if the model flagged it unanswerable OR its confidence is below the calibrated
    threshold. Otherwise answer. (An `error` prediction is never answered; callers should
    report it as an outage rather than an abstention.)"""
    answered = (not pred.get("error")) and bool(pred.get("answerable")) \
        and pred.get("confidence", 0.0) >= threshold
    return {"answered": answered,
            "answer": pred.get("answer", "") if answered else "",
            "confidence": pred.get("confidence", 0.0),
            "citation": pred.get("citation") if answered else None}
