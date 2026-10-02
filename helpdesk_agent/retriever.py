"""Retrieval. Two retrievers, same interface (`search(query, k) -> list[int]`):

- TfidfRetriever: lexical, stdlib only, no API.
- HybridRetriever: fuses lexical (TF-IDF) and dense (Gemini embeddings) rankings with
  Reciprocal Rank Fusion. Falls back to lexical-only if embeddings are unavailable, so the
  agent always works.

RRF is used instead of score-normalization because it's robust and parameter-light: a
document's fused score is sum_r 1/(K + rank_r(d)) over the retrievers r.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from . import embeddings

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


class TfidfRetriever:
    def __init__(self, docs: list[str]) -> None:
        self.docs = docs
        self.tf = [Counter(_tok(d)) for d in docs]
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {w: math.log((n + 1) / (df_w + 1)) + 1.0 for w, df_w in df.items()}
        self.vecs = [self._vec(c) for c in self.tf]
        self.norms = [math.sqrt(sum(v * v for v in vec.values())) or 1e-9 for vec in self.vecs]

    def _vec(self, counts: Counter) -> dict:
        return {w: (1 + math.log(c)) * self.idf.get(w, 0.0) for w, c in counts.items()}

    def ranked(self, query: str) -> list[int]:
        q = self._vec(Counter(_tok(query)))
        qn = math.sqrt(sum(v * v for v in q.values())) or 1e-9
        scores = []
        for i, vec in enumerate(self.vecs):
            dot = sum(v * vec.get(w, 0.0) for w, v in q.items())
            scores.append((dot / (qn * self.norms[i]), i))
        scores.sort(reverse=True)
        # a document that shares no term with the query is not a match, however short the
        # list gets; returning it would hand the agent irrelevant passages to "judge"
        return [i for sc, i in scores if sc > 0]

    def search(self, query: str, k: int = 3) -> list[int]:
        return self.ranked(query)[:k]


class HybridRetriever:
    """Lexical + dense retrieval fused with Reciprocal Rank Fusion. Lexical-only fallback."""

    def __init__(self, docs: list[str], k_rrf: int = 60) -> None:
        self.docs = docs
        self.k_rrf = k_rrf
        self.lexical = TfidfRetriever(docs)
        self.doc_embs = embeddings.embed_many(docs) if embeddings.available() else None
        self.mode = "hybrid" if self.doc_embs else "lexical"

    def _dense_ranked(self, query: str) -> list[int] | None:
        if not self.doc_embs:
            return None
        qe = embeddings.embed(query)
        if qe is None:
            return None
        sims = [(_cosine(qe, de), i) for i, de in enumerate(self.doc_embs)]
        sims.sort(reverse=True)
        return [i for _, i in sims]

    def search(self, query: str, k: int = 3) -> list[int]:
        lex = self.lexical.ranked(query)
        dense = self._dense_ranked(query)
        if dense is None:
            return lex[:k]
        fused: dict[int, float] = defaultdict(float)
        for rank, i in enumerate(lex):
            fused[i] += 1.0 / (self.k_rrf + rank + 1)
        for rank, i in enumerate(dense):
            fused[i] += 1.0 / (self.k_rrf + rank + 1)
        return [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])][:k]
