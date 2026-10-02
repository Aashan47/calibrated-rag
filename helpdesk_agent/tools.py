"""Tools the agent can call.

Keeping tools behind a small, explicit interface is what makes this an *agent* and not a
fixed pipeline: the control loop (agent.py) decides *when* and *with what query* to call a
tool, rather than running a hard-coded retrieve→answer path. Today there is one tool —
`search` — but the loop treats it as a capability it chooses to use, and new tools (a web
lookup, a calculator, a second corpus) would plug in here without touching the loop's logic.
"""

from __future__ import annotations


class SearchTool:
    """Retrieve passages from the document corpus for a (possibly reformulated) query.

    Returns a list of {'corpus_id': int, 'text': str}. The agent calls this with its own
    query wording — it may reformulate after seeing weak results — and accumulates the hits.
    """

    name = "search"
    description = ("search(query: str) -> passages. Look up the document corpus for passages "
                   "relevant to a query. Use a focused query; reformulate if results look off.")

    def __init__(self, retriever, contexts: list[str], k: int = 3) -> None:
        self._retriever = retriever
        self._contexts = contexts
        self.k = k

    def __call__(self, query: str, k: int | None = None) -> list[dict]:
        ids = self._retriever.search(query, k=k or self.k)
        return [{"corpus_id": i, "text": self._contexts[i]} for i in ids]
