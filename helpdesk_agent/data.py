"""Load a balanced slice of SQuAD 2.0 (answerable + unanswerable questions).

SQuAD 2.0 is the right benchmark here precisely because ~1/3 of its questions are
*unanswerable* from the given passage — so it directly measures whether a system knows when
to abstain instead of hallucinating.
"""

from __future__ import annotations

import json
import os
import random
import urllib.request

_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "squad_dev_v2.json")


def _download() -> dict:
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    if not os.path.exists(_CACHE):
        urllib.request.urlretrieve(_URL, _CACHE)
    with open(_CACHE) as f:
        return json.load(f)


def load(n_answerable: int = 160, n_unanswerable: int = 140, seed: int = 7):
    """Return (corpus_contexts, items). Each item: {question, answers, is_impossible, gold_ctx}."""
    raw = _download()
    paras = []
    for article in raw["data"]:
        for p in article["paragraphs"]:
            paras.append(p)
    random.Random(seed).shuffle(paras)

    contexts: list[str] = []
    ctx_index: dict[str, int] = {}
    items: list[dict] = []
    ans = imp = 0
    for p in paras:
        ctx = p["context"].strip()
        if ctx not in ctx_index:
            ctx_index[ctx] = len(contexts)
            contexts.append(ctx)
        for qa in p["qas"]:
            if qa.get("is_impossible"):
                if imp >= n_unanswerable:
                    continue
                items.append({"question": qa["question"], "answers": [],
                              "is_impossible": True, "gold_ctx": ctx_index[ctx]})
                imp += 1
            else:
                if ans >= n_answerable or not qa.get("answers"):
                    continue
                items.append({"question": qa["question"],
                              "answers": [a["text"] for a in qa["answers"]],
                              "is_impossible": False, "gold_ctx": ctx_index[ctx]})
                ans += 1
        if ans >= n_answerable and imp >= n_unanswerable:
            break
    random.Random(seed + 1).shuffle(items)
    return contexts, items
