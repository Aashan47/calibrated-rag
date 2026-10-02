"""Labelled evaluation sets.

Two datasets, chosen for different reasons:

* ``helpdesk`` — the product's own help centre (knowledge_base/) with 68 labelled support
  tickets: 46 answerable (gold answer + the article that contains it) and 22 plausible questions
  the help centre does **not** cover. This is what the shipped abstention threshold is
  calibrated on, so the guarantee applies to the domain the agent actually runs in.
* ``squad`` — a 300-question SQuAD 2.0 slice (140 unanswerable), the public benchmark, used to
  show the method transfers and to compare against published numbers.

Both return ``(contexts, items)`` where each item is
``{"question", "answers": [...], "is_impossible", "gold_ctx": int | None}``.
"""

from __future__ import annotations

import json
import os

from . import corpus, data

KB = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")


def load(name: str = "helpdesk") -> tuple[list[str], list[dict]]:
    if name == "squad":
        return data.load()
    if name != "helpdesk":
        raise ValueError(f"unknown dataset {name!r} (use 'helpdesk' or 'squad')")
    kb = corpus.load_knowledge_base()
    by_slug = {a["slug"]: i for i, a in enumerate(kb["articles"])}
    items = []
    for t in json.load(open(os.path.join(KB, "tickets.json"))):
        gold = by_slug.get(t["article"]) if t.get("article") else None
        if t.get("article") and gold is None:
            raise ValueError(f"ticket references unknown article {t['article']!r}")
        items.append({"question": t["question"], "answers": t["answers"],
                      "is_impossible": bool(t["is_impossible"]), "gold_ctx": gold})
    return kb["contexts"], items
