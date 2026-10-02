"""Where the agent's knowledge comes from.

Three sources, in priority order:

1. ``data/index.json`` — a corpus *you* ingested with ``ingest.py`` (your own help centre).
2. ``knowledge_base/`` — the committed help centre for the fictional Northwind Workspace: 30
   markdown articles in six categories. This is the default, and it is what the shipped
   abstention threshold was calibrated on (knowledge_base/tickets.json).
3. The SQuAD 2.0 slice — benchmark only; never used by the product.
"""

from __future__ import annotations

import glob
import json
import os
import re

HERE = os.path.dirname(__file__)
INDEX = os.path.join(HERE, "..", "data", "index.json")
KB = os.path.join(HERE, "..", "knowledge_base")
RESULTS = os.path.join(HERE, "..", "results", "helpdesk", "results.json")

# Fallback when nothing is calibrated for the corpus in use: answer only when at least
# 3 of 5 independent samples agree.
DEFAULT_THRESHOLD = 0.6


def load_knowledge_base(path: str = KB) -> dict:
    """Read knowledge_base/<category>/<slug>.md. The first '# heading' is the article title."""
    with open(os.path.join(path, "index.json")) as f:
        meta = json.load(f)
    cat_names = meta.get("categories", {})
    articles = []
    for md in sorted(glob.glob(os.path.join(path, "*", "*.md"))):
        cat = os.path.basename(os.path.dirname(md))
        slug = os.path.splitext(os.path.basename(md))[0]
        with open(md, encoding="utf-8") as f:
            text = f.read().strip()
        m = re.match(r"#\s*(.+)\n", text)
        title = m.group(1).strip() if m else slug
        body = text[m.end():].strip() if m else text
        articles.append({"slug": slug, "title": title, "category": cat,
                         "category_name": cat_names.get(cat, cat), "text": body})
    return {"name": meta.get("name", "help centre"), "company": meta.get("company", ""),
            "articles": articles, "contexts": [a["text"] for a in articles]}


def load_corpus() -> dict:
    """Return {'contexts', 'name', 'company', 'is_custom', 'articles'}.

    ``articles`` is per-passage presentation metadata: {'slug','title','category',
    'category_name'}; for an ingested corpus the title is the source file name."""
    if os.path.exists(INDEX):
        obj = json.load(open(INDEX))
        srcs = obj.get("sources") or [None] * len(obj["contexts"])
        arts = [{"slug": f"doc-{i}", "title": (s or f"passage {i}"), "category": "docs",
                 "category_name": "Your documents"} for i, s in enumerate(srcs)]
        return {"contexts": obj["contexts"], "name": obj.get("name", "custom documents"),
                "company": obj.get("name", "your company"), "is_custom": True, "articles": arts}
    kb = load_knowledge_base()
    return {"contexts": kb["contexts"], "name": kb["name"], "company": kb["company"],
            "is_custom": False,
            "articles": [{k: a[k] for k in ("slug", "title", "category", "category_name")}
                         for a in kb["articles"]]}


def load_contexts() -> tuple[list[str], str, bool]:
    c = load_corpus()
    return c["contexts"], c["name"], c["is_custom"]


def abstention_threshold(is_custom: bool) -> float:
    """HDA_THRESHOLD wins; else the threshold calibrated on the help centre's labelled tickets
    (results/helpdesk/results.json); an ingested corpus has no labels, so it gets the default
    until you calibrate on your own tickets (see EVALUATION.md)."""
    env = os.environ.get("HDA_THRESHOLD")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    if is_custom:
        return DEFAULT_THRESHOLD
    try:
        return float(json.load(open(RESULTS))["calibrated_threshold"])
    except Exception:  # noqa: BLE001
        return DEFAULT_THRESHOLD
