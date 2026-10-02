"""The editable help centre: the knowledge base the agent replies from, managed live.

A support team's fix for an escalation is usually "write the missing article". This module
makes that a loop you can close from the console: add, edit or delete an article, the
retriever is rebuilt, and the next ticket is handled against the new help centre with no
restart.

Design
* Article ids are stable (a counter), never list positions, so a ticket handled earlier still
  points at the right article after a deletion.
* Every agent run works on an immutable `Snapshot` (agent + articles + version) taken when
  the run starts. An edit mid-run can neither crash the run nor mislabel its citation.
* Edits are validated and capped (title/body length, article count, category slug), and
  live in memory for the process. On a public demo they are session-scoped: `reset()`
  restores the shipped help centre and `export()` hands them back. With a `write_dir` they
  are also written through to `knowledge_base/<category>/<slug>.md` (opt in: HDA_KB_WRITE=1).
* The shipped escalation threshold was calibrated on the shipped articles; the snapshot
  carries `edits` so the console can say when that calibration no longer strictly applies.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import threading

from . import trace

MAX_ARTICLES = 200
TITLE_MIN, TITLE_MAX = 3, 120
BODY_MIN, BODY_MAX = 20, 6000
CATEGORY_MAX = 40
_SLUG = re.compile(r"[^a-z0-9]+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class KBError(ValueError):
    """The edit cannot be applied; the message is safe to show to the editor."""


def slugify(s: str) -> str:
    s = _SLUG.sub("-", (s or "").lower()).strip("-")
    return s[:60] or "untitled"


def _text(raw: object, what: str, lo: int, hi: int, one_line: bool = False) -> str:
    if not isinstance(raw, str):
        raise KBError(f"{what} must be text")
    t = _CTRL.sub("", raw).replace("\r\n", "\n").replace("\r", "\n")
    if one_line:
        t = " ".join(t.split())
    else:
        t = "\n".join(" ".join(line.split()) for line in t.split("\n")).strip()
        t = re.sub(r"\n{3,}", "\n\n", t)
    if len(t) < lo:
        raise KBError(f"{what} is too short (at least {lo} characters)")
    if len(t) > hi:
        raise KBError(f"{what} is too long ({len(t)} characters; the limit is {hi})")
    return t


class Snapshot:
    """What one agent run sees: a fixed agent over fixed articles."""

    def __init__(self, agent, articles: list[dict], version: int, edits: int) -> None:
        self.agent = agent
        self.articles = articles              # aligned with agent.contexts
        self.version = version
        self.edits = edits
        self.by_id = {a["id"]: a for a in articles}

    def article(self, corpus_id: int) -> dict:
        return self.articles[corpus_id]

    def ids(self, corpus_ids) -> list[int]:
        return [self.articles[i]["id"] for i in corpus_ids]


class KnowledgeBase:
    def __init__(self, articles: list[dict], name: str, company: str,
                 categories: dict[str, str], agent_factory, write_dir: str | None = None,
                 max_articles: int = MAX_ARTICLES) -> None:
        """`articles`: [{slug, title, category, category_name, text}] as corpus.load_* returns.
        `agent_factory(contexts) -> Agent` is called on every change."""
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self._factory = agent_factory
        self.name, self.company = name, company
        self.max_articles = max_articles
        self.write_dir = write_dir
        self._categories: dict[str, str] = dict(categories)
        self._articles: list[dict] = []
        for a in articles:
            self._articles.append(self._row(a["title"], a["category"], a["text"], a.get("slug"),
                                            state=None))
            self._categories.setdefault(a["category"], a.get("category_name", a["category"]))
        self._shipped = [dict(a) for a in self._articles]
        self._shipped_categories = dict(self._categories)
        self.version = 0
        self.edits = 0
        self._snap: Snapshot | None = None
        self._rebuild()

    # -- helpers --------------------------------------------------------------------------
    def _row(self, title: str, category: str, text: str, slug: str | None = None,
             state: str | None = "added") -> dict:
        return {"id": next(self._ids), "slug": slug or slugify(title), "title": title,
                "category": category, "category_name": self._categories.get(category, category),
                "text": text, "words": len(text.split()), "state": state}

    def _rebuild(self) -> None:
        contexts = [a["text"] for a in self._articles]
        agent = self._factory(contexts)
        self.version += 1
        self._snap = Snapshot(agent, [dict(a) for a in self._articles], self.version, self.edits)

    def _validate(self, title, category, text, category_name=None,
                  exclude_id: int | None = None) -> tuple[str, str, str]:
        title = _text(title, "title", TITLE_MIN, TITLE_MAX, one_line=True)
        text = _text(text, "article body", BODY_MIN, BODY_MAX)
        if not isinstance(category, str):
            raise KBError("category must be text")
        cat = slugify(category)
        if cat == "untitled" or len(cat) > CATEGORY_MAX:
            raise KBError("category must be a short name (letters and numbers)")
        for a in self._articles:
            if a["id"] != exclude_id and a["category"] == cat \
                    and a["title"].lower() == title.lower():
                raise KBError(f"an article titled {title!r} already exists in this category")
        if cat not in self._categories:                 # only once every check has passed
            name = category_name if isinstance(category_name, str) and category_name.strip() \
                else category.strip()
            self._categories[cat] = _text(name, "category name", 1, 60, one_line=True)
        return title, cat, text

    def _find(self, aid: int) -> dict:
        for a in self._articles:
            if a["id"] == aid:
                return a
        raise KeyError(aid)

    # -- reads ----------------------------------------------------------------------------
    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snap  # type: ignore[return-value]

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(a) for a in self._articles]

    def categories(self) -> dict[str, str]:
        with self._lock:
            return dict(self._categories)

    def export(self) -> dict:
        with self._lock:
            return {"name": self.name, "company": self.company,
                    "categories": dict(self._categories),
                    "articles": [{k: a[k] for k in ("slug", "title", "category", "text")}
                                 for a in self._articles]}

    # -- mutations ------------------------------------------------------------------------
    def add(self, title, category, text, category_name=None) -> dict:
        with self._lock:
            if len(self._articles) >= self.max_articles:
                raise KBError(f"the help centre is full ({self.max_articles} articles)")
            title, cat, text = self._validate(title, category, text, category_name)
            slug = slugify(title)
            taken = {a["slug"] for a in self._articles if a["category"] == cat}
            n = 2
            while slug in taken:
                slug = f"{slugify(title)[:56]}-{n}"
                n += 1
            a = self._row(title, cat, text, slug)
            self._articles.append(a)
            self.edits += 1
            self._rebuild()
            self._persist("write", a)
        trace.log({"kb": "add", "id": a["id"], "title": title[:80]})
        return dict(a)

    def update(self, aid: int, title=None, category=None, text=None, category_name=None) -> dict:
        with self._lock:
            a = self._find(aid)
            old = dict(a)
            title, cat, text = self._validate(
                a["title"] if title is None else title,
                a["category"] if category is None else category,
                a["text"] if text is None else text, category_name, exclude_id=aid)
            if (title, cat, text) == (a["title"], a["category"], a["text"]):
                return dict(a)                                  # no-op: nothing to rebuild
            a.update({"title": title, "category": cat, "category_name": self._categories[cat],
                      "text": text, "words": len(text.split()),
                      "state": a["state"] or "edited"})
            self.edits += 1
            self._rebuild()
            if old["category"] != cat:
                self._persist("remove", old)
            self._persist("write", a)
        trace.log({"kb": "update", "id": aid})
        return dict(a)

    def delete(self, aid: int) -> dict:
        with self._lock:
            a = self._find(aid)
            if len(self._articles) == 1:
                raise KBError("the help centre cannot be empty")
            self._articles.remove(a)
            self.edits += 1
            self._rebuild()
            self._persist("remove", a)
        trace.log({"kb": "delete", "id": aid})
        return dict(a)

    def reset(self) -> int:
        """Back to the shipped help centre. Returns how many edits were discarded."""
        with self._lock:
            n = self.edits
            if not n:
                return 0
            for a in self._articles:
                if a["state"] == "added":
                    self._persist("remove", a)
            self._articles = [dict(a) for a in self._shipped]
            self._categories = dict(self._shipped_categories)
            self.edits = 0
            self._rebuild()
            for a in self._articles:
                self._persist("write", a)
        trace.log({"kb": "reset", "discarded": n})
        return n

    # -- optional write-through -------------------------------------------------------------
    def _persist(self, action: str, a: dict) -> None:
        """Mirror an edit to <write_dir>/<category>/<slug>.md. Never raises: the in-memory
        help centre is the source of truth for the running process."""
        if not self.write_dir:
            return
        try:
            d = os.path.join(self.write_dir, slugify(a["category"]))
            path = os.path.join(d, slugify(a["slug"]) + ".md")
            if action == "write":
                os.makedirs(d, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"# {a['title']}\n\n{a['text']}\n")
                idx = os.path.join(self.write_dir, "index.json")
                meta = {"name": self.name, "company": self.company}
                if os.path.exists(idx):
                    with open(idx) as f:
                        meta = json.load(f)
                meta["categories"] = {**meta.get("categories", {}), **self._categories}
                with open(idx, "w") as f:
                    json.dump(meta, f, indent=2)
            elif os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            trace.log({"kb": "persist_failed", "error": str(exc)[:200]})
