"""The ticket inbox: the workflow around the agent.

A support product is a queue, not a question box. Tickets arrive (seeded from
knowledge_base/inbox.json, or added live), the agent handles them one at a time while you
watch, and each ends in a state a team can act on:

    new → handling → auto-resolved → sent          (reply approved and sent)
                   ↘ escalated                      (hand-off note attached, human queue)
                   ↘ error                          (model unavailable; retry)

State is in memory (one process, one demo) and every transition is appended to the JSONL
trace so the session is auditable. Thread-safe: the server handles tickets concurrently.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time

from . import trace

SEED = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "inbox.json")

VALID = {"new", "handling", "auto-resolved", "escalated", "sent", "error"}


class Inbox:
    def __init__(self, seed_path: str | None = SEED) -> None:
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._tickets: dict[int, dict] = {}
        if seed_path and os.path.exists(seed_path):
            with open(seed_path) as f:
                seed = json.load(f)
            now = time.time()
            for i, t in enumerate(seed):
                self.add(t["message"], customer=t.get("customer", ""),
                         subject=t.get("subject", ""),
                         received=now - 60 * t.get("minutes_ago", 5 * (len(seed) - i)))

    # -- mutations ------------------------------------------------------------------------
    def add(self, message: str, customer: str = "", subject: str = "",
            received: float | None = None) -> dict:
        with self._lock:
            tid = next(self._ids)
            t = {"id": tid, "customer": customer or "Customer", "subject": subject or "",
                 "message": message.strip(), "received": received or time.time(),
                 "status": "new", "result": None, "handled_at": None, "sent_at": None}
            self._tickets[tid] = t
        trace.log({"inbox": "add", "id": tid, "subject": subject[:80]})
        return self._public(t)

    def set_status(self, tid: int, status: str, result: dict | None = None) -> dict:
        if status not in VALID:
            raise ValueError(f"bad status {status!r}")
        with self._lock:
            t = self._tickets[tid]
            t["status"] = status
            if result is not None:
                t["result"] = result
                t["handled_at"] = time.time()
            if status == "sent":
                t["sent_at"] = time.time()
        trace.log({"inbox": "status", "id": tid, "status": status})
        return self._public(t)

    # -- reads ----------------------------------------------------------------------------
    def get(self, tid: int) -> dict:
        with self._lock:
            return dict(self._tickets[tid])

    def list(self) -> list[dict]:
        with self._lock:
            return [self._public(t) for t in sorted(self._tickets.values(),
                                                    key=lambda t: -t["received"])]

    def counts(self) -> dict:
        with self._lock:
            c = {s: 0 for s in VALID}
            for t in self._tickets.values():
                c[t["status"]] += 1
            c["total"] = len(self._tickets)
            return c

    @staticmethod
    def _public(t: dict) -> dict:
        r = t.get("result") or {}
        return {**{k: t[k] for k in ("id", "customer", "subject", "message", "received",
                                     "status", "handled_at", "sent_at")},
                "outcome": ({"status": r.get("status"), "reason_code": r.get("reason_code"),
                             "confidence": r.get("confidence"), "reply": r.get("reply"),
                             "citation": (r.get("citation") or {}).get("title")}
                            if r else None)}
