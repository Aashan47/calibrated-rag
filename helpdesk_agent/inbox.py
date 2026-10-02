"""The ticket inbox: the workflow around the agent.

A support product is a queue, not a question box. Tickets arrive (seeded from
knowledge_base/inbox.json, or added live), the agent handles them one at a time while you
watch, and each ends in a state a team can act on:

    new → handling → auto-resolved → sent          (reply approved and sent)
                   ↘ escalated                      (hand-off note attached, human queue)
                   ↘ error                          (model unavailable; retry)

Transitions are enforced (`TRANSITIONS`), a ticket stuck in `handling` (server died mid-run,
client vanished) becomes claimable again after `STALE_SECONDS`, and the inbox is capped so a
public demo cannot be filled without bound. State is in memory (one process); every change is
appended to the JSONL trace. Thread-safe.
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
TRANSITIONS = {
    "new": {"handling", "escalated"},
    "handling": {"auto-resolved", "escalated", "error", "new"},
    "auto-resolved": {"sent", "escalated", "new"},
    "escalated": {"new"},
    "sent": set(),                      # final
    "error": {"new", "handling", "escalated"},
}
STALE_SECONDS = 180
MAX_TICKETS = 200


class TransitionError(ValueError):
    pass


class Inbox:
    def __init__(self, seed_path: str | None = SEED, max_tickets: int = MAX_TICKETS) -> None:
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._tickets: dict[int, dict] = {}
        self.max_tickets = max_tickets
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
            if len(self._tickets) >= self.max_tickets:
                # drop the oldest *finished* ticket to make room; never drop live work
                for tid, t in sorted(self._tickets.items(), key=lambda kv: kv[1]["received"]):
                    if t["status"] in ("sent", "escalated", "error"):
                        del self._tickets[tid]
                        break
                else:
                    raise TransitionError("inbox is full")
            tid = next(self._ids)
            t = {"id": tid, "customer": (customer or "Customer")[:60],
                 "subject": (subject or "")[:120], "message": message.strip(),
                 "received": received or time.time(), "status": "new", "result": None,
                 "started_at": None, "handled_at": None, "sent_at": None}
            self._tickets[tid] = t
        trace.log({"inbox": "add", "id": tid, "subject": t["subject"][:80]})
        return self._public(t)

    def set_status(self, tid: int, status: str, result: dict | None = None) -> dict:
        if status not in VALID:
            raise TransitionError(f"unknown status {status!r}")
        with self._lock:
            t = self._tickets[tid]              # KeyError → caller maps to 404
            cur = t["status"]
            if cur == "handling" and status == "handling":
                if time.time() - (t["started_at"] or 0) < STALE_SECONDS:
                    raise TransitionError("ticket is already being handled")
            elif status not in TRANSITIONS[cur]:
                raise TransitionError(f"cannot move a ticket from {cur} to {status}")
            t["status"] = status
            if status == "handling":
                t["started_at"] = time.time()
            if status == "new":
                t["result"], t["started_at"], t["handled_at"], t["sent_at"] = None, None, None, None
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
