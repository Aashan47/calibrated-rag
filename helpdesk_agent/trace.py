"""Append-only JSONL trace of every served query — basic production observability.

One line per request: timestamp, question, the decision (answer/abstain), confidence, latency.
Good enough to answer "what did it do, and how often did it abstain?" after the fact; wire a
real sink (OTel, a warehouse) here in production.
"""

from __future__ import annotations

import json
import os
import time

TRACE = os.path.join(os.path.dirname(__file__), "..", "runs", "traces.jsonl")


def log(record: dict) -> None:
    try:
        os.makedirs(os.path.dirname(TRACE), exist_ok=True)
        with open(TRACE, "a") as f:
            f.write(json.dumps({"ts": round(time.time(), 3), **record}) + "\n")
    except Exception:  # noqa: BLE001 — telemetry must never break a request
        pass
