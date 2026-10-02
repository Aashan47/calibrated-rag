"""Where the agent's documents come from.

If an ingested index exists (data/index.json, written by ingest.py), use it — so the demo
runs over *your own* documents. Otherwise fall back to the SQuAD 2.0 slice so everything
works out of the box with no setup.
"""

from __future__ import annotations

import json
import os

from . import data as squad

INDEX = os.path.join(os.path.dirname(__file__), "..", "data", "index.json")
DEMO = os.path.join(os.path.dirname(__file__), "demo_corpus.json")   # committed, instant boot
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "results.json")

# Sensible fallback for an uncalibrated corpus (require >= 3/5 self-consistency samples).
DEFAULT_THRESHOLD = 0.6


def load_contexts() -> tuple[list[str], str, bool]:
    """Return (passages, source_name, is_custom)."""
    if os.path.exists(INDEX):     # a corpus you ingested
        obj = json.load(open(INDEX))
        return obj["contexts"], obj.get("name", "custom documents"), True
    if os.path.exists(DEMO):      # committed demo corpus — instant, no download
        obj = json.load(open(DEMO))
        return obj["contexts"], obj.get("name", "SQuAD 2.0 passages (demo corpus)"), False
    contexts, _ = squad.load()    # fallback: download the SQuAD slice
    return contexts, "SQuAD 2.0 passages (demo corpus)", False


def abstention_threshold(is_custom: bool) -> float:
    """Pick the abstention threshold.

    The conformal threshold in results.json is calibrated *on the SQuAD benchmark* and only
    applies to that corpus. For your own documents there's no labeled data to calibrate on,
    so we use a sensible default (override with CRAG_THRESHOLD, or run a calibration set to
    get a real guarantee for your corpus)."""
    env = os.environ.get("CRAG_THRESHOLD")
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
