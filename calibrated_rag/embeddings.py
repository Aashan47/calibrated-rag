"""Gemini text embeddings (stdlib only), with an on-disk cache and graceful fallback.

Used by the hybrid retriever. If no API key is set or the API errors, `embed` returns None
and the retriever quietly falls back to lexical-only — the agent keeps working.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request

MODEL = os.environ.get("CRAG_EMBED_MODEL", "gemini-embedding-001")
_ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/models/"
             "{model}:embedContent?key={key}")
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "embed_cache.json")
_cache: dict[str, list[float]] | None = None


def _key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMENI_API_KEY")


def available() -> bool:
    return bool(_key())


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.load(open(_CACHE_PATH))
        except Exception:  # noqa: BLE001
            _cache = {}
    return _cache


def _save_cache() -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        json.dump(_cache, open(_CACHE_PATH, "w"))
    except Exception:  # noqa: BLE001
        pass


def embed(text: str) -> list[float] | None:
    """Embed one string (cached). None on any failure."""
    key = _key()
    if not key:
        return None
    cache = _load_cache()
    h = hashlib.md5(f"{MODEL}:{text}".encode()).hexdigest()
    if h in cache:
        return cache[h]
    url = _ENDPOINT.format(model=MODEL, key=key)
    body = {"content": {"parts": [{"text": text[:8000]}]}}
    try:
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=40) as resp:
            vals = json.load(resp)["embedding"]["values"]
    except Exception:  # noqa: BLE001
        return None
    cache[h] = vals
    return vals


def embed_many(texts: list[str], persist: bool = True) -> list[list[float]] | None:
    """Embed a list (cached). None if any embedding fails (so the caller can fall back)."""
    out = []
    for t in texts:
        v = embed(t)
        if v is None:
            return None
        out.append(v)
    if persist:
        _save_cache()
    return out
