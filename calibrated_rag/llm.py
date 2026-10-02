"""Minimal, model-agnostic LLM client (stdlib only).

The agent only needs one capability: given a question and some retrieved context, either
answer from the context or say it can't, and report how confident it is. This file hides
the provider behind `answer_or_abstain`, so swapping Gemini for Claude/GPT is a one-function
change.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

_MODEL = os.environ.get("CRAG_MODEL", "gemini-2.5-flash")
_ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/models/"
             "{model}:generateContent?key={key}")


def _key() -> str:
    k = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMENI_API_KEY")
    if not k:
        raise RuntimeError("Set GEMINI_API_KEY (or GEMENI_API_KEY) in the environment.")
    return k


class LLMError(RuntimeError):
    """The model could not be reached or refused the request. This is an *availability*
    failure, not a judgement about the question, and callers must keep the two apart:
    an agent that reports "unanswerable" when the API was simply down is lying."""


def _describe(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        code = exc.code
        if code == 429:
            return "rate limited by the model API (HTTP 429)"
        if code in (400, 401, 403):
            return f"model API rejected the request or key (HTTP {code})"
        return f"model API error (HTTP {code})"
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return "could not reach the model API"
    return str(exc) or exc.__class__.__name__


def _generate(prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
    url = _ENDPOINT.format(model=_MODEL, key=_key())
    gen = {"maxOutputTokens": max_tokens, "temperature": temperature}
    if "flash" in _MODEL:
        gen["thinkingConfig"] = {"thinkingBudget": 0}
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    last = None
    for attempt in range(4):   # retry transient errors / rate limits
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
            cand = (data.get("candidates") or [{}])[0]
            return "".join(p.get("text", "")
                           for p in (cand.get("content", {}) or {}).get("parts", []))
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (400, 401, 403, 404):
                break                      # not transient: don't burn 20s retrying
            time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise LLMError(_describe(last) if last else "generation failed")


def complete(prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
    """Generic text completion (used by the agent's decision step).

    Raises LLMError when the model is unavailable so the caller can report *that*,
    rather than silently proceeding as if the model had answered."""
    try:
        return _generate(prompt, max_tokens=max_tokens, temperature=temperature)
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMError(_describe(exc)) from exc


_PROMPT = """You are a careful question-answering assistant. Answer the QUESTION using ONLY \
the numbered PASSAGES below. If the passages do not contain enough information to answer, do \
not guess — mark it unanswerable.

Return ONLY a JSON object, no prose, no markdown:
{{"answerable": <true|false>, "answer": "<short exact answer, or empty string>", \
"cite": <the passage number your answer comes from, or 0 if none>, \
"confidence": <integer 0-100: how confident you are the answer is correct AND supported by a passage>}}

PASSAGES:
{context}

QUESTION: {question}
JSON:"""


def answer_or_abstain(question: str, context: str, temperature: float = 0.0) -> dict:
    """One sample: {'answerable': bool, 'answer': str, 'cite': int, 'confidence': float}.

    Never raises. A failed call comes back as a non-answer *with an `error` field* so the
    aggregator can tell "the model said no" from "the model never answered" (a missing
    answer is better than a hallucinated one, but it must not be mislabelled as a
    judgement)."""
    blank = {"answerable": False, "answer": "", "cite": 0, "confidence": 0.0}
    try:
        raw = _generate(_PROMPT.format(context=context[:8000], question=question),
                        temperature=temperature)
    except Exception as exc:  # noqa: BLE001
        return {**blank, "error": _describe(exc)}
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        obj = json.loads(m.group(0)) if m else {}
        conf = float(obj.get("confidence", 0))
        conf = max(0.0, min(1.0, conf / 100.0 if conf > 1 else conf))
        try:
            cite = int(obj.get("cite", 0) or 0)
        except (TypeError, ValueError):
            cite = 0
        return {"answerable": bool(obj.get("answerable", False)),
                "answer": str(obj.get("answer", "")).strip(),
                "cite": cite, "confidence": conf}
    except Exception:  # noqa: BLE001
        return {**blank, "error": "model returned an unparseable response"}
