"""Guardrails around the agent: what comes in, how often, and whether a reply may go out.

Three concerns, kept out of the agent so they are easy to audit and test on their own:

* **Input** — `clean_message` normalises a customer message and rejects what the agent should
  never see (empty, oversized, control characters). It does not try to detect prompt injection
  by pattern; the prompts treat the message as untrusted data and the output guards below
  catch the consequence (an ungrounded reply) rather than the attempt.
* **Load** — `RateLimiter` is a per-key token bucket (used per client IP) and `Gate` bounds
  how many agent runs may be in flight at once, so one visitor cannot burn the model quota or
  starve another.
* **Output** — `grounded` checks that a reply's key fact actually appears in the article it
  cites. A reply with no citation, or whose fact is not in the cited article, is never sent;
  the server downgrades it to an escalation with a stated reason.
"""

from __future__ import annotations

import re
import threading
import time

MAX_MESSAGE_CHARS = 2000
MIN_MESSAGE_CHARS = 2
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"[ \t]+")
_TOKEN = re.compile(r"[a-z0-9$%.]+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be", "it",
         "you", "your", "we", "our", "can", "with", "at", "by", "as", "that", "this", "not", "no",
         "yes", "per", "only", "any", "all", "if", "from", "within", "plan", "plans"}


class MessageError(ValueError):
    """The message cannot be handled; the text is safe to show to the sender."""


def clean_message(raw: object) -> str:
    """Return a normalised message or raise MessageError with a user-facing reason."""
    if not isinstance(raw, str):
        raise MessageError("message must be text")
    text = _CTRL.sub("", raw).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_WS.sub(" ", line).strip() for line in text.split("\n")).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) < MIN_MESSAGE_CHARS:
        raise MessageError("message is empty")
    if len(text) > MAX_MESSAGE_CHARS:
        raise MessageError(f"message is too long ({len(text)} characters; the limit is "
                           f"{MAX_MESSAGE_CHARS})")
    return text


def _tokens(text: str) -> set[str]:
    return {t.strip(".") for t in _TOKEN.findall(text.lower())} - _STOP - {""}


_NUM = re.compile(r"\d")


def grounded(answer: str, article_text: str) -> bool:
    """Is the key fact stated in the cited article?

    Numbers are the facts that matter most in support ("14 days", "$12", "99.9%"), so every
    numeric token in the answer must appear in the article. Otherwise at least one content
    word must. Bare yes/no answers carry no checkable token and are grounded by the article's
    statement, so they pass. Cheap, but it catches a reply that cites article #12 while
    quoting a number from nowhere."""
    a = _tokens(answer)
    if not a:                               # "Yes." / "No." — nothing checkable, allow
        return bool(answer.strip())
    art = _tokens(article_text)
    nums = {t for t in a if _NUM.search(t)}
    if nums:
        return nums <= art
    return bool(a & art)


class RateLimiter:
    """Token bucket per key. `allow(key)` is True if a token was available."""

    def __init__(self, rate_per_min: float, burst: int) -> None:
        self.rate = rate_per_min / 60.0
        self.burst = burst
        self._b: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            tokens, last = self._b.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens < 1:
                self._b[key] = (tokens, now)
                return False
            self._b[key] = (tokens - 1, now)
            if len(self._b) > 10000:            # never grow without bound
                self._b.clear()
            return True


class Gate:
    """At most `limit` agent runs in flight; `acquire()` returns False instead of queueing."""

    def __init__(self, limit: int) -> None:
        self._sem = threading.BoundedSemaphore(limit)

    def acquire(self) -> bool:
        return self._sem.acquire(blocking=False)

    def release(self) -> None:
        self._sem.release()
