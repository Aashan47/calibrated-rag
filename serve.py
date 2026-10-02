"""A zero-dependency web interface for the agent.

    GEMINI_API_KEY=... python serve.py           # then open http://localhost:8000

Pure Python standard library — no web framework. The UI (calibrated_rag/ui/index.html) shows
the knowledge base the agent answers from, the answer or abstention with its reason, the
confidence against the calibrated threshold, the cited passage, and the agent's full trace.

Endpoints
    GET  /          the interface
    GET  /meta      model, retriever, threshold (and where it came from), calibration stats
    GET  /corpus    every passage the agent can answer from (id, title, text, source)
    POST /ask       {"question"} -> answered / abstained / error, with citation + trace
    POST /compare   {"question"} -> what the same model says with no documents at all
"""

from __future__ import annotations

import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from calibrated_rag import agent as agent_mod
from calibrated_rag import corpus, llm, trace

HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(HERE, "calibrated_rag", "ui", "index.html")
RESULTS = os.path.join(HERE, "results", "results.json")
REPO = "https://github.com/Aashan47/calibrated-rag-agent"

print("Loading corpus + agent...")
_C = corpus.load_corpus()
_CONTEXTS, _SOURCE, _CUSTOM, _SOURCES = _C["contexts"], _C["name"], _C["is_custom"], _C["sources"]
_AGENT = agent_mod.Agent(_CONTEXTS)
_TAU = corpus.abstention_threshold(_CUSTOM)
print(f"Ready: {len(_CONTEXTS)} passages from {_SOURCE}; abstention threshold = {_TAU:.2f}; "
      f"retrieval = {_AGENT.retrieval_mode}; samples = {_AGENT.n}")


# ---- corpus presentation -------------------------------------------------------------
def _title(text: str, i: int) -> str:
    """A short label for a passage: its source file if ingested, else its opening clause."""
    if _SOURCES and i < len(_SOURCES) and _SOURCES[i]:
        return _SOURCES[i]
    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    first = re.sub(r"\[[^\]]*\]", "", first).strip()
    return (first[:72].rsplit(" ", 1)[0] + "…") if len(first) > 72 else first


_PASSAGES = [{"id": i, "title": _title(t, i), "text": t, "words": len(t.split()),
              "source": (_SOURCES[i] if _SOURCES and i < len(_SOURCES) else None)}
             for i, t in enumerate(_CONTEXTS)]

# Example questions for the committed demo corpus (answers verifiably in / not in it).
_EXAMPLES = None if _CUSTOM else {
    "answerable": ["Who was Yersinia pestis named for?",
                   "What is the tallest building in Jacksonville?",
                   "What is Warsaw's symbol?",
                   "When did Sky announce Sky Q?"],
    "unanswerable": ["What year did Alexandre Yersin die?",
                     "What is the capital of Mars?",
                     "Who won the 2050 World Cup?"],
}


def _threshold_source() -> str:
    if os.environ.get("CRAG_THRESHOLD"):
        return "set by CRAG_THRESHOLD (deployment override)"
    if _CUSTOM:
        return "default for an uncalibrated custom corpus (≥ 3 of 5 samples agree)"
    return "calibrated on SQuAD 2.0 with split-conformal selective prediction"


def _calibration() -> dict | None:
    if _CUSTOM or not os.path.exists(RESULTS):
        return None
    try:
        r = json.load(open(RESULTS))
        return {"alpha": r["alpha_target_error"], "ece": r["ece"],
                "selective_error": r["test_selective_error_at_threshold"],
                "guarantee_held": r["guarantee_held"], "dataset": r["dataset"],
                "n": r["n_questions"], "calibrated_threshold": r["calibrated_threshold"]}
    except Exception:  # noqa: BLE001
        return None


def _meta() -> dict:
    return {"name": "calibrated-rag-agent", "repo": REPO,
            "n": len(_CONTEXTS), "source": _SOURCE, "is_custom": _CUSTOM,
            "threshold": round(_TAU, 3), "threshold_source": _threshold_source(),
            "model": os.environ.get("CRAG_MODEL", "gemini-2.5-flash"),
            "model_key": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMENI_API_KEY")),
            "retriever": _AGENT.retrieval_mode, "n_samples": _AGENT.n,
            "max_steps": _AGENT.max_steps, "calibration": _calibration(),
            "examples": _EXAMPLES}


# ---- answering -----------------------------------------------------------------------
def _explain(pred: dict, d: dict) -> tuple[str, str, str]:
    """(status, reason_code, reason). Keeps the three outcomes — answered, abstained for a
    stated reason, model unavailable — strictly apart."""
    n, votes = _AGENT.n, pred.get("votes", 0)
    searches = sum(1 for s in pred.get("steps", []) if s.get("action") == "search")
    if pred.get("error"):
        return "error", "model_unavailable", pred["error"]
    if d["answered"]:
        return ("answered", "ok",
                f"{votes} of {n} independent samples found an answer in the passages and "
                f"{pred['confidence']:.2f} of them agreed on this one, above the {_TAU:.2f} threshold.")
    if any(s.get("action") == "abstain" for s in pred.get("steps", [])):
        return ("abstained", "not_in_documents",
                f"after {searches} search{'es' if searches != 1 else ''} the agent judged the answer "
                f"is not in this knowledge base and stopped before attempting one.")
    if not pred.get("answerable"):
        return ("abstained", "no_supported_answer",
                f"the passages were retrieved, but only {votes} of {n} samples found an answer "
                f"supported by them (a majority is required).")
    return ("abstained", "low_confidence",
            f"{votes} of {n} samples answered, but only {pred['confidence']:.2f} agreed on the same "
            f"answer, below the calibrated threshold of {_TAU:.2f}.")


_TITLES = {"ok": "Answered", "not_in_documents": "Not in these documents",
           "no_supported_answer": "No supported answer", "low_confidence": "Confidence too low",
           "model_unavailable": "Model unavailable"}


def _answer(question: str) -> dict:
    t0 = time.time()
    pred = _AGENT.predict(question)
    d = agent_mod.decide(pred, _TAU)
    status, code, reason = _explain(pred, d)
    steps = list(pred.get("steps", []))
    if pred.get("votes") is not None and not any(s.get("action") == "abstain" for s in steps) \
            and pred.get("retrieved"):
        steps.append({"action": "sample", "n": _AGENT.n, "votes": pred.get("votes", 0),
                      "errors": pred.get("errors", 0), "error": pred.get("error")})
    latency = round((time.time() - t0) * 1000)
    trace.log({"q": question[:200], "status": status, "code": code,
               "confidence": round(d["confidence"], 2), "llm_calls": pred.get("llm_calls", 0),
               "searches": sum(1 for s in steps if s.get("action") == "search"),
               "latency_ms": latency})
    cite = d.get("citation")
    return {"status": status, "question": question, "answer": d["answer"],
            "confidence": round(d["confidence"], 3), "threshold": round(_TAU, 3),
            "votes": pred.get("votes", 0), "votes_total": _AGENT.n if pred.get("retrieved") else 0,
            "reason_code": code, "reason_title": _TITLES.get(code, code), "reason": reason,
            "citation": ({"id": cite["corpus_id"], "title": _PASSAGES[cite["corpus_id"]]["title"],
                          "text": cite["text"]} if cite else None),
            "retrieved": pred.get("retrieved", []), "steps": steps,
            "llm_calls": pred.get("llm_calls", 0), "latency_ms": latency}


def _compare(question: str) -> dict:
    """The same model with no documents and no abstention policy — the thing people compare
    against. Shown side by side so the difference is concrete rather than claimed."""
    t0 = time.time()
    try:
        out = llm.complete("Answer the question concisely, in one or two sentences.\n\n"
                           f"QUESTION: {question}\nANSWER:", max_tokens=160)
        return {"answer": out.strip(), "latency_ms": round((time.time() - t0) * 1000)}
    except llm.LLMError as exc:
        return {"error": str(exc), "latency_ms": round((time.time() - t0) * 1000)}


# ---- http ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/" or path.startswith("/index"):
            with open(UI, encoding="utf-8") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/meta":
            self._send(200, json.dumps(_meta()))
        elif path == "/corpus":
            self._send(200, json.dumps({"n": len(_PASSAGES), "source": _SOURCE,
                                        "passages": _PASSAGES}))
        elif path == "/health":
            self._send(200, json.dumps({"ok": True, "passages": len(_CONTEXTS)}))
        else:
            self._send(404, "{}")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_POST(self):  # noqa: N802
        try:
            q = str(self._body().get("question", "")).strip()[:500]
            if not q:
                self._send(400, json.dumps({"error": "question is required"}))
            elif self.path == "/ask":
                self._send(200, json.dumps(_answer(q)))
            elif self.path == "/compare":
                self._send(200, json.dumps(_compare(q)))
            else:
                self._send(404, "{}")
        except Exception as exc:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(exc)}))

    def log_message(self, *a):  # quiet
        pass


def main():
    port = int(os.environ.get("PORT", "8000"))
    print(f"Serving on http://localhost:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
