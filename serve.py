"""The support console: a zero-dependency web app around the agent.

    GEMINI_API_KEY=... python serve.py           # then open http://localhost:8000

Pure Python standard library, no web framework. The UI (helpdesk_agent/ui/index.html) is a
support console: the help centre the agent replies from, an incoming-ticket box, and for each
ticket either a cited reply draft or an escalation with a hand-off note for the human agent,
plus the confidence against the calibrated threshold and the agent's full trace.

Endpoints
    GET  /           the console
    GET  /meta       company, model, threshold (and where it came from), calibration stats
    GET  /articles   every help-centre article the agent can reply from
    POST /ticket     {"message"} -> resolved / escalated / error, with citation, trace, hand-off
    POST /compare    {"message"} -> what a generic chatbot (same model, no help centre) replies
    GET  /health
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from helpdesk_agent import agent as agent_mod
from helpdesk_agent import corpus, llm, trace

HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(HERE, "helpdesk_agent", "ui", "index.html")
RESULTS = os.path.join(HERE, "results", "helpdesk", "results.json")
REPO = "https://github.com/Aashan47/helpdesk-agent"

print("Loading help centre + agent...")
_C = corpus.load_corpus()
_CONTEXTS, _ARTICLES, _CUSTOM = _C["contexts"], _C["articles"], _C["is_custom"]
_AGENT = agent_mod.Agent(_CONTEXTS)
_TAU = corpus.abstention_threshold(_CUSTOM)
print(f"Ready: {len(_CONTEXTS)} articles from {_C['name']}; escalation threshold = {_TAU:.2f}; "
      f"retrieval = {_AGENT.retrieval_mode}; samples = {_AGENT.n}")

_ART = [{"id": i, **a, "words": len(_CONTEXTS[i].split()), "text": _CONTEXTS[i]}
        for i, a in enumerate(_ARTICLES)]

# Example tickets for the committed Northwind help centre (verifiably in / not in it).
_EXAMPLES = None if _CUSTOM else {
    "resolvable": [
        "I bought the annual plan 10 days ago and changed my mind. Can I get my money back?",
        "Can we pay by bank transfer instead of card?",
        "I deleted a project by accident last week. Is it gone for good?",
        "Does the Team plan include phone support?",
    ],
    "escalate": [
        "Do you offer a self-hosted / on-premise version?",
        "Can I be invoiced in euros?",
        "We're an early-stage startup, is there a discount for us?",
    ],
    "tricky": [
        "I'm on the Free plan and want Slack notifications in three channels. Possible?",
        "Our workspace is in the US. Can you move it to the EU region for GDPR?",
        "My export link stopped working after about ten days, can you resend it?",
    ],
}


def _threshold_source() -> str:
    if os.environ.get("HDA_THRESHOLD"):
        return "set by HDA_THRESHOLD (deployment override)"
    if _CUSTOM:
        return "default for an uncalibrated corpus (at least 3 of 5 samples agree)"
    return "calibrated on 68 labelled Northwind tickets (split-conformal selective prediction)"


def _calibration() -> dict | None:
    if _CUSTOM or not os.path.exists(RESULTS):
        return None
    try:
        r = json.load(open(RESULTS))
        c = r["calibrated_abstention"]
        return {"alpha": r["alpha_target_error"], "ece": r["ece"],
                "selective_error": r["test_selective_error_at_threshold"],
                "guarantee_held": r["guarantee_held"], "dataset": r["dataset"],
                "n": r["n_questions"], "n_test": r.get("n_test"),
                "n_unanswerable": r["n_unanswerable"],
                "calibrated_threshold": r["calibrated_threshold"],
                "coverage": c["coverage"], "selective_accuracy": c["selective_accuracy"],
                "hallucination": c["hallucination_rate_unanswerable"],
                "task_accuracy": c["task_accuracy"],
                "naive_hallucination": r["naive_always_answer"]["hallucination_rate_unanswerable"],
                "trust_hallucination": r["uncalibrated_trust_model"]["hallucination_rate_unanswerable"]}
    except Exception:  # noqa: BLE001
        return None


def _meta() -> dict:
    cats: dict[str, str] = {}
    for a in _ARTICLES:
        cats.setdefault(a["category"], a["category_name"])
    return {"name": "helpdesk-agent", "repo": REPO, "company": _C["company"],
            "kb_name": _C["name"], "n_articles": len(_CONTEXTS), "categories": cats,
            "is_custom": _CUSTOM,
            "threshold": round(_TAU, 3), "threshold_source": _threshold_source(),
            "model": os.environ.get("HDA_MODEL", "gemini-2.5-flash"),
            "model_key": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMENI_API_KEY")),
            "retriever": _AGENT.retrieval_mode, "n_samples": _AGENT.n,
            "max_steps": _AGENT.max_steps, "calibration": _calibration(),
            "examples": _EXAMPLES}


# ---- ticket handling ------------------------------------------------------------------
_TITLES = {"ok": "Resolved from the help centre",
           "not_in_documents": "Not covered by the help centre",
           "no_supported_answer": "No article supports a reply",
           "low_confidence": "Confidence below the calibrated threshold",
           "model_unavailable": "Model unavailable"}


def _explain(pred: dict, d: dict) -> tuple[str, str, str]:
    """(status, reason_code, reason) — resolved, escalated for a stated reason, or model
    unavailable. The three never blur into each other."""
    n, votes = _AGENT.n, pred.get("votes", 0)
    searches = sum(1 for s in pred.get("steps", []) if s.get("action") == "search")
    if pred.get("error"):
        return "error", "model_unavailable", pred["error"]
    if d["answered"]:
        return ("resolved", "ok",
                f"{votes} of {n} independent samples found the answer in the articles and "
                f"{pred['confidence']:.2f} of them agreed on it, above the {_TAU:.2f} threshold.")
    if any(s.get("action") == "abstain" for s in pred.get("steps", [])):
        return ("escalated", "not_in_documents",
                f"after {searches} search{'es' if searches != 1 else ''} the agent judged the "
                f"help centre does not cover this, and stopped before drafting a reply.")
    if not pred.get("answerable"):
        return ("escalated", "no_supported_answer",
                f"articles were retrieved, but only {votes} of {n} samples found a reply "
                f"supported by them (a majority is required).")
    return ("escalated", "low_confidence",
            f"{votes} of {n} samples drafted a reply, but only {pred['confidence']:.2f} agreed on "
            f"the same answer, below the calibrated threshold of {_TAU:.2f}.")


def _handoff(question: str, pred: dict, code: str, reason: str) -> dict:
    """A deterministic hand-off note for the human agent — built from the trace, no extra
    model call, so it is always available and never invents anything."""
    searches = [s["query"] for s in pred.get("steps", []) if s.get("action") == "search"]
    closest = [{"id": i, "title": _ART[i]["title"], "category": _ART[i]["category_name"]}
               for i in pred.get("retrieved", [])[:3]]
    if code == "not_in_documents":
        action = ("Answer manually. If this comes up again, add an article: the agent will then "
                  "resolve it automatically.")
    elif code == "no_supported_answer":
        action = ("Check the closest articles below; the answer may be implied but not stated. "
                  "If so, make the article explicit.")
    else:
        action = ("The articles partly cover this but the samples disagreed on the exact answer. "
                  "Read the closest articles and reply; consider clarifying the article.")
    return {"customer_message": question, "searched": searches, "closest": closest,
            "reason_title": _TITLES.get(code, code), "reason": reason, "suggested_action": action,
            "holding_reply": ("Thanks for getting in touch. I've passed this to a teammate who can "
                              "answer it properly, and you'll hear back from us shortly.")}


def _handle(question: str) -> dict:
    t0 = time.time()
    pred = _AGENT.predict(question)
    d = agent_mod.decide(pred, _TAU)
    status, code, reason = _explain(pred, d)
    steps = list(pred.get("steps", []))
    if pred.get("retrieved") and not any(s.get("action") == "abstain" for s in steps):
        steps.append({"action": "sample", "n": _AGENT.n, "votes": pred.get("votes", 0),
                      "errors": pred.get("errors", 0), "error": pred.get("error")})
    latency = round((time.time() - t0) * 1000)
    trace.log({"q": question[:200], "status": status, "code": code,
               "confidence": round(d["confidence"], 2), "llm_calls": pred.get("llm_calls", 0),
               "searches": sum(1 for s in steps if s.get("action") == "search"),
               "latency_ms": latency})
    cite = d.get("citation")
    out = {"status": status, "question": question,
           "reply": d.get("reply") or d.get("answer", ""), "answer": d.get("answer", ""),
           "confidence": round(d["confidence"], 3), "threshold": round(_TAU, 3),
           "votes": pred.get("votes", 0),
           "votes_total": _AGENT.n if pred.get("retrieved") else 0,
           "reason_code": code, "reason_title": _TITLES.get(code, code), "reason": reason,
           "citation": None, "retrieved": pred.get("retrieved", []), "steps": steps,
           "llm_calls": pred.get("llm_calls", 0), "latency_ms": latency, "handoff": None}
    if cite:
        a = _ART[cite["corpus_id"]]
        out["citation"] = {"id": a["id"], "title": a["title"], "category": a["category_name"],
                           "text": a["text"]}
    if status == "escalated":
        out["handoff"] = _handoff(question, pred, code, reason)
    return out


def _compare(question: str) -> dict:
    """The thing people compare against: the same model, told to be a helpful support bot for
    this company, with no help centre. Shown side by side so the difference is concrete."""
    t0 = time.time()
    try:
        out = llm.complete(
            f"You are the customer-support assistant for {_C['company'] or 'our company'}, a "
            "team-collaboration SaaS product. Reply to the customer's message helpfully in one or "
            f"two sentences.\n\nCUSTOMER: {question}\nREPLY:", max_tokens=160)
        reply = out.strip()
        if reply.upper().startswith("REPLY:"):
            reply = reply[6:].strip()
        return {"reply": reply, "latency_ms": round((time.time() - t0) * 1000)}
    except llm.LLMError as exc:
        return {"error": str(exc), "latency_ms": round((time.time() - t0) * 1000)}


# ---- http -----------------------------------------------------------------------------
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
        elif path == "/articles":
            self._send(200, json.dumps({"n": len(_ART), "name": _C["name"], "articles": _ART}))
        elif path == "/health":
            self._send(200, json.dumps({"ok": True, "articles": len(_CONTEXTS)}))
        else:
            self._send(404, "{}")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_POST(self):  # noqa: N802
        try:
            b = self._body()
            q = str(b.get("message") or b.get("question") or "").strip()[:600]
            if not q:
                self._send(400, json.dumps({"error": "message is required"}))
            elif self.path in ("/ticket", "/ask"):
                self._send(200, json.dumps(_handle(q)))
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
