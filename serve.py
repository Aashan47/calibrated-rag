"""The support console: a zero-dependency web app around the agent.

    GEMINI_API_KEY=... python serve.py           # then open http://localhost:8000

Pure Python standard library, no web framework. The UI (helpdesk_agent/ui/index.html) is a
ticket inbox: the agent handles tickets while each step streams live, and every ticket ends
as a cited reply draft, an escalation with a hand-off note, or an explicit model-unavailable.

Endpoints
    GET  /                    the console
    GET  /meta                company, model, threshold (and where it came from), calibration stats
    GET  /articles            every help-centre article the agent can reply from
    GET  /inbox               the ticket queue with statuses and outcomes
    POST /inbox               {"message","customer","subject"} -> new ticket
    GET  /inbox/<id>/stream   handle a ticket; server-sent events, one per agent step, then the result
    POST /inbox/<id>/send     mark an auto-resolved reply as sent
    POST /inbox/<id>/escalate send a ticket to the human queue regardless of the agent
    POST /inbox/<id>/reopen   back to new
    GET  /session             counts by status for this session
    POST /ticket              {"message"} -> one-shot result (non-streaming API)
    POST /compare             {"message"} -> what a generic chatbot (same model, no help centre) replies
    GET  /health

Guardrails (see helpdesk_agent/guards.py): messages are normalised and size-capped; each
client IP is rate-limited; at most HDA_CONCURRENCY agent runs are in flight; a reply is never
sent without a citation, or when its key fact is not in the cited article; a ticket cannot be
handled twice at once; a model outage is reported as such, never as an escalation.
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from helpdesk_agent import agent as agent_mod
from helpdesk_agent import corpus, guards, llm, trace
from helpdesk_agent.inbox import Inbox, TransitionError

HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(HERE, "helpdesk_agent", "ui", "index.html")
RESULTS = os.path.join(HERE, "results", "helpdesk", "results.json")
REPO = "https://github.com/Aashan47/helpdesk-agent"
MAX_BODY = 64 * 1024

print("Loading help centre + agent...")
_C = corpus.load_corpus()
_CONTEXTS, _ARTICLES, _CUSTOM = _C["contexts"], _C["articles"], _C["is_custom"]
_AGENT = agent_mod.Agent(_CONTEXTS)
_TAU = corpus.abstention_threshold(_CUSTOM)
_ART = [{"id": i, **a, "words": len(_CONTEXTS[i].split()), "text": _CONTEXTS[i]}
        for i, a in enumerate(_ARTICLES)]
_INBOX = Inbox(seed_path=None if _CUSTOM else Inbox.__init__.__defaults__[0])
_LIMIT = guards.RateLimiter(rate_per_min=float(os.environ.get("HDA_RATE_PER_MIN", 30)), burst=15)
_GATE = guards.Gate(int(os.environ.get("HDA_CONCURRENCY", 3)))
print(f"Ready: {len(_CONTEXTS)} articles from {_C['name']}; escalation threshold = {_TAU:.2f}; "
      f"retrieval = {_AGENT.retrieval_mode}; samples = {_AGENT.n}")


# ---- meta -------------------------------------------------------------------------------
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
        with open(RESULTS) as f:
            r = json.load(f)
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
            "commit": os.environ.get("RENDER_GIT_COMMIT", "")[:7] or None,
            "limits": {"max_message_chars": guards.MAX_MESSAGE_CHARS}}


# ---- ticket handling ------------------------------------------------------------------
_TITLES = {"ok": "Resolved from the help centre",
           "not_in_documents": "Not covered by the help centre",
           "no_supported_answer": "No article supports a reply",
           "low_confidence": "Confidence below the calibrated threshold",
           "no_citation": "Reply had no source article",
           "not_grounded": "Reply not grounded in the cited article",
           "model_unavailable": "Model unavailable"}


def _explain(pred: dict, d: dict) -> tuple[str, str, str]:
    """(status, reason_code, reason). Resolved, escalated for a stated reason, or model
    unavailable — never blurred. Two output guards run after the calibrated decision: a reply
    without a citation, or whose key fact is not in the cited article, is escalated."""
    n, votes = _AGENT.n, pred.get("votes", 0)
    searches = sum(1 for s in pred.get("steps", []) if s.get("action") == "search")
    if pred.get("error"):
        return "error", "model_unavailable", pred["error"]
    if d["answered"]:
        cite = d.get("citation")
        if not cite:
            return ("escalated", "no_citation",
                    "the samples agreed on an answer but did not point to an article, so it "
                    "cannot be sent.")
        if not guards.grounded(d.get("answer", ""), cite["text"]):
            return ("escalated", "not_grounded",
                    "the answer's key fact does not appear in the article it cites, so it "
                    "cannot be sent.")
        return ("resolved", "ok",
                f"{votes} of {n} independent samples found the answer in the articles and "
                f"{pred['confidence']:.2f} of them agreed on it, above the {_TAU:.2f} threshold.")
    if any(s.get("action") == "abstain" for s in pred.get("steps", [])):
        return ("escalated", "not_in_documents",
                f"after {searches} search{'es' if searches != 1 else ''} the agent judged the "
                f"help centre does not cover this, and stopped before drafting a reply.")
    if not pred.get("retrieved"):
        return ("escalated", "not_in_documents",
                "no article in the help centre matched the ticket at all.")
    if not pred.get("answerable"):
        return ("escalated", "no_supported_answer",
                f"articles were retrieved, but only {votes} of {n} samples found a reply "
                f"supported by them (a majority is required).")
    return ("escalated", "low_confidence",
            f"{votes} of {n} samples drafted a reply, but only {pred['confidence']:.2f} agreed on "
            f"the same answer, below the calibrated threshold of {_TAU:.2f}.")


def _handoff(question: str, pred: dict, code: str, reason: str) -> dict:
    """Deterministic hand-off note for the human — from the trace, no model call, so it is
    always available and never invents."""
    searches = [s["query"] for s in pred.get("steps", []) if s.get("action") == "search"]
    closest = [{"id": i, "title": _ART[i]["title"], "category": _ART[i]["category_name"]}
               for i in pred.get("retrieved", [])[:3]]
    if code == "not_in_documents":
        action = ("Answer manually. If this comes up again, add an article: the agent will then "
                  "resolve it automatically.")
    elif code == "no_supported_answer":
        action = ("Check the closest articles; the answer may be implied but not stated. If so, "
                  "make the article explicit.")
    elif code in ("no_citation", "not_grounded"):
        action = ("The agent drafted a reply it could not tie to an article. Treat the draft as "
                  "unverified; answer from the articles yourself.")
    else:
        action = ("The articles partly cover this but the samples disagreed on the exact answer. "
                  "Read the closest articles and reply; consider clarifying the article.")
    return {"customer_message": question, "searched": searches, "closest": closest,
            "reason_title": _TITLES.get(code, code), "reason": reason, "suggested_action": action,
            "holding_reply": ("Thanks for getting in touch. I've passed this to a teammate who can "
                              "answer it properly, and you'll hear back from us shortly.")}


def _handle(question: str, on_event=None) -> dict:
    t0 = time.time()
    pred = _AGENT.predict(question, on_event=on_event)
    d = agent_mod.decide(pred, _TAU)
    status, code, reason = _explain(pred, d)
    sent_ok = status == "resolved"
    latency = round((time.time() - t0) * 1000)
    trace.log({"q": question[:200], "status": status, "code": code,
               "confidence": round(d["confidence"], 2), "llm_calls": pred.get("llm_calls", 0),
               "searches": sum(1 for s in pred.get("steps", []) if s.get("action") == "search"),
               "latency_ms": latency})
    cite = d.get("citation") if sent_ok else None
    out = {"status": status, "question": question,
           "reply": (d.get("reply") or d.get("answer", "")) if sent_ok else "",
           "answer": d.get("answer", "") if sent_ok else "",
           "confidence": round(d["confidence"], 3), "threshold": round(_TAU, 3),
           "votes": pred.get("votes", 0),
           "votes_total": _AGENT.n if pred.get("retrieved") else 0,
           "reason_code": code, "reason_title": _TITLES.get(code, code), "reason": reason,
           "citation": None, "retrieved": pred.get("retrieved", []),
           "steps": list(pred.get("steps", [])),
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
class HttpError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


class Handler(BaseHTTPRequestHandler):
    server_version = "helpdesk-agent"
    sys_version = ""

    # -- plumbing ----------------------------------------------------------------------
    def _headers(self, code: int, ctype: str, length: int | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                         "style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; "
                         "frame-ancestors 'none'")
        self.end_headers()

    def _send(self, code: int, body, ctype: str = "application/json") -> None:
        data = (body if isinstance(body, (bytes, str)) else json.dumps(body))
        data = data.encode() if isinstance(data, str) else data
        self._headers(code, ctype, len(data))
        self.wfile.write(data)

    def _client(self) -> str:
        fwd = self.headers.get("X-Forwarded-For", "")
        return (fwd.split(",")[0].strip() if fwd else self.client_address[0]) or "?"

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            raise HttpError(400, "bad Content-Length") from None
        if n > MAX_BODY:
            raise HttpError(413, "request body too large")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            obj = json.loads(raw or b"{}")
        except (ValueError, UnicodeDecodeError):
            raise HttpError(400, "body must be JSON") from None
        if not isinstance(obj, dict):
            raise HttpError(400, "body must be a JSON object")
        return obj

    def _ticket_id(self, part: str) -> int:
        if not part.isdigit() or len(part) > 9:
            raise HttpError(404, "no such ticket")
        return int(part)

    def _limited(self) -> None:
        if not _LIMIT.allow(self._client()):
            raise HttpError(429, "too many requests from this client; try again in a minute")

    def log_message(self, *a):  # quiet
        pass

    # -- streaming --------------------------------------------------------------------
    def _stream(self, tid: int) -> None:
        """Server-sent events: one event per agent step as it happens, then the result."""
        self._limited()
        try:
            t = _INBOX.get(tid)
        except KeyError:
            raise HttpError(404, "no such ticket") from None
        try:
            _INBOX.set_status(tid, "handling")
        except TransitionError as exc:
            raise HttpError(409, str(exc)) from None
        if not _GATE.acquire():
            _INBOX.set_status(tid, "new")
            raise HttpError(429, "the agent is busy with other tickets; try again shortly")

        self._headers(200, "text/event-stream; charset=utf-8")

        def send(kind: str, payload: dict) -> None:
            try:
                self.wfile.write(f"event: {kind}\ndata: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass            # client left; keep going so the ticket still gets its result

        try:
            send("ticket", _INBOX.get(tid))
            try:
                result = _handle(t["message"], on_event=lambda step: send("step", step))
            except Exception as exc:  # noqa: BLE001
                _INBOX.set_status(tid, "error")
                send("done", {"error": f"internal error: {exc.__class__.__name__}"})
                return
            status = {"resolved": "auto-resolved", "escalated": "escalated"}.get(result["status"], "error")
            _INBOX.set_status(tid, status, result=result)
            send("done", {"result": result, "ticket": _INBOX.get(tid), "session": _INBOX.counts()})
        finally:
            _GATE.release()

    # -- routes -----------------------------------------------------------------------
    def do_GET(self):  # noqa: N802
        try:
            path = self.path.split("?", 1)[0]
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "inbox" and parts[2] == "stream":
                self._stream(self._ticket_id(parts[1]))
            elif path == "/" or path == "/index.html":
                with open(UI, encoding="utf-8") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif path == "/meta":
                self._send(200, _meta())
            elif path == "/articles":
                self._send(200, {"n": len(_ART), "name": _C["name"], "articles": _ART})
            elif path == "/inbox":
                self._send(200, {"tickets": _INBOX.list(), "session": _INBOX.counts()})
            elif path == "/session":
                self._send(200, _INBOX.counts())
            elif path == "/health":
                self._send(200, {"ok": True, "articles": len(_CONTEXTS),
                                 "commit": os.environ.get("RENDER_GIT_COMMIT", "")[:7] or None})
            else:
                raise HttpError(404, "not found")
        except HttpError as e:
            self._send(e.code, {"error": e.message})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"internal error: {exc.__class__.__name__}"})

    def do_POST(self):  # noqa: N802
        try:
            path = self.path.split("?", 1)[0]
            parts = path.strip("/").split("/")
            if parts[0] == "inbox" and len(parts) == 3:
                tid, act = self._ticket_id(parts[1]), parts[2]
                target = {"send": "sent", "escalate": "escalated", "reopen": "new"}.get(act)
                if not target:
                    raise HttpError(404, "not found")
                try:
                    t = _INBOX.set_status(tid, target)
                except KeyError:
                    raise HttpError(404, "no such ticket") from None
                except TransitionError as exc:
                    raise HttpError(409, str(exc)) from None
                self._send(200, {"ticket": t, "session": _INBOX.counts()})
                return
            b = self._body()
            try:
                q = guards.clean_message(b.get("message", b.get("question", "")))
            except guards.MessageError as exc:
                raise HttpError(400, str(exc)) from None
            if path == "/inbox":
                self._limited()
                try:
                    t = _INBOX.add(q, customer=str(b.get("customer", ""))[:60],
                                   subject=str(b.get("subject", ""))[:120])
                except TransitionError as exc:
                    raise HttpError(409, str(exc)) from None
                self._send(200, {"ticket": t, "session": _INBOX.counts()})
            elif path in ("/ticket", "/ask", "/compare"):
                self._limited()
                if not _GATE.acquire():
                    raise HttpError(429, "the agent is busy; try again shortly")
                try:
                    self._send(200, _compare(q) if path == "/compare" else _handle(q))
                finally:
                    _GATE.release()
            else:
                raise HttpError(404, "not found")
        except HttpError as e:
            self._send(e.code, {"error": e.message})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"internal error: {exc.__class__.__name__}"})


def main():
    port = int(os.environ.get("PORT", "8000"))
    print(f"Serving on http://localhost:{port}  (Ctrl+C to stop)")
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    srv.serve_forever()


if __name__ == "__main__":
    main()
