# Architecture

## What it is

A customer-support agent for one help centre. For each incoming ticket it produces one of three
outcomes, and keeps them strictly apart:

| Outcome | What the customer gets | What the team gets |
|---|---|---|
| **Resolved** | A reply draft stating the fact, with the article it came from | The article cited, the confidence, the trace |
| **Escalated** | A holding reply | A hand-off note: what was searched, the closest articles, why it stopped, suggested action |
| **Model unavailable** | Nothing yet | The HTTP status of the failed call. Never disguised as an escalation. |

The decision to reply is governed by a confidence threshold **calibrated on labelled tickets**
for a target error rate, so "wrong replies sent" is a number you choose, not a hope.

## Agent, not pipeline

The control flow is **decided by the LLM at run time**. Each step the agent looks at the articles
gathered so far and chooses: search again (with a reformulated query), draft a reply, or stop
because the help centre does not cover this. State carries across steps; the loop is bounded
(`HDA_MAX_STEPS`, default 3) so it always terminates.

```
ticket
   │
   ▼  seed: search(ticket text)
┌── loop (≤ max_steps) ─────────────────────────────────────────────────────────────┐
│  _decide (llm.py)   the LLM reads the gathered articles and returns an action:      │
│      • search  →  tools.py SearchTool(query'): reformulated query, add new articles │
│      • answer  →  break out of the loop and draft a reply                           │
│      • abstain →  stop; the help centre doesn't cover this  → ESCALATE              │
└─────────────────────────────────────────────────────────────────────────────────┘
   │  (answer)
   ▼
[A] Self-consistency  agent.py   reply from ONLY the gathered articles, N times concurrently;
   │                             confidence = agreement on the key fact; cite the article
   ▼
[B] Conformal gate    conformal.py  reply iff answerable AND confidence ≥ τ, where τ is
   │                                calibrated on labelled tickets for a target error rate
   ▼
RESOLVED (reply + citation + confidence)   or   ESCALATED (hand-off note)
```

Every prediction carries a `steps` trace (searches with the article ids they returned, decisions,
sample votes, failed calls) and `llm_calls`, which the console and CLI surface.

## Key design decisions

0. **Two independent reasons to escalate, plus an outage path.** The agent can judge that the help
   centre does not cover a ticket (a first-class action, cheap: one call), or the confidence can
   fall below the calibrated threshold after drafting. A failed API call is a third, separate
   outcome: `llm.py` raises `LLMError`, samples carry an `error` field, and `decide()` refuses to
   reply on an errored prediction. The one thing a reliability layer must not do is blame the
   question when the model was down.

1. **Self-consistency, not self-reported confidence.** A single "I'm 100% sure" is badly
   calibrated. Sampling the reply N times and measuring agreement on the key fact is a far better
   signal (ECE ablation in EVALUATION.md). The customer-facing wording comes from an agreeing
   sample; agreement is measured on the short fact so paraphrases don't split the vote.

2. **Calibrate on the domain the agent runs in.** The threshold shipped with the Northwind help
   centre is calibrated on 68 labelled Northwind tickets (46 answerable, 22 deliberately not
   covered), on a 40% calibration split, and the guarantee is checked on the disjoint 60%. The
   SQuAD 2.0 run is kept as the public benchmark. An ingested corpus has no labels, so it gets a
   default threshold until you label ~50 of your own tickets (EVALUATION.md explains how).

3. **The hand-off note is deterministic.** It is built from the trace — queries run, closest
   articles, reason code — with no extra model call, so it is always available and cannot invent
   anything. It also tells the team what to *add* to the help centre.

4. **Zero dependencies.** Retrieval, metrics, SVG charts, the HTTP server, the UI and the API
   client are all Python standard library. `python serve.py` is the whole deployment.

## Module map

| File | Responsibility |
|---|---|
| `helpdesk_agent/agent.py` | the agent: bounded decision loop (search / reformulate / reply / escalate) → self-consistency + citation; error propagation |
| `helpdesk_agent/tools.py` | tools the agent can call (`search` over the help centre); extension point |
| `helpdesk_agent/llm.py` | model-agnostic client: decision completion, reply samples, `LLMError` (swap point for Claude/GPT) |
| `helpdesk_agent/retriever.py` | lexical (TF-IDF) + hybrid (RRF of TF-IDF + dense) retrieval |
| `helpdesk_agent/embeddings.py` | Gemini embeddings, on-disk cache, graceful fallback |
| `helpdesk_agent/conformal.py` | split-conformal selective-prediction threshold |
| `helpdesk_agent/corpus.py` | load the help centre (`knowledge_base/`) or an ingested corpus; threshold selection |
| `helpdesk_agent/kb.py` | the editable help centre: validated add/edit/delete, stable article ids, agent rebuilt per change, per-run snapshots, reset/export, optional write-through |
| `helpdesk_agent/datasets.py` | labelled sets: `helpdesk` (tickets.json) and `squad` |
| `helpdesk_agent/metrics.py` | EM/F1, ECE, reliability bins |
| `helpdesk_agent/charts.py` | hand-written SVG charts |
| `helpdesk_agent/trace.py` | append-only JSONL ticket log (observability) |
| `helpdesk_agent/ui/index.html` | the support console (no build step, no CDN) |
| `knowledge_base/` | 30 Northwind help-centre articles in 6 categories + 68 labelled tickets |
| `serve.py` / `ask.py` / `ingest.py` | console + JSON API / CLI / index your own help centre |
| `evaluate.py`, `eval/` | calibration run; ablation (loop vs single-shot), behaviour, bootstrap CIs |
| `tests/` | unit, end-to-end eval gate, knowledge-base consistency |

## Extension points

- **Channels:** `serve.py` exposes `POST /ticket`; an email or chat adapter posts the message and
  routes `resolved` to the customer and `escalated` (with the hand-off) to the queue.
- **Human queue:** `GET /queue` is every `escalated` ticket with its hand-off note; `close` takes it
  out once a person has answered, `reopen` re-runs it (typically after the help centre was fixed).
- **Help centre:** `kb.py` keeps the articles and rebuilds the agent on every change. Each run takes
  a snapshot, so an edit mid-run cannot shift a citation. Edits are in memory per process (the demo
  host has an ephemeral disk); `HDA_KB_WRITE=1` mirrors them to `knowledge_base/` as markdown, and a
  real deployment would put the same interface over a database.
- **Tools:** a second corpus, an order-status lookup or an account API would register in
  `tools.py`; the loop can choose them without changes to its logic.
- **Model:** `llm.py` hides the provider behind two functions.
- **Faithfulness:** a claim↔cited-span entailment check would tighten groundedness beyond
  citation + self-consistency.
- **Observability:** `trace.py` writes JSONL; point it at OpenTelemetry or a warehouse in prod.
