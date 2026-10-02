# Architecture

## Agent, not pipeline

The control flow is **decided by the LLM at run time**, not hard-coded. Each step the agent looks at
what it has gathered and chooses the next action — search again (with a reformulated query), answer,
or abstain. State (gathered passages, queries tried) carries across steps; the loop is bounded
(`CRAG_MAX_STEPS`, default 3) so it always terminates. When it chooses to answer, the answer goes
through a self-consistency confidence and a calibrated conformal gate.

```
question
   │
   ▼  seed: search(question)
┌── loop (≤ max_steps) ─────────────────────────────────────────────────────────────┐
│  _decide (llm.py)   the LLM reads the gathered passages and returns an action:      │
│      • search  →  tools.py SearchTool(query'): reformulated query, add new passages │
│      • answer  →  break out of the loop and answer                                  │
│      • abstain →  stop; the answer isn't in this corpus                             │
└─────────────────────────────────────────────────────────────────────────────────┘
   │  (answer)
   ▼
[A] Self-consistency  agent.py   answer from ONLY gathered passages, N times concurrently;
   │                             confidence = agreement across samples; cite the support
   ▼
[B] Conformal gate    conformal.py  answer iff answerable AND confidence ≥ τ, where τ is a
   │                                threshold calibrated for a target error on held-out data
   ▼
answer + confidence + citation      OR     abstain ("can't answer this reliably")
```

The agent appends a `steps` trace to every prediction (the searches it ran, the decisions it made),
which `serve.py` and `ask.py` surface so you can see *why* it answered or abstained.

## Key design decisions

0. **An agent that directs its own retrieval.** A fixed retrieve→answer path fails when the first
   query retrieves the wrong passages. Letting the LLM judge sufficiency and **reformulate** recovers
   those cases, and letting it explicitly **abstain** is a first-class action, not just a side effect
   of a low score. The loop is bounded and refuses to repeat a query it already tried, so it can't
   spin. Tools live behind a small interface (`tools.py`), so adding a second corpus or a web lookup
   doesn't touch the loop.

1. **Self-consistency, not self-reported confidence.** A single LLM "I'm 100% sure" is badly
   calibrated. Sampling the answer several times and measuring agreement produces a far better
   signal (see the ECE ablation in the README). Hallucinations tend to disagree across samples;
   genuine answers repeat.

2. **Conformal selective prediction for the abstention threshold.** Rather than hand-pick a cutoff,
   we calibrate it: on a held-out split, choose the lowest confidence threshold whose empirical
   error among answered questions is ≤ α, then verify the guarantee holds on a disjoint test split.
   This turns "abstain sometimes" into "answer with a *tunable risk guarantee*."

3. **Evaluate what matters, not just accuracy.** SQuAD 2.0 (with its unanswerable questions) lets us
   measure hallucination rate, calibration (ECE), and the accuracy/coverage tradeoff — the things a
   production system actually lives or dies on — not just EM/F1.

4. **Zero dependencies.** Retrieval, metrics, charts (SVG), the HTTP server, and the API client are
   all Python standard library. It runs anywhere with `python`, and there's no supply chain to vet.
   The cost is doing a few things by hand; the benefit is portability and auditability.

## Module map

| File | Responsibility |
|---|---|
| `calibrated_rag/agent.py` | the agent: bounded decision loop (search/reformulate/answer/abstain) → self-consistency + citation |
| `calibrated_rag/tools.py` | tools the agent can call (`search` over the corpus); extension point |
| `calibrated_rag/retriever.py` | lexical (TF-IDF) + hybrid (RRF of TF-IDF + dense) retrieval |
| `calibrated_rag/embeddings.py` | Gemini embeddings, on-disk cache, graceful fallback |
| `calibrated_rag/llm.py` | model-agnostic answer-or-abstain + decision client (swap point for Claude/GPT) |
| `calibrated_rag/conformal.py` | split-conformal selective-prediction threshold |
| `calibrated_rag/metrics.py` | SQuAD EM/F1, ECE, reliability bins |
| `calibrated_rag/charts.py` | hand-written SVG reliability + coverage charts |
| `calibrated_rag/corpus.py` | load SQuAD slice *or* an ingested custom corpus |
| `calibrated_rag/trace.py` | append-only JSONL query trace (observability) |
| `evaluate.py` / `ask.py` / `serve.py` / `ingest.py` | eval driver / CLI / web demo / doc indexer |

## Extension points (where production work would go next)

- **Retrieval:** hybrid (TF-IDF + embeddings, fused with RRF) ships; a cross-encoder reranker is the
  next step and is isolated to `retriever.py`. The rest of the pipeline is agnostic to how passages
  are found.
- **Tools:** the agent calls tools through `tools.py`; a web lookup, a calculator, or a second
  corpus would register there and the decision loop could choose them without changing its logic.
- **Model:** `llm.py` hides the provider behind one function — Claude/GPT is a small change.
- **Faithfulness:** a verifier step (claim ↔ cited-span entailment) would tighten groundedness
  beyond the current citation + self-consistency.
- **Observability:** `trace.py` writes JSONL; point it at OpenTelemetry / a warehouse in prod.
