# Architecture

## The pipeline

```
question
   │
   ▼
[1] Retrieve          retriever.py   hybrid: TF-IDF + embeddings fused with RRF → top-k (numbered)
   │
   ▼
[2] Answer-or-abstain llm.py         the model answers from ONLY those passages (or says it
   │                                 can't), cites the supporting passage, N times concurrently
   ▼
[3] Aggregate         agent.py       self-consistency: confidence = agreement across samples;
   │                                 citation = the passage the agreeing samples pointed to
   ▼
[4] Decide            conformal.py   answer iff answerable AND confidence ≥ τ, where τ is a
   │                                 threshold calibrated for a target error on held-out data
   ▼
answer + confidence + citation      OR     abstain ("can't answer this reliably")
```

## Key design decisions

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
| `calibrated_rag/retriever.py` | lexical (TF-IDF) + hybrid (RRF of TF-IDF + dense) retrieval |
| `calibrated_rag/embeddings.py` | Gemini embeddings, on-disk cache, graceful fallback |
| `calibrated_rag/llm.py` | model-agnostic answer-or-abstain client (swap point for Claude/GPT) |
| `calibrated_rag/agent.py` | retrieve → N concurrent samples → self-consistency + citation |
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
- **Model:** `llm.py` hides the provider behind one function — Claude/GPT is a small change.
- **Faithfulness:** a verifier step (claim ↔ cited-span entailment) would tighten groundedness
  beyond the current citation + self-consistency.
- **Observability:** `trace.py` writes JSONL; point it at OpenTelemetry / a warehouse in prod.
