# calibrated-rag

![CI](https://github.com/Aashan47/calibrated-rag/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)
![license](https://img.shields.io/badge/license-MIT-green)

**A retrieval-QA agent that knows when to abstain — with a conformal-calibrated confidence.**

Most RAG systems answer every question, which means they confidently hallucinate on the ones
they can't actually support. In production that's the whole problem: you can't deploy an agent
you can't trust. `calibrated-rag` adds the missing piece — it **abstains instead of guessing**,
and its decision to answer is governed by a confidence threshold **calibrated with split-conformal
selective prediction**, so *when it answers, it's right at a target rate you set.*

Evaluated on **SQuAD 2.0** (which deliberately includes unanswerable questions), measuring not just
accuracy but **calibration, hallucination rate, and the accuracy/coverage tradeoff.**

- 🧠 **Self-consistency confidence** (agreement across samples), not the LLM's miscalibrated "100%"
- 📉 **Conformal abstention** with a risk target — a reliability guarantee on answered questions
- 🔎 **Citations** — every answer points to the passage that supports it
- 🔀 **Hybrid retrieval** — TF-IDF + Gemini embeddings fused with Reciprocal Rank Fusion (lexical fallback)
- 🖥️ **Web demo + CLI** — ask in the browser and watch it answer or abstain (`serve.py`)
- 📁 **Bring your own docs** — index a folder and query it, not just the benchmark (`ingest.py`)
- 📊 **Measured, not claimed** — selective accuracy, ECE, hallucination reduction, accuracy-vs-coverage
- ✅ **Tested + CI** — offline unit tests run in GitHub Actions
- 🪶 **Zero dependencies** — pure Python stdlib (retrieval, charts, HTTP server, API client)

---

## Results

SQuAD 2.0 dev slice — **300 questions** (140 unanswerable) over 32 passages, model
`gemini-2.5-flash`, target error α = 0.20. Three systems, same underlying model:

| | Naive RAG (answers all) | + model self-check | **+ conformal dial** |
|---|---:|---:|---:|
| **Hallucination rate on unanswerable Qs** | 100% | 18.0% | **6.7%** |
| **Accuracy on answered questions** | 41% | 72.0% | **86.6%** |
| Task accuracy (answer correctly *or* abstain) | 41% | 80.6% | 78.3% |
| Coverage (fraction answered) | 100% | 56% | 37% |

Two things to read here:

1. **Abstention is the whole game.** A vanilla RAG that always answers hallucinates on *every*
   unanswerable question (100%) and lands at 41% task accuracy. Letting the agent say "I can't
   answer this" takes task accuracy to 81%.
2. **The conformal layer adds a tunable *guarantee* on top.** Set a target error α; it calibrates
   the confidence threshold on held-out data so accuracy on answered questions clears it — verified
   on a disjoint test split (**13.4% error ≤ 20% target**). It lifts answered-accuracy from 72% to
   **87%** and cuts the hallucination rate on unanswerable questions by nearly two-thirds (18% →
   **7%**), at a coverage cost. A naive RAG gives you no such dial.

**Calibration ablation — the confidence signal matters.** With the same model, the *raw
self-reported* confidence gives ECE **0.263**; **self-consistency** (agreement across samples) gives
ECE **0.172**. A better-calibrated signal is what makes the abstention threshold trustworthy.

<p>
<img src="results/reliability.svg" width="440" alt="Reliability diagram">
<img src="results/coverage.svg" width="440" alt="Selective accuracy vs coverage">
</p>

Left: how closely the self-consistency confidence tracks real accuracy. Right: the core tradeoff —
answer fewer questions, and the ones you answer get more accurate; the circled point is the
calibrated operating point.

### Retrieval ablation

Retrieval is **hybrid** — TF-IDF fused with Gemini embeddings via Reciprocal Rank Fusion, with
automatic lexical-only fallback when no embedding key is present. Recall of the gold passage on the
same 300 questions:

| recall@k | lexical (TF-IDF) | hybrid |
|---|---:|---:|
| @1 | 93.7% | **96.0%** |
| @3 | 98.3% | **99.3%** |
| @5 | 99.3% | 99.7% |

On this small, clean corpus both saturate by k=5; hybrid's edge (notably at @1) grows on larger,
noisier document sets. `CRAG_RETRIEVER=tfidf` forces lexical-only.

---

## How it works

```
question
   │
   ▼
[1] Retrieve        TF-IDF over the passage corpus → top-k passages
   │
   ▼
[2] Answer-or-abstain   the model answers from ONLY those passages, or says it can't
   │                    (sampled N times, concurrently)
   ▼
[3] Confidence      self-consistency: agreement across the N samples on the top answer
   │
   ▼
[4] Decide          conformal threshold τ (calibrated for a target error on a held-out
   │                 split) decides: answer (conf ≥ τ) or abstain
   ▼
answer + citations-worth of context + confidence     OR     "I can't answer this reliably"
```

**The core idea (step 4).** A single LLM confidence is poorly calibrated — the model says it's
certain even when it's wrong. So confidence here is *self-consistency*: ask several times, measure
agreement. Then, instead of picking a threshold by hand, we **calibrate** it: on a held-out split
we find the lowest confidence threshold whose empirical error among answered questions is ≤ α, and
verify the guarantee holds on a disjoint test split. This is the selective-prediction / conformal
risk-control idea — trade a little coverage for a reliability guarantee on what you answer.

---

## Quickstart

No install needed (Python 3.10+, standard library only). Set a Gemini API key:

```bash
export GEMINI_API_KEY=...        # or GEMENI_API_KEY
```

```bash
# Web demo — ask in the browser, see the answer/abstention, confidence, and citation
python serve.py                          # open http://localhost:8000

# Ask from the CLI
python ask.py "Who was yersinia pestis named for?"    # -> Alexandre Yersin (with citation)
python ask.py "What is the capital of Mars?"          # -> abstains

# Query YOUR OWN documents instead of the demo corpus
python ingest.py path/to/docs --name "My Docs"        # index a folder of .txt/.md
python serve.py                                       # now answers over your docs

# Reproduce the evaluation (downloads SQuAD 2.0, runs the agent, writes results + charts)
python evaluate.py --alpha 0.20

# Measure retrieval quality (lexical vs hybrid recall@k)
python retrieval_eval.py

# Run the tests (offline, no API key needed)
python -m unittest discover -s tests -v
```

Config via env: `CRAG_MODEL` (LLM), `CRAG_EMBED_MODEL` (embeddings), `CRAG_RETRIEVER=tfidf|hybrid`,
`CRAG_THRESHOLD` (abstention cutoff for custom corpora). The LLM provider is isolated to
`calibrated_rag/llm.py` (one function), so moving to Claude or GPT is a small change.

### Deploy

A `render.yaml` blueprint is included for a free one-click deploy on
[Render](https://render.com): **New → Blueprint → pick this repo → paste your `GEMINI_API_KEY`**.
The hosted demo boots in under a second (TF-IDF over a small committed corpus) and answers readily
while still abstaining on unanswerable questions.

---

## Repo layout

```
calibrated_rag/
  llm.py          # model-agnostic answer-or-abstain client (stdlib urllib)
  embeddings.py   # Gemini embeddings with on-disk cache + graceful fallback
  retriever.py    # TF-IDF + hybrid (RRF of lexical + dense) retrieval
  agent.py        # retrieve → N concurrent samples → self-consistency confidence + citation
  conformal.py    # split-conformal selective-prediction threshold
  metrics.py      # SQuAD EM/F1, ECE, reliability bins
  charts.py       # hand-written SVG reliability + coverage charts
  corpus.py       # load the SQuAD slice OR an ingested custom corpus
  data.py         # SQuAD 2.0 loader
  trace.py        # append-only JSONL query trace (observability)
evaluate.py       # full QA evaluation → results/results.json + charts
retrieval_eval.py # retrieval ablation (lexical vs hybrid recall@k)
serve.py          # zero-dep web demo (HTTP API + single-page UI)
ask.py            # interactive CLI
ingest.py         # index your own .txt/.md docs
tests/            # offline unit tests (unittest)
.github/workflows/ci.yml   # runs the tests on every push/PR
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design decisions and extension points.

---

## Design notes & honest limits

- **Retrieval is hybrid** (TF-IDF + Gemini embeddings, fused with Reciprocal Rank Fusion), with an
  automatic lexical-only fallback when no embedding key is available. A cross-encoder reranker would
  be the next step and is isolated to `retriever.py`.
- **Confidence = self-consistency over N samples.** Cheap and far better calibrated than
  self-reported confidence, but a model that is *consistently* wrong can still slip through — the
  conformal layer bounds the risk, it doesn't eliminate it.
- **Conformal guarantee is split-conformal / finite-sample**, reported empirically on a held-out
  test split rather than proven here; with a small slice the coverage estimate has variance.
- **Calibration is corpus-specific.** The conformal threshold is calibrated on SQuAD and is the
  right number *for that benchmark*. On your own documents there's no labeled data to calibrate
  against, so `ingest` mode uses a sensible default (answer when ≥ 3/5 samples agree); override with
  `CRAG_THRESHOLD=0.x`, or supply a labeled Q&A set to get a real guarantee for your corpus.
- **Scope is intentionally tight** (one corpus, read-only QA) so the contribution — *calibrated
  abstention* — is the thing that's done well.

---

## Why I built this

Agents that run real work fail on *trust*, not capability — they hallucinate confidently and nobody
can rely on them. The interesting engineering is making them know when **not** to answer, with a
number behind it. This is a small, honest demonstration of exactly that.

MIT licensed.
