# Evaluation

How this agent is evaluated, and what the numbers mean. The goal is the standard a deployed
system is held to — not a single accuracy number, but **calibration, groundedness, agent
behaviour, an ablation that justifies the design, and uncertainty intervals on all of it.**

All results are reproducible:

```bash
export GEMINI_API_KEY=...
python evaluate.py --alpha 0.20     # QA metrics, conformal calibration, charts  -> results/results.json
python -m eval.ablation --alpha 0.20 # agent loop vs single-shot baseline          -> results/ablation.json
python -m eval.behavior             # behaviour, groundedness, calibration CIs     -> results/behavior.json
python retrieval_eval.py            # retrieval recall@k (lexical vs hybrid)       -> results/retrieval.json
python -m unittest discover -s tests # unit + end-to-end eval-gate tests
```

## 1. Benchmark & protocol

- **Dataset:** SQuAD 2.0 dev slice — **300 questions over 32 passages**, deliberately including
  **140 unanswerable** questions. Unanswerable questions are the point: they are where a RAG system
  either hallucinates or correctly abstains, and most QA benchmarks don't test it.
- **Model:** `gemini-2.5-flash`, fixed across every system compared (so differences are the
  *method*, not the model). Provider isolated to `calibrated_rag/llm.py`.
- **Correctness:** SQuAD Exact-Match / token-F1; an answer counts as correct at EM or F1 ≥ 0.5
  (`metrics.is_correct`).
- **Train/test discipline:** the conformal threshold is **calibrated on a 40% split and all headline
  numbers are reported on the disjoint 60% test split** (`random.Random(13)`), so the reliability
  guarantee is measured out-of-sample, never on the data that set it.
- **Determinism:** fixed seeds for the data slice (7) and the split (13); raw predictions are cached
  (`results/preds.json`) so the analysis is reproducible without re-querying.

## 2. What we measure, and why

| Dimension | Metric | Why it matters in production |
|---|---|---|
| **Task quality** | selective accuracy, task accuracy, coverage | Accuracy *among answered* is the number a user feels; coverage is what you give up for it. |
| **Safety** | hallucination rate on unanswerable Qs | The failure that destroys trust: a confident answer to a question with no answer. |
| **Calibration** | ECE, reliability diagram | Does "confidence 0.8" mean 80% right? A threshold is only as good as the signal under it. |
| **Guarantee** | selective error ≤ α on held-out test | Turns "abstains sometimes" into a *tunable risk contract* you can set and verify. |
| **Groundedness** | citation-is-gold-passage rate | When it answers, is the answer actually supported by the passage it points to? |
| **Agent behaviour** | searches/query, reformulation %, abstain source | Is it really deciding and acting, and does reformulation recover misses? |
| **Design justification** | agent loop vs single-shot ablation | Does the loop earn its extra LLM calls, or is "agent" just a label? |
| **Uncertainty** | 95% bootstrap CIs on every headline metric | A point estimate on 300 questions hides sampling noise; report the interval. |

## 3. Three-system comparison (held-out test split)

Same model, three policies, so the delta is the *reliability method*:

1. **Naive RAG** — retrieve, always answer (no abstention).
2. **+ model self-check** — answer only when the model itself says the question is answerable.
3. **+ conformal dial** — also require self-consistency confidence ≥ the calibrated threshold τ.

| Metric (test split, n=180) | Naive RAG | + self-check | **+ conformal** |
|---|---:|---:|---:|
| Hallucination rate on unanswerable Qs | 100% | 16.9% | **9.0%** `[3.4, 15.7]` |
| Selective accuracy (answered Qs) | 41.1% | 74.5% | **81.9%** `[72.9, 90.6]` |
| Task accuracy (answer correctly *or* abstain) | 41.1% | 81.7% | 77.8% |
| Coverage (fraction answered) | 100% | 54.4% | 40.0% `[32.8, 47.2]` |

`[...]` = 95% bootstrap CI. **Conformal guarantee:** target α = 0.20; measured selective error on
the held-out test split = **18.1%** `[9.5, 27.1]` ≤ 20% → the guarantee **holds at the point
estimate**. The upper CI bound (27.1%) sits above the α + 5pp margin, i.e. on 300 questions the
*estimate* of the guarantee still carries real variance — reported rather than hidden (see §7, §9).

**Reading it.** Abstention is what moves task accuracy off the floor (a naive RAG hallucinates on
100% of unanswerable questions). The conformal layer then adds a *tunable guarantee*: set a target
error α, and selective error on the held-out split comes in under it — a dial a naive RAG doesn't have.

## 4. Calibration — the confidence signal

The abstention threshold is only trustworthy if the confidence under it is calibrated. We compare
two signals on the same model:

- **Self-reported** — ask the model for a 0–100 confidence. Badly calibrated (says "100%" when wrong).
- **Self-consistency** — sample the answer N times, use the agreement fraction. Hallucinations tend
  to disagree across samples; genuine answers repeat.

| Confidence signal | ECE (lower is better) |
|---|---:|
| Self-reported (model's own 0–100) | 0.247 |
| **Self-consistency (agreement across N=5 samples)** | **0.190** |

Lower ECE = confidence tracks real accuracy more closely. The reliability diagram
(`results/reliability.svg`) shows this per-bin.

## 5. Agent behaviour & groundedness

Computed offline from the prediction traces (`eval/behavior.py`), so it costs no API calls.

| Behaviour (300 questions) | Value |
|---|---:|
| Mean searches / query | 1.11 (max 4) |
| Search-count distribution | 1→281, 2→10, 3→5, 4→4 |
| Reformulated (ran > 1 search) | 6.3% |
| Agent-initiated abstain (judged not in corpus) | 30.0% |
| **Reformulation recovery** — seed-miss gold passages recovered | 1 of 2 (50%) |
| Gold-passage recall: seed → after loop | 98.8% → 99.4% |
| **Citation present on correct answers** | 100% |
| **Citation points to the gold passage** (groundedness) | 100% (108/108) |

- **Reformulation recovery** is the core agentic win: of the answerable questions whose *first*
  search missed the gold passage, how many did a reformulated query recover? This is retrieval the
  loop fixes that a single shot cannot.
- **Citation groundedness** checks faithfulness: when the agent answers a question correctly, does
  the passage it cites actually contain the gold answer?
- **Abstention source** separates the two independent safety mechanisms — the agent deciding the
  corpus can't support an answer vs. the confidence falling below τ.

## 6. Ablation — does the decision loop earn its cost?

The honest test of "agent vs. renamed pipeline." Same stack, two configurations on the same slice:
**single-shot** (`CRAG_MAX_STEPS=0`: retrieve once, answer) vs. the **agent loop** (judge, reformulate,
re-search, answer/abstain). `eval/ablation.py`.

| Metric | Single-shot (max_steps=0) | **Agent loop** |
|---|---:|---:|
| Gold-passage recall | 98.8% | **99.4%** |
| Task accuracy (trust-model policy) | 80.6% | **81.7%** |
| Hallucination on unanswerable (trust-model) | 20.2% | **16.9%** |
| Selective accuracy at α=0.20 (calibrated) | — (abstain-all) | **81.9%** |
| Coverage at α=0.20 (calibrated) | 0% | **40.0%** |
| LLM calls / query (avg) | 5.0 | **4.6** |

Two honest readings. (1) On this **small, clean** corpus the first retrieval is already strong
(98.8% recall), so the loop's *retrieval* lift is modest and reformulation fires on only 6% of
queries — the gains show up on larger/noisier corpora where the first query misses more often.
(2) Even here, the loop is a net win on the metrics that matter: lower hallucination and higher task
accuracy at the same operating point, **no extra average cost** (early abstention skips the answer
sampling, so 4.6 < 5.0 calls/query), and — most tellingly — under the same α=0.20 target the
single-shot system's confidence could not meet the error bound at any coverage (conformal fell back
to abstain-all, threshold 1.01), whereas the agent met it at 40% coverage. The decision loop makes
the system *reliably answerable* where the single-shot one isn't.

If the loop lifts gold-passage recall and end-to-end task accuracy at a modest extra call budget, the
agentic design is justified; if it didn't, the honest thing would be to drop it. (The abstention /
calibration machinery is identical in both arms, so this isolates the *loop*.)

## 7. Uncertainty

Every headline metric in §3 is reported with a **95% percentile bootstrap CI** (2,000 resamples,
`eval/stats.py`) over the test split. The conformal guarantee is reported both at the point estimate
and as whether the **upper CI bound** stays within the target margin — the stricter, more honest bar.

## 8. Tests as an eval gate

`tests/` runs offline in CI on every push:

- `test_core.py` — unit tests for retrieval (incl. RRF fusion), metrics (EM/F1/ECE), conformal
  calibration, self-consistency aggregation, and the agent loop's control flow (reformulate /
  abstain / answer / no-spin).
- `test_eval_gate.py` — **end-to-end regression**: the whole agent over a golden corpus with a
  content-aware mock model, asserting it answers known questions with a citation to the *correct*
  passage, abstains out-of-corpus, terminates within the step bound, and that single-shot mode does
  exactly one retrieval. These are the behaviours a deploy must not silently regress.

## 9. Honest limits

- **Single corpus, read-only QA.** Scope is deliberately tight so the contributions — an agent that
  directs its own retrieval, and calibrated abstention — are done well rather than broadly.
- **Small slice → variance.** 300 questions over 32 passages means the CIs are not tight; they are
  reported precisely so the reader isn't misled by a point estimate.
- **Calibration is corpus-specific.** τ is calibrated on SQuAD and is the right number *for SQuAD*.
  Your own documents have no labels to calibrate on, so `ingest` mode uses a sensible default
  (`CRAG_THRESHOLD`); a labeled Q&A set would give a real guarantee for your corpus.
- **Self-consistency is not a proof.** A model that is *consistently* wrong can still pass; the
  conformal layer bounds that risk empirically, it doesn't eliminate it.
- **Citation = retrieval grounding, not entailment.** We check the cited passage is the gold one; a
  claim↔span entailment verifier would be the stronger (and next) faithfulness check.
