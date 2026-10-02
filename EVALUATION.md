# Evaluation

How the agent is evaluated and what the numbers mean. The bar is the one a deployed support system
is held to: not a single accuracy number but **calibration, groundedness, agent behaviour, an
ablation that justifies the design, and uncertainty intervals on all of it**, on the domain the
agent actually runs in.

Reproduce everything:

```bash
export GEMINI_API_KEY=...
python evaluate.py --dataset helpdesk --alpha 0.10   # the product: calibrate on labelled tickets
python evaluate.py --dataset squad    --alpha 0.20   # public benchmark
python -m eval.ablation --dataset helpdesk           # agent loop vs single-shot
python -m eval.behavior --dataset helpdesk           # behaviour, groundedness, bootstrap CIs
python retrieval_eval.py                             # retrieval recall@k (SQuAD)
python -m unittest discover -s tests                 # unit + eval gate + knowledge-base consistency
```

Outputs land in `results/helpdesk/` and `results/squad/` (JSON + SVG), all committed.

## 1. Two datasets, two jobs

| | **helpdesk** (the product) | **squad** (the benchmark) |
|---|---|---|
| Corpus | 30 Northwind help-centre articles, 6 categories (`knowledge_base/`) | 32 SQuAD 2.0 passages |
| Labelled questions | **68 tickets**: 46 answerable (gold answer + article), 22 plausible but **not covered** | **300**: 160 answerable, 140 unanswerable |
| Why | The shipped threshold must be calibrated on the domain it runs in | Public, harder, comparable; shows the method transfers |
| Split | 27 calibrate / **41 held-out** (`Random(13)`) | 120 calibrate / **180 held-out** |
| Error target α | 0.10 | 0.20 |

Correctness = SQuAD exact-match / token-F1 (correct at EM or F1 ≥ 0.5). Model `gemini-2.5-flash`
throughout, so differences are the *method*. Predictions are cached, so analyses are reproducible
without re-querying.

## 2. What we measure, and why

| Dimension | Metric | In support terms |
|---|---|---|
| Safety | wrong replies to uncovered tickets | The failure that costs trust: a confident invented policy |
| Quality | selective accuracy, coverage | Accuracy of replies actually sent; share of tickets auto-resolved |
| Outcome | task accuracy | Right reply **or** correct escalation |
| Guarantee | selective error ≤ α on held-out | "Wrong replies stay under X%" as a verifiable contract |
| Calibration | ECE, reliability diagram | Does confidence 0.8 mean 80% right? The threshold is only as good as the signal under it |
| Groundedness | cited article = gold article | Is the reply actually supported by the article it cites? |
| Behaviour | searches/ticket, reformulation %, escalation source | Is it deciding, and are escalations the agent's judgement or the gate's? |
| Design | agent loop vs single-shot | Does the loop earn its calls, or is "agent" a label? |
| Uncertainty | 95% bootstrap CIs | A point estimate on 41 tickets hides sampling noise |

## 3. Help-centre results (held-out, n = 41)

| | Chatbot (always replies) | Agent, no threshold | **Agent, calibrated** |
|---|---:|---:|---:|
| Wrong replies to uncovered tickets | 100% | 0.0% | **0.0%** `[0.0, 0.0]` |
| Accuracy of replies sent | 65.9% | 96.4% | **96.4%** `[88.0, 100]` |
| Correct outcome (reply or escalate) | 65.9% | 95.1% | 95.1% |
| Tickets auto-resolved | 100% | 68.3% | 68.3% `[53.7, 82.9]` |

**Guarantee:** selective error on held-out tickets **3.6%** `[0.0, 12.5]` against a 10% target — holds
at the point estimate, and the upper CI bound is inside the α + 5pp margin.

**Why the two agent columns match.** Conformal calibration searches for the lowest confidence
threshold whose error on the calibration split is ≤ α. Here the agent's own "not covered" decision
already achieves that (one error in 27 tickets), so τ = **0.00** and the gate adds nothing. That is
the correct outcome of the procedure, not a shortcut: on SQuAD the same procedure sets τ = 1.00
(unanimous samples only). At α = 0.05 the 27-ticket split is too coarse — a single error exceeds 5% —
and conformal falls back to escalate-all; more labelled tickets fix that.

## 4. Calibration — the confidence signal

| Signal | ECE helpdesk | ECE SQuAD |
|---|---:|---:|
| Self-reported (model's own 0–100) | **0.028** | 0.247 |
| Self-consistency, exact-string agreement | 0.156 | 0.190 |
| **Self-consistency, meaning-clustered** (shipped) | **0.053** | — |

Exact-string agreement split votes between paraphrases of one fact ("$12" vs "$12 per user per
month") and made the agent look less sure than it was. Clustering samples by token-F1 ≥ 0.5 (the
same tolerance the correctness metric uses) brought ECE from 0.156 to 0.053. Self-reported
confidence looks well calibrated on this easy set only because almost every attempted reply is
right; on SQuAD it over-claims badly (0.247). Self-consistency is the signal that survives both.
Reliability diagram: `results/helpdesk/reliability.svg`.

## 5. Behaviour and groundedness (`eval/behavior.py`, offline)

| Behaviour (68 tickets) | Value |
|---|---:|
| Mean searches / ticket | 1.10 (max 4) |
| Reformulated (> 1 search) | 5.9% |
| Escalation initiated by the agent's judgement | 30.9% of tickets |
| Gold-article recall: first search → after loop | 97.8% → 97.8% |
| **Citation points to the gold article** (correct replies) | **100%** (38/38) |

Escalations here come from the agent judging "not covered" (one cheap call) rather than from the
confidence gate — the cheapest and most explainable path, and the one that produces the hand-off note.

## 6. Ablation — agent loop vs single-shot (`eval/ablation.py`)

| Metric | Single-shot (`HDA_MAX_STEPS=0`) | Agent loop |
|---|---:|---:|
| Gold-article recall | 97.8% | 97.8% |
| Task accuracy (trust-model) | 95.1% | 95.1% |
| Selective accuracy (calibrated) | 96.3% | 96.4% |
| Wrong replies to uncovered (trust-model) | 0.0% | 0.0% |
| LLM calls / ticket | 5.0 | **4.5** |

Honest reading: on 30 clean articles the first search almost always finds the right one, so the
loop has nothing to recover and the two arms tie on quality. The loop still costs *less* on
average, because judging "not covered" costs one call instead of five reply samples. Its retrieval
value shows on the larger, noisier SQuAD corpus (§7), and would on a real help centre with hundreds
of overlapping articles.

## 7. SQuAD 2.0 benchmark (held-out, n = 180)

| | Naive RAG | + self-check | **+ conformal** |
|---|---:|---:|---:|
| Hallucination on unanswerable | 100% | 16.9% | **9.0%** `[3.4, 15.7]` |
| Selective accuracy | 41.1% | 74.5% | **81.9%** `[72.9, 90.6]` |
| Task accuracy | 41.1% | 81.7% | 77.8% |
| Coverage | 100% | 54.4% | 40.0% `[32.8, 47.2]` |

Guarantee: selective error 18.1% `[9.5, 27.1]` ≤ 20% target (holds at the point estimate; upper CI
above the margin on 300 questions). Ablation on SQuAD: the loop lifts gold recall 98.8% → 99.4%,
cuts hallucination 20.2% → 16.9% (trust-model), costs 4.6 vs 5.0 calls, and — most tellingly — under
the same α the single-shot system could not meet the bound at any coverage (escalate-all) while the
agent met it at 40%. Retrieval ablation (lexical vs hybrid recall@k): `results/squad/retrieval.json`.

## 8. Uncertainty

Every headline metric is reported with a **95% percentile bootstrap CI** (2,000 resamples,
`eval/stats.py`) over the held-out split, and the guarantee is checked both at the point estimate
and at the upper CI bound.

## 9. Tests as an eval gate (`tests/`, offline, CI on every push)

- `test_core.py` — retrieval (incl. RRF), metrics, conformal calibration, self-consistency
  aggregation, the loop's control flow, and **outage handling**: a failed model call is reported as
  `error`, the trace stays honest, `decide()` refuses to reply.
- `test_eval_gate.py` — the whole agent over a golden corpus with a content-aware mock model:
  correct replies cite the correct article, uncovered questions escalate, the loop terminates,
  single-shot does one retrieval.
- `test_knowledge_base.py` — every ticket's gold article exists and textually contains an accepted
  answer; lexical recall@3 ≥ 85%; the default corpus is the help centre. The shipped threshold is
  calibrated on these files, so a broken link here would silently break the guarantee.

## 10. Honest limits

- **Small, clean, fictional corpus.** Northwind makes retrieval easy and the CIs wide; it is a
  faithful shape of a real help centre, not a real one.
- **68 tickets.** Enough to calibrate a 10% target with margin, not a 5% one.
- **Citation ≠ entailment.** We check the cited article is the gold one, not that every clause of
  the reply is entailed by it; a claim↔span verifier is the next step.
- **Self-consistency is not a proof.** A consistently wrong model still passes; the conformal layer
  bounds the risk empirically.
- **Calibration is corpus-specific.** Ingest your own help centre and the threshold is a default
  until you label ~50 of your tickets; the README shows how.
