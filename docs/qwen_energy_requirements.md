# Qwen Energy Model Requirements

What a future **Qwen3-1.7B** energy model will need, derived from what Phase 5
measured. This document prepares Phase 6. It does not download, fine-tune or modify
Qwen, and it makes no claim that such a model will help — only what would have to be
true for it to.

The companion document is `world_model_requirements.md`, which establishes that
SMART-DS cannot support an **action-conditioned** model. Everything below is therefore
about **demand dynamics**, which is what this dataset can actually support.

---

## 1. The bar a foundation model must clear

Measured on the **identical** 179,520-row chronological test split of dataset
`ds-16ff1dabe80b` (Phase 4) / `sq-437c16b5d925` (Phase 5):

| Model | h=1 (15 min) | h=4 (1 h) | h=96 (24 h) |
|---|---|---|---|
| naive last value | 0.4043 | 0.8897 | 2.4343 |
| classical hist GBM | 0.4242 | 0.8579 | **1.5648** |
| Phase 4 neural MLP | 0.4638 | 0.9315 | 6.4850 |
| Phase 4 neural GRU | 0.5691 | 0.8976 | 2.2856 |
| **Phase 5 temporal TCN** | **0.4178** | **0.8307** | 1.6510 |

MAE in kW. Reproduce with `energy-intel temporal run` and `energy-intel ml run`.

A Qwen-based model is only worth building if it beats **1.5648 kW at 24 hours**, not
merely the naive baseline. On this data that is a **36% MAE reduction** over the best
classical result, and persistence is within 3% of the TCN at 15 minutes.

---

## 2. Input representation

| Requirement | Value | Why |
|---|---|---|
| **Historical context** | one week (672 native steps) | the TCN's ablation measured the value of context directly; see §7 |
| **Input form** | ordered tokens, not a flat feature vector | that is the entire Phase 5 hypothesis: order carries information a lag set discards |
| **Subsampling** | every 4th step → 168 tokens | measured: the same week at a quarter of the compute. A foundation model may afford full resolution and should test it |
| **Channels** | demand, first difference, sin/cos hour, sin/cos day-of-year | the Phase 5 ablation measured the cyclic channels' contribution; if negligible, they are decoration and should be dropped |
| **Tokenisation** | **numeric, not text** | the target is a smooth physical quantity. BPE tokenisation of floats is lossy and unjustified. A patch or linear projection is the correct front end |
| **Normalisation** | per unit of the series' own rated kW; scaler fitted on training rows only | a 1.4 kW household and a 388 kW shop must reach the model on comparable footing, and the divisor must be a physical quantity known for the whole horizon |

**What must not happen:** the model must not receive a text rendering of the series and
must not receive any value after the forecast origin. Both are silent, irreversible
leaks.

---

## 3. Forecast horizon and output structure

Three horizons — 15 min, 1 h, 24 h — from one shared representation, each with its own
head. Phase 5's `single_horizon` ablation measured what sharing is worth; that number
is the reference for whether multi-task framing is justified here.

Horizon weights must stay equal. h=96 is not up-weighted to manufacture an improvement
(D-075).

---

## 4. Target representation

Two formulations were compared in Phase 5: predicting the **level** and predicting the
**change from the value at the origin** (`delta`). The result is recorded in
`artifacts/phase5/ablations.md` and is the evidence for which a future model should
pick. A delta formulation is attractive for a decoder-only model because the
persistence shortcut is built into the output parameterisation — persistence is within
3% of the best 15-minute result, so any model that does not start there is starting
behind.

---

## 5. Temporal encoding

The decisive open question, and the reason Phase 5 measured a transformer at all.

* **Convolutional** (Phase 5 primary): translation-equivariant, cheap on CPU, strong
  at 15 min and 1 h.
* **Attention**: the natural choice for a 1.7B decoder, and the architecture a future
  Qwen specialisation would use.

Phase 5's `transformer` ablation answers whether attention *earns its place on this
data*. If a compact patch transformer cannot match a 74k-parameter TCN, then scaling
attention is unlikely to be the missing ingredient, and a Qwen specialisation should be
justified on **transfer and few-shot** grounds rather than on raw forecasting accuracy.
That is a decision for Phase 6, and the evidence is in this repository rather than in
a preference.

---

## 6. Evaluation requirements

1. The chronological test split, never shuffled, never used for selection.
2. **All three horizons reported.** h=96 is where the classical model is strongest and
   where a foundation model would have to prove itself.
3. MAE in kW as the headline, plus RMSE, sMAPE, R², peak-decile error, ramp error and
   the **error distribution** (median, p90, p99). Average MAE alone would hide whether a
   model reduces large failures.
4. **Regime breakdown** using the axes Phase 4 established: demand terciles, ramp
   terciles.
5. **Per-series** breakdown, because error concentrates: the worst customer carries
   roughly 7x the median customer's MAE.
6. **Resource accounting**: parameter count, training duration, and inference cost. A
   1.7B model must beat a 74k-parameter TCN **and** justify its cost.

---

## 7. What Phase 5 measured that should shape the design

1. **Temporal order helps, but not everywhere.** The TCN beat gradient boosting at
   15 min and 1 h, and lost at 24 h. Any foundation model must therefore be evaluated
   per horizon; a single average would hide the one place it must beat GBM.
2. **Persistence is the real short-horizon baseline.** At 15 minutes it is within 3%
   of the TCN. Most of the apparent difficulty there is not learnable from history.
3. **The hard series are high-volatility residential customers**, not large ones, and
   the hard hours are the morning and evening ramps.
4. **Error is heavy-tailed**: p99 is over 100x the median. A model that improves MAE
   slightly while worsening p99 has not helped an operator.
5. **Context length matters and can be measured.** The lookback ablation brackets the
   answer rather than assuming it.

---

## 8. What is deliberately NOT claimed

* Not that Qwen will forecast better. Phase 5's evidence makes that an open question,
  not a premise.
* Not that a larger model helps. Nothing measured so far suggests parameter count is
  the bottleneck: a 74k-parameter TCN already beat gradient boosting at two of three
  horizons, and Phase 4's MLP and GRU lost while being far larger.
* Not that a foundation model is a world model. See `world_model_requirements.md`.
* Not that one seed establishes significance. No confidence intervals are claimed
  anywhere in this repository; re-running with several seeds is a prerequisite for any
  "significantly better" statement.

---

## 9. Minimum bar for starting Phase 6

| Criterion | Requirement |
|---|---|
| Data | A dataset whose evaluation spans **more than one year**, or an explicit statement that annual generalisation is out of scope. One year cannot validate an annual model. |
| Baselines | Phase 4's GBM and Phase 5's TCN re-measured on the Phase 6 split, in the same run |
| Seeds | At least 3 seeds for any comparison claimed as significant |
| Compute | A stated budget; Phase 5's 74k-parameter TCN took 6,524 s on CPU, which is the order-of-magnitude reference |
| Success criterion | Defined **before** training, in the same form as §1 |
| Honest fallback | A recorded negative result is an acceptable Phase 6 outcome and is written up as one |
