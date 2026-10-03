# Energy-MoE Design Requirements

What a Mixture-of-Experts model for energy demand would require, and what Phase 5
measured that bears on it. This document prepares Phase 7; it does not build,
train, or recommend training one.

The central question is not "can an MoE be built" but "**is there a routing problem
worth solving here at all**". Phases 4 and 5 measured the answer's preconditions.

---

## 1. The precondition: measured heterogeneity

An MoE earns its complexity only if different conditions genuinely need different
predictors. Measured at the shortest horizon, kW MAE (Phase 5 TCN vs the Phase 4
baselines, on the identical 179,520-row test split):

| Demand tercile | n | Phase 5 TCN | persistence | GBM |
|---|---|---|---|---|
| low | 59,840 | **0.0772** | 0.0664 | 0.0875 |
| typical | 59,840 | 0.2482 | 0.2494 | 0.2503 |
| high | 59,840 | **0.9280** | 0.8972 | 0.9850 |

| Ramp tercile | Phase 5 TCN | persistence | GBM |
|---|---|---|---|
| low | 0.0324 | **0.0034** | 0.0739 |
| typical | 0.0980 | 0.0603 | **0.1708** |
| high | 1.1231 | 1.1493 | **1.0780** |

Read this carefully, because it cuts both ways:

* **The spread is real and large.** High-demand rows cost 12x low-demand rows.
  Across horizons the high tercile grows from 0.928 kW at 15 minutes to 4.051 kW at
  24 hours.
* **Error is concentrated, not diffuse.** The high demand tercile is a third of the
  rows and **74% of the total error**. The high ramp tercile is **90% of the total
  error**. That is the shape a router could exploit.
* **But no single model wins everywhere.** The TCN wins the demand terciles;
  persistence wins low-ramp periods by an order of magnitude; GBM wins high-ramp
  periods. The best available predictor is currently *regime-dependent*.

That last row is the actual finding. It is weaker than "an MoE will help" and
stronger than "nothing is going on".

---

## 2. The precondition that is NOT met: a deployable gate

**The terciles above are computed from the demand the model is being asked to
predict.** They are diagnostics, not a routing signal. A router cannot use them,
because at forecast time the future demand level and future ramp are unknown by
construction. Any design that routes on them has leaked the target, and this
project's leakage guard would (correctly) reject it.

So a deployable gate must be driven by features observable **at or before the
forecast origin**:

| Candidate gate feature | Observable? | Note |
|---|---|---|
| Demand level over the last few hours | yes | the natural proxy for the demand tercile |
| Trailing ramp magnitude | yes | the natural proxy for the ramp tercile |
| Hour of day, day of week | yes | Phase 5's ablation found the cyclic channels were *not* earning their place, so these are weak on their own |
| Predicted level for the target horizon | only if a level model exists first | implies a two-stage design |
| **Future demand level or ramp** | **no** | this is the target; using it is leakage |
| **Future weather** | **no** | not in this dataset at all |

Any Phase 7 proposal must state which of these its gate uses and show that every
gate input is available at the origin. A gate built on unobservable features is
worthless no matter how good it scores on this test split, because the score would
be an artifact.

There is a further cost: building a *usable* regime proxy means labelling regions
by a trailing-window rule and accepting that the proxy is imperfect. Phase 7 must
report how well any proxy separates the terciles above, not assume it does.

---

## 3. Expert inventory: what the evidence already supports

Phase 5's ablations (`artifacts/phase5/ablations.md`) measured which components earn
their place. Two results change what an MoE should contain:

1. **The delta parameterisation is not optional.** Predicting the level instead of the
   change from `y(t)` costs **+253% at 15 min, +52% at 1 h, +20% at 24 h**. Any
   expert that predicts levels is starting from a much worse position. Every expert
   shares the delta target.
2. **Context length is a real axis.** One day of context costs +38% at 15 min versus a
   week; two days still cost +14%. Experts differing in lookback are therefore a
   defensible axis of specialisation — the effect is measured, not assumed.

A third result argues *against* a particular expert class:

3. **Attention did not beat convolution here.** A 112k-parameter patch Transformer was
   49.8% worse at 15 min and 0.6% worse at 24 h than the 74k-parameter TCN, and 6.0%
   better at 1 h, at a matched reduced budget. On this data an attention expert is not
   obviously a useful specialist.

---

## 4. Candidate architectures

| Design | Assessment |
|---|---|
| **Horizon experts** (one per horizon) | Weak. Horizons are already separate output heads sharing a trunk, and the `single_horizon` ablation showed sharing *helps* by 38.5%. Splitting them discards a measured win. |
| **Regime experts** (stable vs ramp, low vs high) | The best-supported option, conditional on §2's gate problem being solved. |
| **Lookback experts** (1 day vs 1 week) | Supported by measurement; increases memory sharply, since each expert needs its own window. |
| **Model-family experts** (TCN vs GBM vs persistence) | Effectively a learned stacking ensemble. Note that "persistence" and "GBM" here are *not* neural experts, which is a design smell: the gate would be selecting among different model classes, which is an ensemble, not an MoE. |
| **Per-customer experts** | Rejected. Error concentration is by *regime*, not by customer size; Phase 4 already showed the worst customer carries roughly 7x the median. Per-customer experts would multiply parameters by 40 for no measured gain. |

---

## 5. Requirements any Phase 7 proposal must meet

1. **A gate over observable features only**, with each feature's availability at the
   forecast origin stated explicitly (§2).
2. **A leakage test**, in the same spirit as the Phase 4 guard: assert that the gate
   cannot see any input row at or after the target timestamp. The existing guard
   checks windows; this one must check the gate's inputs too.
3. **Load-balancing loss**, chosen and justified. Without it an MoE collapses onto one
   expert and becomes a single slow model.
4. **Capacity factors reported per expert.** An MoE whose experts each handle 3% of rows
   has multiplied parameters without buying coverage. If that is what the gate
   produces, the honest conclusion is that one model is enough.
5. **A no-MoE control at matched budget.** The comparison is against the TCN and GBM
   measured on the same split in the same run, or the result means nothing.
6. **Per-regime error reported against the best single model**, not only against the
   average. A router that improves the average by degrading one regime is not a win,
   and the low-ramp regime is where persistence is currently untouchable
   (0.0034 kW against 0.0324 kW). Losing that regime to gain elsewhere is a plausible
   and undesirable outcome.

---

## 6. Compute reality

Phase 5's TCN is 74,099 parameters and took **6,524 s** to train on CPU for 8 epochs.
Its ablation suite took a further 1,339 s. An MoE on the same data would add
per-expert parameters, and each expert with its own lookback multiplies that again.
The realistic CPU budget for this project is one or two such runs, which is enough
to test one routing hypothesis well and not enough to sweep architectures.

---

## 7. What is deliberately NOT claimed

* Not that an MoE would beat the TCN. §1 shows a *reason to test* one, not a result.
* Not that the regime terciles are available at inference. §2 shows they are not.
* Not that the ablation budget supports absolute comparisons. The ablation suite
  trains at 2 epochs and a coarse training stride, so its reference is markedly worse
  than the main run (0.5435 vs 0.4178 kW at 15 min). Comparisons **within**
  `ablations.md` are valid; comparisons against the main run are not.
* Not that attention is exhausted as an option. The Transformer ablation ran at a
  matched reduced budget, and undertrained attention is the usual reason it loses.
* Not that more parameters help. Nothing measured suggests parameter count is the
  bottleneck.

---

## 8. Minimum bar for starting Phase 7

| Criterion | Requirement |
|---|---|
| Gate inputs | Every one listed with its availability at the origin; no future-derived feature |
| Gate quality | The trailing-window proxy's separation of the terciles in §1 measured and reported, before any MoE is trained |
| Leakage test | Automated, covering the gate's inputs, not just the sequence window |
| Control | TCN and GBM on the same split in the same run |
| Budget | Stated in the units of §6 |
| Fallback | "One model is enough" is a recorded, acceptable outcome |
