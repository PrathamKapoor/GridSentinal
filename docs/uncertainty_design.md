# Phase 8 design: calibrated predictive uncertainty

**Status:** complete. Measured on the sealed test split, 179,520 rows.
**Verdict:** **YES** on all three components, with one negative finding that matters more
than the positive one.

---

## 1. What this phase asks, and what it deliberately does not

Phase 7 left the point forecast where it should be: a fixed weighted ensemble of
persistence, a gradient-boosted tree and a temporal convolution, at 0.3983 / 0.7917 /
1.5399 kW MAE for h=1 / 4 / 96. Phase 8 changes none of that. The forecast is read back
from Phase 7's own artifact, checked against the copy in `configs/uncertainty.toml`, and the
run **stops** if the two disagree.

What the phase asks instead is two questions that a point forecast cannot answer:

1. **How uncertain is it?** - a prediction interval at a stated coverage level.
2. **Does saying so identify the forecasts that will be wrong?** - is the interval wide on
   the rows that turn out to be badly forecast?

The second is the question that matters. An interval that is always the right width and
never varies is a number, not an uncertainty estimate.

### Refused by scope, and why

| Not done | Why |
|---|---|
| Aleatoric / epistemic decomposition | Separating the two needs assumptions about the error-generating process that this data cannot support. Phase 6 already measured that the pretrained representation carries almost no usable structure. The vocabulary here is *predictive uncertainty*, *model disagreement* and *data variability*. |
| Retraining or re-weighting the point forecast | Then the phase would be measuring a different forecast, and the intervals would not describe the system that exists. |
| A time-ordered conformal method | Would give a real coverage guarantee, at the cost of a method family this phase does not have time to validate. Recorded as the direction, not built. |
| Injecting data corruption to test quality conditioning | Would answer a question about the injected corruption. SMART-DS carries no per-timestep quality flags at all - see §7. |
| Any post-hoc gate, regime policy or blend fitted on test | Each would be fitted on the split the phase reports. |

---

## 2. The partition, and the one thing it does not fix

```text
TRAIN          Phase 4/5 split. 950,400 rows. The experts were fitted here.
                      |
VALIDATION     Phase 5/7 split. 179,520 rows. Expert predictions are out-of-sample.
    +-- CAL_FIT      first 50%, 89,760 rows. Scale functions, quantile models.
    +-- CAL_CONF     last  50%, 89,760 rows. Conformity scores, band cut points.
                      |
TEST           Phase 5/7 split. 179,520 rows. Read once, at the end.
```

**Why validation is halved rather than reusing Phase 7's own 70/30 router split.** Phase 7
fitted the fixed ensemble's three weights per horizon on the first 70% of validation. A scale
model fitted on those same rows would inherit that fit. Phase 7's cross-fitting check
(§14 of its design doc) measured that in-sample expert behaviour changes the achievable
headroom by up to 2x, so "three parameters, probably negligible" was not a number worth
relying on.

**The overlap that remains, measured rather than assumed.** The conformity half and the
fitting half are not both free of Phase 7's weight fit, and `split_parity_check` reports the
point forecast's own error on all three row sets:

| horizon | CAL_FIT MAE | CAL_CONF MAE | test MAE | fit vs conform |
|---|---|---|---|---|
| h=1 | 0.4510 | 0.4154 | 0.3983 | **+8.576%** |
| h=4 | 0.8000 | 0.8058 | 0.7917 | -0.721% |
| h=96 | 1.3816 | 1.5528 | 1.5399 | **-11.025%** |

This is the single most important number in the phase, because it predicts the headline
negative result. The two halves of validation are not equally hard: at h=1 the fitting
half is 8.6% worse than the conformity half, and at h=96 it is 11.0% *better*. Anything
calibrated on CAL_CONF and evaluated on test is therefore being asked to transfer across a
difficulty shift of that size, and the table in §5 shows exactly that happening.

---

## 3. The methods, as one idea

Every symmetric method here is

```text
interval = point ± multiplier × scale(row)
```

so the families differ only in how `multiplier` is obtained, and the **scale** is the one
concept that decides whether the width is allowed to vary.

| Method | `multiplier` from | `scale` |
|---|---|---|
| `global_residual` | empirical quantile of \|residual\| on CAL_CONF | 1 (constant) |
| `conformal_global` | `ceil((n+1)(1-α))`-th smallest \|residual\| | 1 (constant) |
| `state_residual` | empirical quantile of \|residual\|/scale on CAL_CONF | learned |
| `conformal_state` | conformal quantile of \|residual\|/scale | learned |
| `conformal_dispersion` | conformal quantile of \|residual\|/spread | expert spread |
| `quantile_regression` | pinball-loss model of the residual quantiles | (asymmetric) |

`global_residual` is the **baseline** and the thing every other method must beat. It is the
`±1σ` rule a practitioner reaches for by default, which makes it the fairest possible
strawman - and §5 shows it is not good enough.

The same baseline method is applied to **each individual expert** as well, so the answer does
not depend on which point predictor is in use.

### The learned scale

A gradient-boosted model of `E[|residual| | x]` with an L1 objective - the conditional
*median* of `|residual|`, not its mean. The absolute residual has a long right tail (peak
errors are 5-8x typical) and a model of the conditional mean would chase it and inflate every
row's width.

Its inputs are Phase 7's 24 origin-observable columns, reused rather than rebuilt, with one
change: the imputation reference is CAL_FIT rather than the whole panel. The origin-poisoning
guard runs on **every** Phase 8 experiment, not once in Phase 7, because a reused component's
invariants are not automatically preserved.

### What the conformal guarantee does and does not say

Given a fixed point predictor and `n` calibration observations whose residuals are
**exchangeable** with future ones, `[yhat - q, yhat + q]` with `q` the
`ceil((n+1)(1-α))`-th smallest `|residual|` satisfies `P(y_new in interval) ≥ 1 - α`.

**Ordinary time-ordered forecasting residuals are not exchangeable.** A residual from January
and one from November come from visibly different distributions, and §2 shows an 8-11%
difficulty shift inside a single split. So the coverage reported here is an **empirical
measurement on held-out data, not an invoked guarantee**, and the code says so in the string
attached to every conformal interval. A time-ordered (adaptive or weighted) conformal method
is what a real guarantee would need.

---

## 4. What is measured, per method

At four nominal levels (50 / 80 / 90 / 95%) and three horizons, on the sealed test split:

- **coverage** and the signed **coverage error**, with a **PASS/FAIL** flag against a
  pre-stated tolerance;
- **mean and median width**, and the **width/MAE** sharpness ratio;
- **Winkler interval score** per level and the **weighted interval score** across levels,
  which is the only one of these that scores coverage and sharpness together with published
  weights;
- **Spearman correlation** between interval width and realised absolute error, and the
  error multiple carried by the top width decile;
- **reliability by width bin**, and the count of **confident-and-severely-wrong** rows.

### The coverage tolerance, derived before the numbers were seen

`max(0.01, 3 × sqrt(p(1-p)/n))`. At n=179,520 three sigma is 0.0021, so the binding number
is the **1 percentage-point floor**. The floor exists because at this sample size a 0.3pp miss
is statistically detectable and operationally irrelevant, and a pure significance test would
declare it a failure and imply that something should be done about it.

---

## 5. Results

### Coverage at every level (`*` = outside the 0.0100 tolerance)

**h=1**

| method | 50% | 80% | 90% | 95% | ρ(width, error) | top decile |
|---|---|---|---|---|---|---|
| `global_residual` | 0.6993\* | 0.9049\* | 0.9604\* | 0.9836\* | +0.539 | 2.25x |
| `conformal_global` | 0.5065 | 0.8047 | 0.9038 | 0.9534 | +0.507 | 1.60x |
| `state_residual` | 0.5178\* | 0.7795\* | 0.8692\* | 0.9267\* | **+0.709** | **4.62x** |
| `conformal_state` | 0.5270\* | 0.8050 | 0.9020 | 0.9536 | +0.709 | 4.61x |
| `conformal_dispersion` | 0.4979 | 0.8014 | 0.8996 | 0.9494 | +0.664 | 4.70x |
| `quantile_regression` | 0.3803\* | 0.7138\* | 0.8391\* | 0.9164\* | +0.677 | 4.59x |

**h=4**

| method | 50% | 80% | 90% | 95% | ρ | top decile |
|---|---|---|---|---|---|---|
| `global_residual` | 0.4717\* | 0.7937 | 0.9121\* | 0.9653\* | +0.559 | 1.87x |
| `conformal_global` | 0.5030 | 0.7925 | 0.8976 | 0.9516 | +0.571 | 2.19x |
| `state_residual` | 0.4879\* | 0.7902 | 0.9051 | 0.9561 | +0.672 | 4.20x |
| `conformal_state` | 0.5159\* | 0.7925 | 0.8973 | 0.9479 | +0.673 | 4.21x |
| `conformal_dispersion` | 0.4575\* | 0.7782\* | 0.8858\* | 0.9401 | +0.648 | 4.16x |
| `quantile_regression` | 0.3106\* | 0.6556\* | 0.7832\* | 0.8640\* | +0.671 | 4.10x |

**h=96**

| method | 50% | 80% | 90% | 95% | ρ | top decile |
|---|---|---|---|---|---|---|
| `global_residual` | 0.3586\* | 0.6825\* | 0.8211\* | 0.9071\* | +0.643 | 1.82x |
| `conformal_global` | 0.5076 | 0.7890\* | 0.9012 | 0.9538 | +0.665 | 3.20x |
| `state_residual` | 0.5716\* | 0.8347\* | 0.9359\* | 0.9728\* | +0.663 | 4.42x |
| `conformal_state` | 0.5368\* | 0.8053 | 0.9118\* | 0.9571 | +0.667 | 4.42x |
| `conformal_dispersion` | 0.4606\* | 0.7853\* | 0.8934 | 0.9453 | +0.658 | 4.80x |
| `quantile_regression` | 0.2300\* | 0.4205\* | 0.5780\* | 0.7159\* | +0.637 | 3.69x |

### The three findings

**1. The default `±1σ` rule is not calibrated, and the split parity table predicted it.**

`global_residual` fails **11 of 12** level-horizon cells, in opposite directions: it
over-covers at h=1 (0.9836 against a nominal 0.95) and under-covers at h=96 (0.9071). That
is the §2 difficulty shift showing through, and it is why the conformal variants exist and
why the conformal variants pass.

This is worth stating plainly because it is the phase's most transferable negative result: a
constant-width interval calibrated on one half of validation is **not** transferable to a
test period on this data, and no amount of care in choosing the quantile fixes it. Only
`conformal_global` is calibrated at 11 of 12 cells, because it is the only method that
normalises by a per-horizon multiplier fitted on CAL_FIT and then reads a *rank* from
CAL_CONF, which is far less sensitive to a shift in the residual distribution's scale.

**2. Expert disagreement is a real but weak signal, and normalising it destroys the signal.**

Spearman between the expert pool's spread and the fixed ensemble's realised absolute error:

| measure | h=1 | h=4 | h=96 |
|---|---|---|---|
| `max_minus_min` | +0.410 | +0.247 | +0.182 |
| `std` | +0.413 | +0.245 | +0.183 |
| `mean_pairwise_range` | +0.410 | +0.247 | +0.182 |
| `ensemble_minus_each` | +0.412 | +0.244 | +0.178 |
| `disagreement_to_error_scale` | +0.100 | **-0.064** | **-0.244** |

The hypothesis of §23 - "greater expert disagreement may indicate greater forecast
uncertainty" - is **supported in direction and weak in magnitude**. It is positive at every
horizon for the four absolute spread statistics, and it decays with horizon exactly as one
would expect: at 24 hours ahead the three experts have mostly converged on the daily shape,
and what they still disagree about is not what is hard.

The last row is the instructive one. `disagreement_to_error_scale` divides the spread by the
ensemble's own per-unit magnitude and is **negative at h=4 and h=96**. Making a spread
"scale-free" by dividing by level removes the part of it that carries information, because
the level is what the spread is about.

As a *width scale* the spread is the best available choice at h=96 - it is one of only two
methods calibrated there - but it produces the widest intervals in the whole study (15.73 kW
mean width at h=96 against `conformal_state`'s 8.79). It ranks errors well and costs a great
deal of width to do it.

**3. A learned state-conditioned scale is the most informative uncertainty here, and the most
miscalibrated at the nominal levels.**

`state_residual` and `conformal_state` have the highest rank correlation with error
(+0.709 / +0.673 / +0.667) and the top width decile carries **4.2-4.8x** the mean absolute
error, against 1.6-3.2x for `conformal_global`. The width follows the difficulty.

But both fail the 50% level at every horizon, and `state_residual` fails the 90% and 95%
levels at h=1 and h=96. The cause is structural: the scale is a conditional *median* of
`|residual|`, and the empirical quantile of `|residual| / scale` inherits that median's
optimism. When the ratio distribution is tight the interval is narrow and under-covers;
where it is heavy the multiplier grows and over-covers. `conformal_state` repairs part of it
by taking a rank rather than a mean, and is the published method at h=1 and h=4.

### Improvement over the constant-width baseline (weighted interval score)

| method | h=1 | h=4 | h=96 |
|---|---|---|---|
| `conformal_state` (published h=1, h=4) | **+25.5%** | **+21.0%** | +26.8% |
| `state_residual` | +26.9% | +20.7% | **+27.0%** |
| `quantile_regression` | +33.1% | +18.7% | -0.5% |
| `conformal_dispersion` (published h=96) | +19.0% | +17.2% | +17.3% |
| `conformal_global` | +6.2% | -0.6% | +7.2% |

`quantile_regression` has the best score at h=1 and is not published there, because its
coverage at 95% is 0.9164. **The narrowest method is not the calibrated one**, which is why
the verdict's third component restricts its comparison to calibrated methods. Interval score
is a proper scoring rule and does penalise under-coverage - but linearly in the overshoot,
and not enough here to stop an interval that is confidently too narrow from winning.

### Quantile regression, specifically

Fitting residual quantiles directly by pinball loss is the most attractive idea in the study
and it is the worst performer. Coverage degrades with horizon - 0.9164 / 0.8640 / 0.7159 at
a nominal 95% - because the independently fitted pinball models are systematically
under-dispersed at long range, where the residual distribution is widest and most skewed.
The standard monotone rearrangement is applied so the interval stays valid, and the crossing
count is reported at the pairs actually used: **172 rows at h=1, 9 at h=4, 0 at h=96** out
of 359,040.

(A grid-wide crossing count is 34-44% and is meaningless: it counts adjacent fitted
quantiles such as 0.90 against 0.95, which the phase never uses. The count is taken at the
reported pairs and nothing else.)

### The published procedure

Chosen per horizon at the 90% band level, from the measured coverage: among the methods
calibrated within tolerance at that horizon and level, the lowest weighted interval score.

| horizon | procedure | calibrated candidates | WIS kW | coverage error |
|---|---|---|---|---|
| h=1 | `conformal_state` | 3 | 2.156 | +0.0020 |
| h=4 | `conformal_state` | 3 | 3.775 | -0.0027 |
| h=96 | `conformal_dispersion` | 2 | 8.435 | -0.0066 |

All three horizons publish `PASS`.

---

## 6. Regime breakdown

Width tracks error in **every** regime and at every horizon. Width ratios across the
load-level terciles run 7.8x / 7.8x / 16.0x against error ratios of 7.5x / 7.5x / 11.2x
(h=1 / 4 / 96); across the one-step-ramp terciles, 6.8x / 4.1x / 6.4x against 27.9x / 5.8x /
4.9x.

The h=1 load-level breakdown is the clearest statement of what the interval is for:

| regime | n | MAE kW | mean width kW | coverage |
|---|---|---|---|---|
| low | 70,151 | 0.0864 | 0.7501 | 0.9733 |
| typical | 51,714 | 0.2627 | 2.1709 | 0.9160 |
| high | 57,655 | 0.8995 | 9.5336 | 0.9633 |

The width moves by 12.7x across a 10.4x change in error. This is what a decision-assurance
layer would consume: the rows where the interval is narrow are the rows where being wrong
costs 0.09 kW.

Coverage is *not* uniform across regimes - the typical-demand tercile at h=1 sits at 0.9160
against 0.9730 for low - so the published intervals are calibrated **on average**, not
conditionally. That is a limitation, and the right place to fix it is a scale conditioned on
the regime, which the regime labels make awkward precisely because they are computed from the
target being predicted.

---

## 7. Data-quality conditioning: reported as impossible, not simulated

SMART-DS load profiles carry **no per-timestep quality flags**. Phase 3 recorded
`flags=['ok']` across all 35,040 steps, and the `TargetSeries` `DataQuality` is clean for
every row. The conditioning variable therefore has zero variance, and an experiment
conditioned on it would be vacuous.

That is recorded as a measurement with the reason attached, rather than resolved by
manufacturing corruption to condition on - which would have answered a question about the
injected corruption. (Phase 3's 4 unknown and 2 unmapped elements are ingestion-mapping
counts, not per-row input quality.)

For SMART-DS the equivalent question is answered by the horizon and regime breakdowns above,
where difficulty genuinely varies.

---

## 8. The published artifact

`ProbabilisticForecast`, built on the Phase 2 domain objects rather than beside them: a
Phase 8 forecast is a `Forecast` whose optional `uncertainty` field is finally populated.
That required one repair to the domain, recorded as D-099 - `UncertaintyEstimate` counted
`lower_bound` and `upper_bound` as two separate quantities and refused more than one, so the
one representation this phase exists to produce was the one representation the container
could not hold.

Per forecast: the point value, a paired interval at a stated level, a **measured**
calibration status rather than an assumed one, a measured coverage error, a coarse
uncertainty band, the procedure that produced it, the model version, and the row's quality
state.

It carries **no** decision, no confidence score for an action, and nothing derived from
uncertainty and an objective together. Those belong to Phase 13's decision-assurance layer,
and defining them here would fix an assurance formula on the evidence of one forecasting
experiment.

### Two shapes that were wrong and are now right

**Band cut points are per-horizon.** A single tercile pair across horizons whose mean widths
differ by 5x called 61% of h=1 rows "low" and 43% of h=96 rows "high" - a label that tracked
the horizon rather than the row. Cut points are now terciles of each horizon's own
calibration width distribution.

**The method name is per-horizon.** No single procedure was both calibrated and
best-scoring at all three horizons. Publishing one name for all three would credit a horizon
with a method that was not selected for it.

---

## 9. Bugs this phase found in its own machinery

Recorded because each would have produced a plausible, wrong table.

| Defect | How it would have shown |
|---|---|
| `quantile_levels` returned one-sided tails `(1-α, α)` while its docstring specified `α/2` | "nominal 90%" intervals covering 80%, and a nominal 50% interval collapsing to the median with zero width. The symmetric methods were unaffected, so only quantile regression would have been silently wrong. |
| `_assemble_quantile_intervals` refilled one shared bounds array per level and returned it under every key | Four nominal levels, four identical intervals, coverage repeating to six decimal places because one interval was being measured four times. |
| The artifact published the *baseline* method | `calibration_status` reading `FAIL, FAIL, FAIL` on an artifact presented as a result. |
| The verdict rewarded the narrowest method regardless of calibration | `is_worth_the_machinery: YES` earned by an interval that under-covered by 4pp. |
| Band cut points pooled across horizons | A band that reported the horizon, not the uncertainty. |
| The poisoning guard raised `IndexError` instead of naming a column mismatch | A leakage guard failing with a message that reads as a bug in the guard. |
| `interval_to_estimate` indexed a `[1, 1]` array as if flat | The domain join raised on every single-forecast conversion. |

---

## 10. What this phase does not establish

- **Not a coverage guarantee.** The conformal guarantee is conditional on exchangeability,
  and §2 shows the assumption is violated on this data. The reported coverages are empirical.
- **Not an aleatoric/epistemic decomposition.** See §1.
- **Not conditional calibration.** Coverage is uniform on average and varies by up to 6pp
  across regime terciles.
- **Not a decision rule.** No objective, no cost, no action.
- **Not a claim that disagreement is a good uncertainty signal.** It is positive but weak
  (ρ = 0.18-0.41) and it is expensive in width.
- **Not a claim about other datasets.** Every number here is one 40-customer SMART-DS sample
  with no missing-data flags, 96 steps per day, and horizons of 1, 4 and 96.

## 11. The direction this points to

A **time-ordered conformal** calibrator (adaptive or weighted), which is the only way to turn
the conditional guarantee into a real one on non-exchangeable residuals. It would address the
one thing this phase could not: the 8-11% difficulty shift across the split that makes the
default constant-width rule fail in opposite directions at h=1 and h=96.