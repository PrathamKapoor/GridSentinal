# The heterogeneous Energy Expert Router

Phase 7's design, and what it measured. This document replaces the assumption that Phase
7 would add experts inside Qwen. It explains what was built instead, why, and what the
result was.

Every number here comes from `artifacts/phase7/router-main/result.json`, produced by
`energy-intel router run` on the same 179,520 test rows Phase 4 and Phase 5 reported.

---

## 1. The architecture changed, and why

Phase 6 tested `Qwen3-1.7B-Base` as a frozen backbone with an energy prediction head. It
lost to every existing baseline at every horizon:

| Model | h=1 | h=4 | h=96 |
|---|---|---|---|
| persistence | **0.3952** | 0.8925 | 2.4427 |
| Phase 5 TCN | 0.4081 | **0.8224** | 1.6522 |
| classical GBM | 0.4946 | 0.9749 | **1.6168** |
| Qwen probe, best head | 0.5452 | 1.6060 | 2.7400 |

And it measured *why*: the frozen backbone's final-layer readout has a participation
ratio of **2.1** across 2,048 dimensions. Whatever survives to the top of that trunk is
almost rank-2. An MoE built inside it would inherit the collapse.

So Phase 7 routes over **heterogeneous models that already work**:

```text
                    CURRENT ENERGY STATE
                            |
                            v
                     ROUTER FEATURES
                    (origin-observable)
                            |
                  +---------+---------+
                  |    ENERGY ROUTER  |   1,097 parameters
                  +---------+---------+
                            |
           +----------------+----------------+
           v                v                v
     Persistence        GBM                TCN
       Expert           Expert            Expert
      0 fitted      36,600 nodes       74,099 weights
           |                |                |
           +----------------+----------------+
                            |
                    EXPERT PREDICTIONS
                            |
                     ROUTER WEIGHTS
                            |
                     MIX / SELECT
                            |
                   FINAL FORECAST
```

This is a mixture of experts in the real sense - multiple specialist predictors, a
mechanism that decides how much each contributes, and a final forecast built from the
routed outputs - but the experts are different *model classes*, not identical neural
subnetworks. It is not "a transformer with sparse expert FFN layers".

---

## 2. The expert pool

Three models, deliberately **not** made alike. Their differences are the entire reason
routing has anything to do.

| Expert | What it is | Why it is in the pool | Parameters |
|---|---|---|---|
| `persistence` | `yhat(t+h) = y(t)`, unfitted | Almost no inductive assumption, and untouchable at 15 minutes | 0 |
| `classical_hist_gbm` | Phase 4's `HistGradientBoostingRegressor`, one per horizon | Strongest classical baseline; nonlinear feature map | 36,600 tree nodes |
| `phase5_tcn` | Phase 5's dilated causal TCN, from its committed checkpoint | Learned temporal representation over a 7-day window | 74,099 weights |

### 2.1 The GBM is Phase 4's GBM, reproduced rather than approximated

The first attempt fitted the GBM on **kW** targets with unscaled features. It scored
**10.09 kW** where Phase 4 scored **0.42 kW**. Phase 4 fits on **per-unit** targets with a
`FeatureScaler` fitted on training rows, plus three per-series context columns whose two
statistics come from training rows only (`experts.py:_phase4_series_context`). Reproducing
the model means reproducing all of that.

With it corrected, the Phase 7 experts reproduce Phase 4 and Phase 5 to the digit:

| Expert | Phase 7 test MAE (h=1 / h=4 / h=96) | Phase 4/5 published | Max relative difference |
|---|---|---|---|
| persistence | 0.4043 / 0.8897 / 2.4343 | 0.4043 / 0.8897 / 2.4343 | 0.000% |
| classical GBM | 0.4242 / 0.8576 / 1.5737 | 0.4242 / 0.8579 / 1.5648 | **0.570%** |
| Phase 5 TCN | 0.4178 / 0.8307 / 1.6510 | 0.4178 / 0.8307 / 1.6510 | 0.000% |

The GBM's h=96 differs by 0.57% and the reason is known, not assumed: Phase 7 recomputes
the Phase 4 feature matrix in float32 while the Phase 4 artifact stores it that way, and
`HistGradientBoostingRegressor` bins features from a 200,000-row random subsample. A
1e-7 difference in a feature value moves a quantile across an observation, which moves a
bin edge, which changes the fitted trees. The check is automated
(`experiment.py:_parity`, tolerance 2%) so a genuine regression cannot hide behind a
float-rounding excuse.

### 2.2 Every expert speaks kW

Phase 5's `predict_series` returns **per-unit** levels. Phase 6 shipped a bug where a
per-unit prediction was compared against a kW target, giving a 13.3 kW error where the
truth was 0.4178 kW. The conversion happens once, in `TcnExpert.predict`, and
`ExpertPool.forecast_block` refuses any expert that breaks the contract.

---

## 3. Router inputs

Everything the router sees is a function of the demand **at or before the forecast origin**.
Six groups, 24 columns, switchable by name so an ablation can drop one.

| Group | Columns | What it is |
|---|---|---|
| `level` | `level_pu`, `level_kw` | Demand now, per-unit and in kW |
| `trajectory` | `traj_mean_4`, `traj_mean_24`, `ramp_1` | Trailing 1 h and 6 h means, and the last 15-minute ramp |
| `volatility` | `vol_std_96` | Rolling standard deviation over 24 h |
| `calendar` | `hour_sin/cos`, `doy_sin/cos`, `weekend` | Sin/cos of hour-of-day and day-of-year |
| `scale` | `rated_kw_log`, `level_kw` | The customer's rated kW, and demand in physical units |
| `past_error` | `past_{h}_e{e}` x 3 experts, `past_seen_{h}` | Each expert's **lagged realised** error per horizon |

**The regime terciles are deliberately absent.** Phases 4, 5 and 6 slice the data into
demand and ramp terciles, and those slices are excellent diagnostics. They are computed
from the demand *being predicted*, so a router cannot use them without leaking the target.
They are used in this phase's evaluation; they are not inputs.

### 3.1 The lagged expert error, and the `h + 1` that matters

`past_error` is the only group that reads the experts rather than the demand, and it is
where a leak would be easiest to write. At origin `t` the router may legitimately know how
each expert has *already* performed. For horizon `h`, the most recent forecast whose
outcome is known at `t` is the one issued at origin `t - h - 1`, which targets `t - 1`:

```text
past_error[e, h] = | f_e(t - h - 1; h) - y(t - 1) |  /  rated_kw
```

Using the forecast issued at `t - 1` instead would read a target at `t + h - 1`, which has
not happened yet. The delay is `h + 1` for that reason and no other.

The values are normalised by rated kW. A router weighting on raw kW error would learn one
policy for a 1.4 kW household and another for a 388 kW shop, from 40 series.

`test_lagged_error_reads_the_forecast_issued_h_plus_1_steps_earlier` checks the arithmetic
row by row against the panel.

### 3.2 Missing values

The first `h + 1` origins of each split have no predecessor: about 0.05% of rows at h=1
and 2.2% at h=96, none at all in the test split. Those cells are `NaN`, a
`past_seen_{h}` flag says so, and `fill_missing` imputes the column mean **computed from
the router's own training rows**. Imputing from the whole panel would let a test row's
error rate choose the value a test row is scored against.

---

## 4. Leakage protection

Three independent mechanisms, because one would not be enough.

**A poisoning test on every feature.** For a sampled row, every value after its own origin
in its series is overwritten with a large negative sentinel, and the features of *every row
of that series up to that origin* are rebuilt and required to be bit-identical. The
realised targets are recomputed from the poisoned series rather than passed in, so a
look-up that reached past the origin really would change. `test_causality_check_actually_catches_a_leaking_builder`
injects a deliberately leaking builder and asserts the guard fires, because a guard that
has never rejected anything is not evidence of anything.

**Cross-fitting, structurally.** The learned experts are fitted on the **training** split.
The router is fitted on a chronological **70% of validation**, early-stopped on the last
30%, and evaluated on **test** once. Every expert prediction the router ever learns from is
therefore out-of-sample.

**Oracle isolation.** The oracle lives in `router/offline/oracle.py`. No module in
`router/` imports anything from `router/offline/`, and two tests enforce it by reading the
source and the AST: one checks imports, one checks that no deployable module *names* an
oracle in code (prose is allowed, calls are not).

### 4.1 How large was the avoided effect?

`offline/crossfit.py` measures it rather than asserting it. The validation block is halved
in time, one GBM is fitted on the first half, and the oracle gain over persistence is
computed on rows it did and did not see:

| Horizon | In-sample oracle gain | Out-of-sample oracle gain | Ratio |
|---|---|---|---|
| h=1 | 21.57% | 21.12% | 1.02 |
| h=4 | 19.31% | 29.80% | **0.65** |
| h=96 | 18.39% | 37.49% | **0.49** |

The direction is worth stating precisely because it is not the naive worry. An expert that
looks better in-sample than it is makes the pool look *more* interchangeable, so the oracle
gain **shrinks**: routing labels built from in-sample predictions would have understated
the available headroom at long horizons by roughly half. A router trained on them would
have chased a ceiling that was too low, and would have reported a small honest improvement
as if it were the whole opportunity.

---

## 5. Router architecture

```text
24 features
  -> [Linear(24, 32), ReLU]        shared trunk
  -> Linear(32, 3 experts x 3 horizons)
  -> softmax over experts, per horizon
  -> [N, 3 horizons, 3 experts] weights, rows summing to 1
```

1,097 parameters. One output head per horizon, because the expert ranking genuinely changes
with horizon and Phase 5's `single_horizon` ablation showed the *trunk* should stay shared.
Horizon is not a feature; it is an output axis.

The router does **not** see the experts' forecasts. It could - they have already been run,
and their spread is genuinely informative - but then it stops being a router and becomes a
blender that can regress a new forecast rather than a weight over experts that exist.

### 5.1 The training objective, and why not the alternatives

The loss is the routed forecast's own absolute error in kW. Not a classification label, not
a regression onto oracle weights.

- **Not classification.** A label discards the magnitude of the win. If the GBM beats
  persistence by 0.0001 kW on one row and 2 kW on the next, both are labelled "GBM". Worse,
  the achievable forecast from a label is "use that expert", which is the *hard* routing
  upper bound - it discards the possibility that a blend beats both.
- **Not regressing oracle weights.** Oracle weights are computed from the realised target.
  Regressing on them teaches the router to imitate hindsight, and hindsight weights are not
  what minimises expected error under uncertainty.
- **The direct objective** is the quantity the phase asks about. Weights become one-hot or
  fractional exactly when that minimises error.

#and why the cold-start variant is also trained

Starting near uniform, an L1 objective through a softmax **collapses onto a single expert**
within a few epochs. Measured: the cold variant settles at mean weights (0.24, 0.00, 0.76)
at h=1 and (0.34, 0.00, 0.66) at h=4, because the gradient through the softmax vanishes as
a vertex is approached. At h=1 the collapse happened to be harmless; at h=4 it cost more than
a fixed weighted ensemble achieved, since a hedge that reduces absolute error is exactly the
solution a vertex cannot express.

So two variants are trained from the same rows with the same protocol:

- **warm** - initialised at the fixed-ensemble weights fitted on the router's own training
  rows (head bias `log(w)`, head weights zeroed, so epoch 0 *is* that mixture exactly);
- **cold** - initialised near uniform.

Epoch 0 is scored and eligible for selection, so a warm-started router can never regress
below where it started on the selection rows. The variant is chosen **per horizon on the
selection rows only**:

| Horizon | Warm selection MAE | Cold selection MAE | Chosen |
|---|---|---|---|
| h=1 | 0.4316 | **0.4260** | cold |
| h=4 | **0.8331** | 0.8456 | warm |
| h=96 | 1.5247 | **1.5014** | cold |

And then the warm router **selected epoch 0 at every horizon.** Gradient descent found no
state-dependent weighting that improved on the constant one it started from, at any horizon,
on this data. The warm variant's weights therefore *are* the fixed ensemble's weights, and
its test MAE reproduces the fixed ensemble's to float noise (0.3983 / 0.7917 / 1.5400
against 0.3983 / 0.7917 / 1.5399).

Everything the deployed router learned comes from the cold variant, and its validation gains
did not transfer:

| Horizon | Cold wins on validation by | Cold on test | Warm (= fixed) on test | Transfer |
|---|---|---|---|---|
| h=1 | 0.0056 kW | 0.4069 | 0.3983 | **failed** - worse |
| h=96 | 0.0232 kW | 1.5600 | 1.5400 | **failed** - worse |
| h=4 | loses by 0.0125 kW | - | 0.7917 | not selected |

A validation gain of 0.0056 kW is a sixth of a watt per forecast, and it did not survive.

---

## 6. Baselines the router has to beat

All built from the same cached expert forecasts, so no comparison can be decided on
different rows.

| Method | What it is |
|---|---|
| `persistence`, `classical_hist_gbm`, `phase5_tcn` | The experts, alone |
| `uniform_ensemble` | Equal weights. The null hypothesis for "combining helps at all" |
| `best_single` | The one expert with the lowest MAE per horizon, chosen on router-training rows |
| `fixed_ensemble` | One weight vector per horizon from an exhaustive simplex search on router-training rows |
| `router_soft` | The learned weights |
| `router_hard` | `argmax` of the same weights - not a separately trained model |
| `oracle_expert_assignment` | Per-row best expert with hindsight. Upper bound on hard routing |
| `oracle_mixture` | Best *constant* convex combination with hindsight. **Not** an upper bound on the router |

The simplex search is a 0.01-step grid, 5,151 points per horizon, scored in row chunks. With
three experts this is deterministic, reproducible and finishes in a second; there is nothing
a gradient descent would add except a seed to get wrong.

### 6.1 `oracle_mixture` is not an upper bound, and the phase proved it

The deployed router's h=1 forecast (0.4069 kW) does not beat `oracle_mixture` (0.3956 kW).
The relationship was still established and tested, on a constructed case where a
state-dependent rule does beat the best constant one: three experts predicting the constants
0, 5 and 10 kW against a series alternating between 0 and 10. Any fixed combination is
itself a constant, so it cannot beat 5 kW of mean error whatever the weights are, while
choosing the low expert on low rows and the high expert on high rows is exact.

This corrects a claim the module originally made in its own docstring, and
`test_oracle_mixture_is_not_an_upper_bound_on_a_state_dependent_router` asserts the
relationship so the docstring cannot drift back.

---

## 7. Result

### 7.1 Expert performance, sealed test split, kW MAE

| Method | h=1 | h=4 | h=96 |
|---|---|---|---|
| persistence | **0.4043** | 0.8897 | 2.4343 |
| classical GBM | 0.4242 | 0.8576 | 1.5737 |
| Phase 5 TCN | 0.4178 | 0.8307 | 1.6510 |
| uniform ensemble | 0.3955 | 0.7943 | 1.7615 |
| best single | 0.4178 | 0.8307 | 1.6510 |
| fixed ensemble | **0.3983** | **0.7917** | **1.5399** |
| router (soft) | 0.4069 | 0.7917 | 1.5600 |
| router (hard) | 0.4150 | 0.8307 | 1.5709 |
| oracle expert assignment | 0.2871 | 0.5217 | 0.9942 |
| oracle mixture | 0.3956 | 0.7911 | 1.4956 |

### 7.2 Verdict: PARTIALLY - and the negative half is the important half

Two comparisons are reported separately and the verdict is the stricter one, because the
looser one is the easier claim to make.

| Comparison | Answer | Detail |
|---|---|---|
| vs the best single expert | **PARTIALLY**, 2 of 3 | wins h=4 by 4.70% and h=96 by 0.87%; **loses h=1 by 0.64%** |
| vs a fixed weighted ensemble | **NO**, 0 of 3 | h=4 tied; loses h=1 by 2.16% and h=96 by 1.31% |

**The router did not improve on a fixed weighted ensemble at any horizon.** Combining the
experts beats the best single expert at two horizons, but that benefit is captured entirely
by a constant weighting - by definition, since the warm router's weights *are* the constant
weights.

A "win" must exceed 0.0001 kW - a tenth of a watt. Two methods differing by 2e-6 kW are the
same method, and counting that as a result would be a rounding artefact. The h=4 tie is
such a case: the router reproduces the fixed ensemble to six significant figures.

### 7.3 Expert diversity

| Horizon | Best expert | MAE | Mean error correlation | Winner margin |
|---|---|---|---|---|
| h=1 | persistence | 0.4328 | 0.8997 | 0.08% |
| h=4 | Phase 5 TCN | 0.8319 | 0.8216 | 4.62% |
| h=96 | Phase 5 TCN | 1.5837 | 0.6862 | 1.73% |

Overall: mean pairwise Pearson correlation of absolute errors **0.7134**, Spearman
**0.7553**, and of **signed** errors **0.6438**.

Error correlation falls with horizon - 0.90 at 15 minutes, 0.69 at 24 hours - which is the
right direction for routing: the further ahead the forecast, the more the experts disagree
about what will happen. The signed correlation being lower than the absolute one matters
too: two experts that are *equally* wrong and *oppositely* wrong is the case where a 50/50
blend beats both.

Agreement, defined as two forecasts within 1% of the row's demand (0.05 kW floor):

| Pair | Agreement | Mean gap | p99 gap |
|---|---|---|---|
| persistence - GBM | 0.2476 | 0.9559 kW | 12.24 kW |
| persistence - TCN | 0.4115 | 0.7441 kW | 10.44 kW |
| GBM - TCN | 0.2590 | 0.6607 kW | 7.54 kW |

**The experts disagree on 59-75% of rows.** That is not a pool of three similar models; on
almost any given forecast there is a real choice to make.

### 7.4 The oracle

| Horizon | Oracle MAE | Best single | Gain available | Best fixed mixture |
|---|---|---|---|
| h=1 | 0.3028 | 0.4328 | 30.05% | 0.4149 |
| h=4 | 0.5318 | 0.8319 | 36.07% | 0.7970 |
| h=96 | 0.9795 | 1.5837 | 38.15% | 1.4852 |

**The MoE premise holds on diversity and fails on capture.** 30-38% of the achievable gain
is present in the pool, the experts disagree on most rows, and the best expert changes with
both horizon and regime. The router captures **-2% to 13%** of it, and the capture ratio is
*negative* at h=1, meaning the router did worse than a constant weighting on that horizon.
Cutting the pool to pairs shows the third expert earns 0.003-0.009 kW at h=1: it is not dead
weight, but neither is it transformative.

### 7.5 Routing quality

| Horizon | Selection accuracy | Routing regret (kW) | Capture ratio | Mean top weight |
|---|---|---|---|---|
| h=1 | 0.2327 | 0.1198 | **-0.022** | 0.7866 |
| h=4 | 0.2733 | 0.2700 | 0.126 | 0.6599 |
| h=96 | 0.3892 | 0.5658 | 0.024 | 0.9692 |

Selection accuracy of 23-39% against a three-way choice is close to chance (33%). The router
is not identifying which expert is right; it is applying a near-constant preference.

Definitions, carried in every artifact so two reports cannot disagree while both sound
right (`oracle.routing_regret_definition`):

- **routing regret** = router MAE - oracle MAE. Zero means the router matched the oracle's
  selection exactly; positive means it gave up part of the achievable gain.
- **capture ratio** = 1 - regret / (oracle's gain over the best single expert). Zero means
  none of the headroom was captured, one means the oracle was matched, negative means the
  router did worse than a constant weighting.
- **selection accuracy** = share of rows where the highest-weight expert equals the
  oracle's.

### 7.6 Routing stability and calibration

| Horizon | Mean total variation | Median | p99 | Selection flip rate |
|---|---|---|---|---|
| h=1 | 0.1902 | 0.0421 | 1.5413 | 0.1314 |
| h=4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| h=96 | 0.0503 | 0.0004 | 1.2004 | 0.0227 |

Measured within a series, across consecutive origins. At h=1 the router flips its selected
expert on 13.1% of adjacent 15-minute steps while the median step barely moves: mostly
stable with a minority of sharp transitions. At h=4 the weights are literally constant. No
smoothing penalty is applied, because there is no evidence that oscillation costs accuracy
and imposing one would be an unmeasured assumption.

**The weights are not calibrated probabilities.**

| Horizon | Mean top weight | Selection accuracy | Expected calibration error |
|---|---|---|---|
| h=1 | 0.7866 | 0.2327 | **0.5539** |
| h=4 | 0.6599 | 0.2733 | 0.3866 |
| h=96 | 0.9692 | 0.3892 | **0.5800** |

At h=96 the router assigns an average top weight of 0.97 to an expert that is in fact the
most accurate 39% of the time. Weights summing to one is a normalisation, not a claim. In a
live system this is the number to fix before the weights are shown to anyone.

### 7.7 Router behaviour by regime, and the mechanism of the failure

**load_level** - terciles of actual per-unit demand

| Regime | n | Router | Best single | Fixed ensemble | Oracle | Router weights |
|---|---|---|---|---|---|---|
| low | 70,151 | 0.0864 | **0.0792** | 0.0864 | 0.0613 | 0.00 / 0.29 / 0.71 |
| typical | 51,714 | **0.2622** | 0.2709 | 0.2627 | 0.2048 | 0.00 / 0.34 / 0.66 |
| high | 57,655 | 0.9266 | **0.9197** | 0.8995 | 0.6356 | 0.00 / 0.24 / 0.76 |

**load_ramp** - terciles of the absolute step-to-step change

| Regime | n | Router | Best single | Fixed ensemble | Oracle | Router weights |
|---|---|---|---|---|---|---|
| low | 69,863 | 0.0480 | **0.0051** | 0.0392 | 0.0041 | 0.00 / 0.26 / 0.74 |
| typical | 52,587 | 0.1393 | **0.0744** | 0.1181 | 0.0531 | 0.00 / 0.28 / 0.72 |
| high | 57,070 | **1.0929** | 1.1970 | 1.0962 | 0.8490 | 0.00 / 0.33 / 0.67 |

(weights: persistence / GBM / TCN)

**This is where the phase's result comes from, and it is a single mechanism.** The router
learned to route essentially everything to the TCN and the GBM, assigning persistence a mean
weight between **0.0000 and 0.0011** at every horizon. It wins only in the high-ramp tercile,
where the TCN's temporal model genuinely is the best tool. In the other five regimes it
loses, and in the low-ramp tercile it is **9.4x worse** than persistence (0.0480 against
0.0051 kW).

`docs/moe_design_requirements.md` §6.7 named this risk in advance:

> The low-ramp regime, where persistence reaches 0.0034 kW, is the one a router is most
> likely to damage.

It was damaged, by the largest factor in the phase. Persistence is exact when demand is not
moving; mixing a learned model into that regime can only add error. The learned features -
level, trailing means, 24-hour volatility, calendar - do not distinguish "flat" from "about
to ramp" sharply enough at the 15-minute horizon where the router operates, so the router
cannot learn to hand those rows to the rule that is right for them.

### 7.8 Failure analysis

At h=1, 110,499 of 179,520 rows (61.6%) are worse than the best single expert by some
amount, and the router is 0.64% worse on average. That is the whole result in one line.

Broken down by what the router chose:

| Selected | Rows | Hit rate | Mean regret | Regret when right | Regret when wrong |
|---|---|---|---|---|---|
| Phase 5 TCN | 149,260 | 0.203 | 0.1024 | 0.0197 | 0.1234 |
| classical GBM | 30,247 | 0.380 | 0.2059 | 0.0708 | 0.2886 |
| persistence | 13 | 0.846 | 0.0188 | 0.0108 | 0.0624 |

The router selected persistence **13 times out of 179,520**. Regret when the router chose
wrong is 4-6x regret when it chose right, at every selection. The 20 worst individual
routing mistakes are listed in `result.json` with their origins, weights and expert
forecasts; their realised errors exceed 10 kW.

### 7.9 Compute cost

| Measurement | Value |
|---|---|
| Router parameters | 1,097 |
| Router training | 0.89 s |
| Router inference | 0.094 s over 179,520 rows |
| Router per row | 5.2e-7 s |
| Expert inference | 174.8 s over 359,040 rows |
| Expert per row | 4.9e-4 s |
| **Router overhead vs experts** | **0.11%** |
| Phase 6 reference (frozen 1.7B backbone) | 0.248 s/row |

The router is a rounding error next to the experts, and roughly 475,000x cheaper per row
than Phase 6's frozen backbone. **Compute was never the reason this failed**, and neither
cheapness nor expense is evidence for building more of it: the constraint on this problem is
the data and the experts, not the gate.

One caveat stated in the artifact: routing does not avoid running the experts it routes
over, so its cost is **additive**. Hard routing would only pay for itself if expert inference
could be gated, which this implementation does not do.

---

## 8. Ablations

Four, each on identical rows with the identical protocol, differing only in the feature
contract. A positive delta means removing the group made the forecast *worse*.

| Ablation | Question | h=1 | h=4 | h=96 | Verdict |
|---|---|---|---|---|---|
| `no_past_error` | does the router need to know how the experts have recently been performing? | 0.3983 (-0.0086) | 0.7917 (0.0000) | 1.5400 (-0.0200) | does not help |
| `no_trajectory` | is the recent demand path carrying the routing decision, or only the current level? | 0.3934 (-0.0135) | 0.8705 (+0.0788) | 1.5407 (-0.0192) | helps at h=1 and h=96, hurts badly at h=4 |
| `no_scale` | does the router need the customer's rated kW? | 0.4237 (+0.0168) | 0.8105 (+0.0187) | 1.5316 (-0.0283) | mixed |
| `shared_horizon_head` | do the experts' strengths really change with horizon? | 0.4081 (+0.0012) | 0.7968 (+0.0051) | 1.5230 (-0.0370) | mixed |

**These ablations are another negative result, and they corroborate the main one.** With the
calendar bug fixed, two of the four feature groups are *negative* contributions: removing
`past_error` improves h=1 and h=96, and removing `trajectory` improves h=1 and h=96. The
groups that were carrying the earlier result were not earning their place.

`no_scale` is the one group that clearly earns its keep at short horizons (+0.0168 kW at
h=1, +0.0187 at h=4 when removed), which is Phase 4's own finding reproduced: a pooled model
over 40 customers cannot tell a 1.4 kW household from a 388 kW shop unless told.

`no_trajectory`'s +0.0788 kW at h=4 is worth separating out: it is the cold variant's
soft-hedge, which at h=4 collapses to the TCN anyway. Its behaviour is not evidence about
the trajectory group; it is evidence about the h=4 collapse.

`shared_horizon_head` keeps every feature and changes only the output head, so its comparison
is not confounded with a change of inputs. It is better at h=96 by 0.0370 kW, which is a
point *against* the per-horizon head design at that horizon - the opposite of what the
per-horizon structure was argued for.

---

## 9. What Phase 7 concludes

**The MoE premise holds on diversity.** The three experts make materially different
mistakes: they disagree on 59-75% of rows, error correlation falls from 0.90 to 0.69 as the
horizon lengthens, the best expert changes with both horizon and regime, and 30-38% of
achievable accuracy sits in the pool. A heterogeneous expert pool is a reasonable thing to
build for this problem.

**A learned router did not exploit it.** Measured on the sealed test split:

- Versus the best single expert, PARTIALLY: 2 of 3 horizons, and it **loses at 15 minutes**,
  the shortest and most favorable horizon for routing.
- Versus a fixed weighted ensemble, NO: 0 of 3 horizons. That is the comparison that
  decides whether the learned, state-dependent component is doing any work, and it is doing
  none.
- Warm-started from the state-independent optimum, gradient descent selected **epoch 0 at
  every horizon**: no state-dependent weighting improved on a constant one.
- Cold-started, it found validation gains of 0.006-0.023 kW that did **not** survive to test.
- Selection accuracy is 23-39% against a three-way choice, close to chance.
- The router abandoned persistence (13 selections in 179,520 rows) and is 9.4x worse than
  it in the low-ramp tercile.

**So the phase's answer, stated as measured rather than as planned:**

> Different forecasting experts do have different strengths across energy conditions, and
> 30-38% of achievable accuracy is available between them. On this data, **a learned
> energy-aware router does not capture it**: a fixed weighted ensemble is better at every
> horizon, and the router's advantage over any single expert comes from combining rather
> than from routing on state.

That is a negative result, and it is the phase's product. It is also a more useful one than
a marginal win, because the failure has a diagnosed mechanism (§7.7): the router's learned
features cannot distinguish "demand is flat" from "demand is about to ramp" at 15-minute
resolution, so it cannot learn to hand the flat rows to the rule that is exact for them.

### 9.1 What would be worth trying, and what would not

Stated as directions for a **pre-registered** experiment, not as fixes to apply now. Acting
on any of them using these test results would be exactly the tuning against the test split
that D-075 and D-090 forbid.

| Direction | Why it might work | Why it might not |
|---|---|---|
| A learned low-volatility gate that hands flat rows to persistence | Directly targets the diagnosed failure: the low-ramp tercile is where the router loses 9.4x | A gate that fires on the wrong rows is worse than no gate; and the regime label itself is computed from the future, so a *trailing*-window rule is needed, which is exactly the proxy this phase could not make work |
| Router features at a coarser temporal resolution | The router may simply lack the resolution at which "flat" is detectable | Phase 5 measured that coarser context hurts the experts too |
| More router capacity, or a larger warm-start search | - | The warm start already searched and preferred its own initialisation; more capacity would buy overfitting, not signal |
| Per-customer routers | Error concentrates by regime, not by customer; Phase 6 rejected this | 40 series cannot support 40 routers |
| Qwen experts | - | Phase 6's result stands unchanged and no new experiment argues against it |

The honest recommendation for the next phase is **not** a bigger router. It is either a
fixed weighted ensemble, which is the best deployable combiner measured here, or a
pre-registered experiment on volatility-gated routing.

---

## 10. What is deliberately not built

No Qwen experts (§1). No agents, optimiser, digital twin, MLOps layer, scheduler or UI.
The Phase 7 output is a forecast plus routing metadata, and nothing was allowed to control a
battery, an EV or a feeder.

The router is a forecasting architecture, not a world model. SMART-DS provides no
action-conditioned transitions, so there is nothing here that could support that claim.

---

## 11. Reproducing

```text
energy-intel router experts    refit the pool, cache 359,040 rows of forecasts (~5 min)
energy-intel router run        diversity, oracle, router, ablations, sealed evaluation
energy-intel router config     the resolved experiment configuration
energy-intel router experiments  the recorded result
```

Artifacts, all regenerable:

```text
artifacts/phase7/experts.npz           cached expert forecasts, the single source for every number
artifacts/phase7/experts.json          provenance: fits, seeds, timings, sizes
artifacts/phase7/router-main/result.json        everything measured
artifacts/phase7/router-main/summary.md          the tables in this document
artifacts/phase7/router-main/diversity.json      error correlations, agreement, regime splits
artifacts/phase7/router-main/regimes.json         per-regime expert preference
artifacts/phase7/router-main/router.pt            the warm-start checkpoint (selected epoch 0)
artifacts/phase7/router-main/router_cold.pt       the cold-start checkpoint (the one that shipped)
artifacts/phase7/router-main/router_weights.npz   every test row's weights and selection
artifacts/phase7/router-main/routing_trace.jsonl  per-forecast audit trail
artifacts/phase7/router-main/training_log.jsonl   per-epoch record
experiments/registry.jsonl                        the cross-phase record
```

The routing trace carries, per forecast: timestamp, asset, horizon, all 24 router features,
the three expert weights, the selected experts, each expert's own forecast, the final
forecast and the realised value. A 2,000-row sample is written in readable JSON; every test
row's weights are written in full as float32. That trace is what caught a calendar-feature bug
during this phase - the weekend flag was set on a Wednesday - which had been flattering the
router's h=1 result by 0.02 kW.
