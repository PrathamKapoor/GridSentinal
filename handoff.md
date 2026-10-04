# Handoff - Phase 8

**Project:** Energy Intelligence - adaptive, self-verifying energy management
**Event:** Yuva Yodha Energy Tech Hackathon 2026, Schneider Electric
**Current phase:** **Phase 8 of 20 - Calibrated predictive uncertainty - COMPLETE**
**Previous phases:** Phases 1-7 complete and preserved, zero regressions
**Date:** 2026-10-04
**Author of all work:** PrathamKapoor <prathamkapoor027@gmail.com>

> Phase 7 was completed but never handed off; this document covers **both** Phase 7 and
> Phase 8. Phase 6's handoff is preserved in git history at commit `66bd723`.

---

## 1. The answer, first

Two questions were answered, one in each phase, and **both are negative where the community
would expect a positive**.

> Do heterogeneous forecasting experts have different strengths across energy conditions,
> and can a learned energy-aware router exploit that?

**The premise holds; the router does not.** The oracle shows 30-38% of achievable gain is
present in the expert pool and the experts disagree on 59-75% of rows. A learned,
energy-aware gate captured **-2% to 13%** of it, and lost to a fixed weighted ensemble at
**0 of 3** horizons.

> Can calibrated, informative uncertainty be produced around the energy demand forecast, and
> does it identify when the forecast cannot be trusted?

**YES**, on all three components: coverage within 0.0100 of nominal at every horizon for
three of six methods, a positive rank correlation between interval width and realised error
at every horizon with the top width decile carrying 4.2-4.8x the mean error, and a 19-27%
weighted-interval-score improvement over the constant-width baseline.

**The most useful finding of Phase 8 is negative:** the default `±1σ` interval - the rule a
practitioner reaches for without thinking - **fails 11 of 12 coverage cells**, in opposite
directions at the two ends of the horizon range.

---

## 2. The point forecast is fixed, and that is enforced

Neither phase changed what the system forecasts. Phase 8 recovers Phase 7's fitted
fixed-ensemble weights from `artifacts/phase7/router-main/result.json`, checks them against
`configs/uncertainty.toml`, and **stops the run if they disagree**. The recovered ensemble
reproduces Phase 7's published test MAE to **0.0000%**, which is the parity check that makes
every uncertainty number in the phase comparable with Phase 7's.

This is structural, not a convention. The config loader refuses weights that differ from
Phase 7's recorded values, refuses a renamed expert pool, and refuses rows that do not sum to
one.

## 3. The Phase 8 calibration split

```text
TRAIN          950,400 rows. The experts were fitted here.
                      |
VALIDATION     179,520 rows. Expert predictions are out-of-sample.
    +-- CAL_FIT      first 50%, 89,760 rows. Scale functions, pinball models.
    +-- CAL_CONF     last  50%, 89,760 rows. Conformity scores, band cut points.
                      |
TEST           179,520 rows. Read once, at the end.
```

Validation is halved rather than reusing Phase 7's 70/30 router split, because the scale must
not be fitted on rows the ensemble's weights were fitted on.

**The overlap that remains is measured, and it is the headline.** The point forecast's own
MAE differs between the two validation halves by:

| horizon | CAL_FIT | CAL_CONF | test | fit vs conform |
|---|---|---|---|---|
| h=1 | 0.4510 | 0.4154 | 0.3983 | **+8.576%** |
| h=4 | 0.8000 | 0.8058 | 0.7917 | -0.721% |
| h=96 | 1.3816 | 1.5528 | 1.5399 | **-11.025%** |

## 4. Methods, and the one idea that separates them

Every symmetric method is `point ± multiplier × scale(row)`.

| Method | `multiplier` from | `scale` | published at |
|---|---|---|---|
| `global_residual` | empirical quantile of \|residual\| | constant | no (fails 11/12) |
| `conformal_global` | `ceil((n+1)(1-α))`-th smallest score | constant | no |
| `state_residual` | empirical quantile of \|residual\|/scale | learned | no (fails 50% level) |
| `conformal_state` | conformal quantile of \|residual\|/scale | learned | **h=1, h=4** |
| `conformal_dispersion` | conformal quantile of \|residual\|/spread | expert spread | **h=96** |
| `quantile_regression` | pinball-loss model of the residual quantiles | asymmetric | no |

The learned scale fits `E[|residual| | x]` with an **L1** objective - the conditional median,
not the mean, because the absolute residual's right tail would otherwise inflate every row's
width.

## 5. Results on identical rows

Coverage at four levels on the sealed test split (`*` = outside the 0.0100 tolerance):

| method | h=1 @50/80/90/95 | h=4 | h=96 |
|---|---|---|---|
| `global_residual` | 0.6993\* 0.9049\* 0.9604\* 0.9836\* | 0.4717\* 0.7937 0.9121\* 0.9653\* | 0.3586\* 0.6825\* 0.8211\* 0.9071\* |
| `conformal_global` | 0.5065 0.8047 0.9038 0.9534 | 0.5030 0.7925 0.8976 0.9516 | 0.5076 0.7890\* 0.9012 0.9538 |
| `state_residual` | 0.5178\* 0.7795\* 0.8692\* 0.9267\* | 0.4879\* 0.7902 0.9051 0.9561 | 0.5716\* 0.8347\* 0.9359\* 0.9728\* |
| `conformal_state` | 0.5270\* 0.8050 0.9020 0.9536 | 0.5159\* 0.7925 0.8973 0.9479 | 0.5368\* 0.8053 0.9118\* 0.9571 |
| `conformal_dispersion` | 0.4979 0.8014 0.8996 0.9494 | 0.4575\* 0.7782\* 0.8858\* 0.9401 | 0.4606\* 0.7853\* 0.8934 0.9453 |
| `quantile_regression` | 0.3803\* 0.7138\* 0.8391\* 0.9164\* | 0.3106\* 0.6556\* 0.7832\* 0.8640\* | 0.2300\* 0.4205\* 0.5780\* 0.7159\* |

Published procedure, chosen per horizon at the 90% band level from measured coverage:

| horizon | procedure | coverage @90% | error | ρ(width, error) | top decile |
|---|---|---|---|---|---|
| h=1 | `conformal_state` | 0.9020 | +0.0020 | +0.709 | 4.61x |
| h=4 | `conformal_state` | 0.8973 | -0.0027 | +0.673 | 4.21x |
| h=96 | `conformal_dispersion` | 0.8934 | -0.0066 | +0.658 | 4.80x |

## 6. Four findings that change Phase 9

1. **The default constant-width interval does not transfer across a difficulty shift.** It
   fails 11 of 12 cells, over-covering at h=1 and under-covering at h=96, and the sign of each
   failure matches the sign of the split-parity gap. No choice of quantile fixes it. Any
   future consumer that widens a forecast by a constant factor should be told this.
2. **Expert disagreement is real but weak, and it decays with horizon.** Spearman +0.41 /
   +0.25 / +0.18 at h=1 / 4 / 96. Normalising it by the forecast level makes it **negative** at
   h=4 and h=96 - the level is what the spread is about. As a width scale it is the best
   available choice at h=96 and produces the widest intervals in the study.
3. **Width tracks error in every regime at every horizon.** Across the load-level terciles the
   width ratio runs 7.8x / 7.8x / 16.0x against error ratios of 7.5x / 7.5x / 11.2x. The rows
   where the interval is narrow are the rows where being wrong costs 0.09 kW. This is the
   input a decision-assurance layer needs.
4. **Conditional calibration is not achieved.** Coverage is uniform *on average* and varies by
   up to 6pp across regime terciles (0.9160 for typical demand against 0.9730 for low, at
   h=1). The regime labels make this awkward to fix, because they are computed from the target
   being predicted.

## 7. Bugs this phase found in its own machinery

Recorded because each would have produced a plausible, wrong table.

| Defect | How it would have shown |
|---|---|
| `quantile_levels` returned one-sided tails while its docstring specified `α/2` | "nominal 90%" intervals covering 80%; a nominal 50% interval with zero width. Only quantile regression would have been silently wrong. |
| `_assemble_quantile_intervals` refilled one shared bounds array per level | Four nominal levels, four identical intervals, coverage repeating to six decimals. |
| The artifact published the *baseline* method | `calibration_status` reading `FAIL, FAIL, FAIL` on a published result. |
| The verdict rewarded the narrowest method regardless of calibration | `is_worth_the_machinery: YES` earned by an interval under-covering by 4pp. |
| Band cut points pooled across horizons | A band that reported the horizon, not the uncertainty. |
| The poisoning guard raised `IndexError` instead of naming a column mismatch | A leakage guard failing with a message that reads as a bug in the guard. |
| `interval_to_estimate` indexed a `[1, 1]` array as if flat | The domain join raised on every single-forecast conversion. |

Plus one Phase 2 defect: `UncertaintyEstimate` counted `lower_bound` and `upper_bound` as two
separate quantities and refused more than one, so **a prediction interval was the one
representation the domain could not hold**. Repaired as D-099.

## 8. Ablations

Weighted-interval-score improvement over the constant-width baseline:

| method | h=1 | h=4 | h=96 |
|---|---|---|---|
| `conformal_state` | +25.5% | +21.0% | +26.8% |
| `state_residual` | +26.9% | +20.7% | +27.0% |
| `quantile_regression` | +33.1% | +18.7% | -0.5% |
| `conformal_dispersion` | +19.0% | +17.2% | +17.3% |
| `conformal_global` | +6.2% | -0.6% | +7.2% |

The finite-sample conformal correction over a plain empirical quantile is worth 6.2% / -0.6% /
7.2% - **negligible at this sample size**, and the phase reports it as negligible rather than
claiming the correction matters. It would matter at n=100.

## 9. Blocked or out of scope

| Item | Status |
|---|---|
| **Coverage guarantee** | **Not claimed.** Conformal's guarantee is conditional on exchangeability; §3's own numbers show the assumption is violated. Every conformal interval carries the caveat in its metadata. |
| Time-ordered conformal | The recorded direction for earning a real guarantee. Not built. |
| Aleatoric / epistemic decomposition | Not claimed; see D-105. |
| Data-quality conditioning | **Impossible on this dataset** - SMART-DS carries no per-timestep quality flags, so the conditioning variable has zero variance. Reported, not simulated (D-106). |
| Battery dispatch | **Unavailable (G-01)** - unchanged from Phase 3. No flexibility modelling is possible from SMART-DS. |
| Per-node power balance | **Unavailable (G-02)** - Phase 11 must produce flows, not read them. |
| Decision rule | Deliberately absent (D-109). Phase 13 composes the interval with its own objective. |

## 10. Phase 9 requirements and recommendation

Phase 9 is flexibility modelling. The blocker is unchanged from Phase 3 and is a **data**
blocker, not a modelling one: SMART-DS supplies a single `kWhStored` nameplate scalar and no
state-of-charge series, so battery dispatch cannot be derived.

**Recommendation:** do not begin Phase 9 modelling until G-01 is resolved by data, because any
flexibility model built on this dataset would be fitted to a quantity that does not exist.
Phase 9's honest deliverable on the current dataset is the **representation only** - the
domain already has `FlexibilityEstimate`, `FlexibilitySource` and the uncertainty container -
plus an explicit, measured statement of what cannot be filled.

## 11. Critical context

1. **The point forecast must not be refitted.** Phase 8's uncertainty numbers describe Phase 7's
   fixed ensemble. A different forecast makes every interval in the phase a description of
   something else.
2. **The sealed test split has been read once per phase**, at the final evaluation, and nothing
   adjusts to what is found there. No gate, policy or blend was added after seeing a test
   result - that is why the shipped Phase 5 configuration is not the best the ablation found
   (D-077) and the shipped Phase 7 router still has a negative verdict.
3. **Exchangeability is violated on this data.** Any future claim of a coverage guarantee needs
   a time-ordered conformal method first.
4. **CPU-only torch.** The RTX 4050 has ~3.7 GB free, which is why Phase 6 could not attempt
   LoRA or full fine-tuning. Recorded as untested, not refuted.
5. **The SMART-DS licence is UNKNOWN.** It is fetched, never redistributed, and must be
   confirmed before any publication of the data itself.

## 12. Repository state

| Property | Value |
|---|---|
| Tracked files | 204 |
| Largest tracked file | `decisions.md`, 137 KB |
| Committed binaries | **none** |
| Committed dataset | **none** - `data/raw/*` is ignored |
| Committed artifacts | **none** - `artifacts/*`, `models/*`, `logs/*` ignored except `.gitkeep` |
| Secrets | **none.** No API key, token, private key or credential file is tracked, in the tree or in history. The project requires none at runtime. |
| Test suite | **1235 passed**, zero regressions against Phases 1-7 |
| Phase 7 decision records | D-091 … D-098 |
| Phase 8 decision records | **D-099 … D-109** |
| Design documents | `docs/expert_router_design.md`, `docs/uncertainty_design.md` |
| Registry | `experiments/registry.jsonl`, 29 records, project-relative artifact paths |

## 13. How to reproduce

```powershell
uv venv --python 3.12
uv sync --extra dev
uv run pytest                                     # 1235 tests, no dataset needed

# The dataset is fetched, never committed - see README "Data".
energy-intel data ingest && energy-intel data validate

energy-intel router experts                      # the expensive step (~40 min, CPU)
energy-intel router run
energy-intel uncertainty run                     # ~8 min, CPU
```

Phase 8 is deterministic for a fixed seed: `tests/ml/test_uncertainty_offline.py` asserts that
re-running produces an identical comparison table, selection and verdict.

## 14. Entry points

```text
energy-intel router config        resolved Phase 7 configuration
energy-intel router experts       refit the expert pool and cache forecasts (the slow step)
energy-intel router run           routing, ablations, sealed evaluation
energy-intel router experiments   the recorded Phase 7 result

energy-intel uncertainty config   resolved Phase 8 configuration
energy-intel uncertainty run      six interval methods, one sealed evaluation
energy-intel uncertainty experiments   the recorded Phase 8 result
```

---

## 15. License

Proprietary. All rights reserved. Recorded as a project fact, not interpreted - no legal claim
is made in this document.