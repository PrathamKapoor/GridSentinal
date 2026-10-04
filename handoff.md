# Handoff - Phase 9

**Project:** Energy Intelligence - adaptive, self-verifying energy management
**Event:** Yuva Yodha Energy Tech Hackathon 2026, Schneider Electric
**Current phase:** **Phase 9 of 20 - Uncertainty-aware flexibility estimation - COMPLETE**
**Previous phases:** Phases 1-8 complete and preserved, zero regressions
**Date:** 2026-10-04
**Author of all work:** PrathamKapoor <prathamkapoor027@gmail.com>

> Phase 9 completed and handed off. Phases 1-8 are preserved with zero regressions.

---

## 1. The answer, first

**Phase 9 verdict: NO.** A behavioural flexibility envelope was estimated, calibrated, and
evaluated once on a sealed split. **No physical flexibility is supported by this dataset**, so
no dispatchable capability is claimed.

| component | answer | what decided it |
|---|---|---|
| `physical_support` | **NO** | 0 of 8 flexibility dimensions carry evidence of a physical limit and a control interface |
| `can_be_estimated` | **PARTIALLY** | sealed-test coverage within 1.4pp of nominal at h=1, 1.1pp at h=4, 0.75pp at h=96 (tolerance 1.0pp) |
| `is_directional` | **PARTIALLY** | per-direction conditional coverage 0.882-0.910 against a nominal of 0.900 |
| `is_stable` | **YES** | width is constant within each customer; within-series lag-1 autocorrelation 1.0 |
| `uncertainty_is_informative` | **YES** | Spearman(Phase 8 width, realised deviation) = +0.393 / +0.246 / +0.312 |
| `aggregation_creates_capability` | **NO** | aggregating historical variation cannot create the ability to command anything |

The phase verdict is the weakest component, so the phase reports NO. That is the correct
outcome for an honest characterisation of this dataset, and it is what makes Phase 10's
prerequisites explicit rather than implied.

Every published figure is `basis=STATISTICAL_PROXY` with `authority=NOT_CONTROLLABLE`, and the
domain constructor **refuses** to build an object that says otherwise.

## 2. The capability audit, which is the real result

Run `energy-intel flexibility audit` - it prints this in about a second and fits nothing,
because the question deserves an answer that cannot be mistaken for a fitted result.

| dimension | basis | gap | what is missing |
|---|---|---|---|
| `battery_discharge` | UNKNOWN | G-01 | 93/93 `Storage` elements report `State=IDLING`, ratings only, one `kWhStored` scalar; no SOC series, no dispatch record |
| `battery_charge` | UNKNOWN | G-01 | as above |
| `ev_charging_shift` | UNKNOWN | G-07 | no EV charging, SOC or departure series |
| `hvac_setpoint` | UNKNOWN | G-08 | no thermostat, setpoint or indoor temperature |
| `flexible_load_shift` | UNKNOWN | G-09 | no interruptible or shiftable load identified |
| `demand_response` | UNKNOWN | G-10 | no DR programme, event log or enrolment field |
| `der_curtailment` | UNKNOWN | G-03 | the feeder's 1,216 declared PV systems are not the one measured 1,000 kW array |
| `grid_import_limit` | UNKNOWN | G-02 | no per-node power-flow time series exists |

Two facts constrain everything downstream: **consumption is not controllability** (customer
load is observed consumption, and nothing says any of it can be moved), and **losses are
unmodelled at 3.2175%** (G-06), so the 0.5 kW network balance tolerance cannot close.

## 3. The domain contract: how a claim cannot be smuggled in

`src/energy_intelligence/domain/flexibility.py` adds `FlexibilityDirection`,
`FlexibilityBasis`, `AggregationLevel`, `FlexibilityEstimate` and `FlexibilityEnvelope`. The
rules live in `__post_init__`, not in a docstring:

1. **A statistical proxy may not claim a control authority.** `STATISTICAL_PROXY` with
   `SCHEDULEABLE` raises. This is the one rule that stops "demand moved this much before"
   becoming "this much can be commanded".
2. **An unknown may not carry a magnitude.** `basis=UNKNOWN` with `value=0.0` raises, so "not
   known" can never be published as "none available".
3. **Every estimate is `role=DERIVED` and carries provenance.** Nothing here writes a
   measurement.

Direction semantics are pinned by the contract: `DOWNWARD` means net demand decreases;
`FLEXIBLE_LOAD_SHIFT` is a signed load setpoint so a downward 5 kW estimate gives
`as_signed_setpoint() == -5.0`; `NodePower` treats load as negative so the same estimate gives
`as_node_power_delta() == +5.0`. Getting this backwards would let a caller shed load by
increasing it.

## 4. What was estimated, and on which rows

The panel carries targets only at `t+1`, `t+4` and `t+96`, which cannot characterise a day, so
the full series was rebuilt from the per-unit array and the row's rated kW.

- **Reconstruction verified exactly**: max absolute gap **0.0 kW over all 359,040 rows**. The
  run stops unless it reproduces the panel.
- **Chronological partition**: `CAL_FIT` 89,760 rows fits, `CAL_CONF` 89,760 rows selects and
  calibrates, sealed test 179,520 rows **read once**.
- **Split parity reported, not assumed**: fixed-ensemble MAE 0.451/0.800/1.382 kW on the fit
  half against 0.415/0.806/1.553 kW on the conformity half.

## 5. Four configurations, selected on held-out calibration data

Fitted on `CAL_FIT`, evaluated on `CAL_CONF`, selected by lowest weighted interval score among
configurations inside the coverage tolerance - width alone would reward a band that covers
nothing, and uncalibrated configurations rank last.

**`persistence` at the flat `horizon` granularity won.** The calendar variants covered
0.43-0.66. Instructively `calendar|horizon` had a band **4.4x wider** than the selected one and
covered **less** (0.6579 vs 0.9150): a mis-centred band is worse than a well-centred narrow one,
because conditioning on a calendar slot while centring on a pooled median puts a wide band
around the wrong number. That is a reason to distrust naive slot pooling, not to distrust
calendar conditioning in general - the finer granularities were also the thinnest cells.

## 6. Results on identical rows

| horizon | coverage | error | mean width kW | width/MAE | WIS kW | up cov | down cov | calibrated |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.9140 | +0.0140 | 2.7258 | 6.742 | 3.9853 | 0.9079 | 0.9048 | no |
| 4 | 0.9110 | +0.0110 | 5.5658 | 6.256 | 7.7263 | 0.9103 | 0.9101 | no |
| 96 | 0.8925 | -0.0075 | 14.1165 | 5.799 | 18.0102 | 0.8969 | 0.8819 | **yes** |

Conformal scales solved on `CAL_CONF` and applied unchanged to test: 0.9184 / 0.9063 / 0.9572,
taking conformity coverage 0.9150 / 0.9237 / 0.9109 to exactly 0.9000.

**The band is about six times wider than the mean absolute deviation it bounds.** That is not a
defect - it is what 90% two-sided coverage costs on near-symmetric heavy-tailed deviations -
but it means the envelope is a *containment* statement, not a planning resolution. Read "within
±6 kW" as "we were surprised outside ±6 kW about 10% of the time", not "we can move 6 kW".

## 7. Five findings that change Phase 10

1. **No physical flexibility exists here.** Phase 10 cannot optimise against a proxy; it needs
   battery energy *and* power, per-node limits with losses modelled, EV windows, HVAC
   setpoints, an identified flexible load, and a DR programme - all from an external source.
2. **Pooling horizons destroys calibration.** The first implementation returned the same
   4.1622 kW band at every horizon, with coverage 0.949 / 0.853 / 0.700. Cells must be fitted
   per horizon.
3. **A two-sided 90% band needs `alpha/2` per tail.** Using the total miss probability as the
   per-tail quantile silently produces an **80%** band labelled 90%.
4. **A quantile band does not follow the evaluation period.** The fitted band over-covered by
   +2.71pp and +3.39pp at h=1 and h=4 - about 47 binomial standard errors. A conformal scale on
   the conformity split fixes most of it; the residual is reported, not tuned away.
5. **Uncertainty does predict flexibility here, and the discount is still withheld.** The
   coupling is *supported* (+0.393/+0.246/+0.312) but was computed and judged post hoc on the
   sealed split, so applying it would reuse test information to build the quantity under
   evaluation. It is worth 12-31% of band width - a change in meaning, not a refinement.

## 8. Aggregation: measured, explicitly not capability

| horizon | aggregate up kW | naive sum kW | diversification | mean pairwise corr | aggregate coverage |
|---|---|---|---|---|---|
| 1 | 0.527 | 57.713 | 0.9909 | -0.0000 | 0.8212 |
| 4 | 1.351 | 115.491 | 0.9883 | -0.0000 | 0.8353 |
| 96 | 1.837 | 271.691 | 0.9932 | -0.0000 | 0.7894 |

Diversification of 99% is a statement about **independence, not headroom**. These customers
move almost independently, so summing individual bands gives a fleet figure ~100x the pooled
band. That is a real property of the data and it is **not** evidence the fleet can move
271 kW. Near-zero correlation is why pooling is defensible for a containment statement *and*
why it is not a capability. `AggregationResult.describe()` carries a `warning` string so a
consumer reading only the aggregate still sees this.

## 9. Bugs this phase found in its own machinery

Each was found by a measurement, not by review, and each is pinned by a named regression test
in `tests/ml/test_flexibility_offline.py`.

| defect | how it was caught | consequence had it shipped |
|---|---|---|
| calendar baseline indexed by panel row position instead of series index | raised while indexing a 40-series matrix with values up to 89,760 | crash, or a clipped-index wrong number |
| fixed-ensemble weight matrix transposed in the einsum | Phase 7 MAE parity check, run before estimation | plausible forecast from transposed weights |
| cells pooled across horizons | identical 4.1622 kW width at all three horizons | h=1 over-covers by 5pp, unfixable downstream |
| total miss probability used as the per-tail quantile | in-sample coverage 0.6485 against a nominal 0.90 | an 80% band labelled 90% |
| stability lag-1 taken across the customer boundary | autocorrelation 0.16 for a band that is constant per customer | `is_stable` failed on a perfectly smooth band |
| domain layer imported numpy | `tests/test_package.py` architecture invariant | plain-Python consumers locked out of the domain |
| fitted band not corrected for the period shift | coverage +2.71pp / +3.39pp at 47 standard errors | a published figure whose stated level is wrong |

## 10. Blocked or out of scope

- **Physical flexibility, dispatchable capability, guaranteed demand response:** not claimed.
  Each is refused by the domain contract, not merely omitted from the report.
- **Optimisation:** out of scope. Nothing here solves a dispatch problem; Phase 10 needs
  physical limits this dataset does not contain.
- **The reliance discount:** measured and published as evidence, not applied.
- **Regime-conditioned envelopes:** evaluated per Phase 4/5 tercile but **not fitted**. The
  terciles are computed from the quantity being predicted, so conditioning on them would be
  leakage.
- **A properly conditioned calendar reference:** the loser here was mis-centred, not
  disproven. Worth trying with more calibration data; it needs its own measurement.

## 11. Phase 10 requirements and recommendation

Phase 10 needs, from an external source, before it can optimise anything:

1. battery usable energy **and** rated power per site (G-01),
2. per-node power-flow limits with losses modelled (G-02, G-06),
3. an EV fleet with charging windows, SOC and departure times (G-07),
4. thermostat setpoints and indoor temperature for any HVAC flexibility (G-08),
5. an identified interruptible or shiftable load with its limits (G-09),
6. a demand-response programme with enrolment and event history (G-10).

**Recommendation:** Phase 10 should treat Phase 9's envelope as a *demand-distribution
description* it can validate against, and source its capability from wherever items 1-6 exist.
Where a physical limit is available it must carry `basis=PHYSICAL` and a real authority; where
it is not, the proxy must stay `NOT_CONTROLLABLE`. The scenario entry point exists for
demonstrating the control workflow without contaminating a measurement.

## 12. Critical context

- **The sealed test split was read once.** The reliance coupling was computed post hoc on it
  and therefore withheld; the conformal scale was solved on `CAL_CONF` and applied unchanged.
- **Phase 8's rows are proved, not assumed.** The run refuses to start unless the rebuilt
  ensemble reproduces Phase 7's published MAE (measured max relative difference **0.0%**) and
  Phase 8's artifact point forecast matches the rebuilt one on test to 1e-9 kW. Matching row
  *counts* would not catch a permutation, and a permutation would pair every interval with the
  wrong deviation.
- **The run is deterministic** - calendar medians, empirical order statistics and a
  bisection, no sampling. `ExperimentRecord.seed` is recorded as `None` rather than a number
  that was never used.
- **Absolute kW is meaningful per row only.** The group array is not a fleet total.
- **The domain layer imports no numeric library**, enforced by `tests/test_package.py`.

## 13. Repository state

- Branch `main`, remote `origin` = `https://github.com/PrathamKapoor/GridSentinal.git`.
- Full suite: **1290 passed, 1 failed** - the failure is
  `test_the_real_backbone_matches_every_architecture_invariant`, an `OSError 1455` (Windows
  pagefile exhaustion) loading a large Qwen checkpoint under memory pressure. It **passes in
  isolation** (21 passed) and is environmental, not a regression.
- Phase 9 adds 56 tests (29 ML, 27 config) covering the contract, the estimators and every
  regression listed in §9.
- Artifacts under `artifacts/phase9/flexibility-main/`; one registry record appended to
  `experiments/registry.jsonl`.
- No credentials, caches, raw data or virtual environments are tracked.

## 14. How to reproduce

```bash
energy-intel flexibility audit          # capability table, fits nothing, ~1s
energy-intel flexibility config         # resolved configuration
energy-intel flexibility run            # the full phase, ~7 minutes
energy-intel flexibility experiments   # the recorded summary
```

Inputs, all read and never regenerated: `artifacts/phase7/experts.npz` (panel geometry, rated
kW, origins, per-expert forecasts), `artifacts/phase7/router-main/result.json` (the recorded
weights), `artifacts/phase8/uncertainty-main/probabilistic_forecasts.npz` (published interval
widths), `configs/flexibility.toml`.

Outputs: `result.json`, `summary.md`, `capability.json`, `flexibility_envelope.npz`,
`flexibility_envelope.json`, `run.log`.

Full suite: `.venv\Scripts\python.exe -m pytest -q --no-header -p no:cacheprovider`

## 15. Entry points

| file | role |
|---|---|
| `src/energy_intelligence/domain/flexibility.py` | `FlexibilityEstimate`, `FlexibilityEnvelope`, the authority rules |
| `src/energy_intelligence/domain/enums.py` | `FlexibilityDirection`, `FlexibilityBasis` |
| `src/energy_intelligence/ml/flexibility/demand.py` | demand reconstruction and calendar slots |
| `src/energy_intelligence/ml/flexibility/baselines.py` | calendar and persistence baselines |
| `src/energy_intelligence/ml/flexibility/envelope.py` | per-horizon fit, `conformal_scale` |
| `src/energy_intelligence/ml/flexibility/reliance.py` | causal coupling and its justification |
| `src/energy_intelligence/ml/flexibility/capability.py` | the capability audit |
| `src/energy_intelligence/ml/flexibility/aggregation.py` | pooled envelope beside the naive sum |
| `src/energy_intelligence/ml/flexibility/scenarios.py` | quarantined assumption mode |
| `src/energy_intelligence/ml/flexibility/offline/` | analysis, runner and verdict |
| `src/energy_intelligence/config/flexibility.py` | config and validation |
| `configs/flexibility.toml` | inherited task, validated not trusted |
| `docs/flexibility_design.md` | full design, results and reasoning |
| `decisions.md` | D-110 through D-120 |

## 16. License

No `LICENSE` file exists in this repository. That remains an open item carried forward from
earlier phases and should be settled before any external distribution.