# Phase 9 - Uncertainty-aware flexibility estimation

**Verdict: NO.** A behavioural envelope was estimated, calibrated, and evaluated once on
held-out data. **No physical flexibility is supported by this dataset**, so nothing
dispatchable is claimed, and Phase 10 cannot plan against this phase's output without
obtaining physical limits from an external source.

Run it with `energy-intel flexibility run`. Reproduce the whole phase with
`energy-intel flexibility audit && energy-intel flexibility run`.

---

## 1. The question, and the honest answer

"Can a flexibility envelope be estimated from this data, is it usable and directional, and is
any flexibility physically supported?"

The three halves of that come out very differently, and reporting them as one number would be
the first dishonest thing this document could do.

| component | answer | what decided it |
|---|---|---|
| `physical_support` | **NO** | 0 of 8 flexibility dimensions carry evidence of a physical limit and a control interface in SMART-DS's own metadata |
| `can_be_estimated` | **PARTIALLY** | sealed-test coverage within 1.4pp of nominal at h=1, 1.1pp at h=4, 0.75pp at h=96, against a 1.0pp tolerance |
| `is_directional` | **PARTIALLY** | per-direction conditional coverage 0.882-0.910 against a nominal of 0.900; h=1 and h=4 miss the tolerance in the same direction the overall band does |
| `is_stable` | **YES** | the published band's width is constant within each customer, so its within-series lag-1 autocorrelation is 1.0 |
| `uncertainty_is_informative` | **YES** | Spearman(Phase 8 width, realised deviation) = +0.393 / +0.246 / +0.312 |
| `aggregation_creates_capability` | **NO** | not a measurement - aggregating historical variation cannot create the ability to command anything |

The phase verdict is the **weakest** component, so the phase reports NO. That is the honest
outcome and it is a successful phase: the deliverable was an honest characterisation of what
this data can and cannot support, and the characterisation is now measured rather than
assumed.

---

## 2. The capability audit comes first, and it is an output

Before estimating anything statistical, the phase asks which flexibility dimensions could
*ever* be physical on this dataset. On SMART-DS the answer is none, for all eight dimensions,
and each answer points at a documented gap:

| dimension | basis | gap | what is missing |
|---|---|---|---|
| `battery_discharge` | UNKNOWN | G-01 | 93/93 `Storage` elements report `State=IDLING`, ratings only, one `kWhStored` scalar; no SOC series and no dispatch record |
| `battery_charge` | UNKNOWN | G-01 | as above |
| `ev_charging_shift` | UNKNOWN | G-07 | no EV charging, SOC or departure series exists |
| `hvac_setpoint` | UNKNOWN | G-08 | no thermostat, setpoint or indoor-temperature series |
| `flexible_load_shift` | UNKNOWN | G-09 | no interruptible or shiftable load is identified anywhere in the metadata |
| `demand_response` | UNKNOWN | G-10 | no demand-response programme, event log or enrolment field exists |
| `der_curtailment` | UNKNOWN | G-03 | the feeder's 1,216 declared PV systems are not the one measured 1,000 kW array |
| `grid_import_limit` | UNKNOWN | G-02 | no per-node power-flow time series exists, so no limit can be evaluated |

Two further facts constrain everything downstream:

- **Consumption is not controllability.** Customer load is observed *consumption*. Nothing in
  the dataset says any of it can be moved.
- **Losses are unmodelled at 3.2175%**, so the 0.5 kW network balance tolerance cannot close,
  and no feeder-level limit can be derived from the aggregate even in principle (G-06).

`energy-intel flexibility audit` prints this table in about a second and fits nothing, because
the question deserves an answer that cannot be mistaken for a result.

---

## 3. The domain contract: how a claim cannot be smuggled in

`src/energy_intelligence/domain/flexibility.py` adds two enums and two objects, and the
rules live in `__post_init__` rather than in documentation.

`FlexibilityBasis` is the load-bearing addition:

| basis | meaning | may claim a control authority? |
|---|---|---|
| `PHYSICAL` | measured from an asset's own limit and control interface | yes |
| `STATISTICAL_PROXY` | derived from observed behavioural variation | **no - refused by the constructor** |
| `ASSUMED` | supplied for a demonstration scenario | yes, at advisory authority only |
| `UNKNOWN` | not established | **no - may not carry a magnitude at all** |

Three rules do the work:

1. **A statistical proxy cannot claim a control authority.** Constructing a
   `STATISTICAL_PROXY` estimate with `SCHEDULEABLE` raises `DomainValidationError`. This is
   the one rule that stops a caller turning "demand has moved this much before" into "this
   much can be commanded".
2. **An unknown may not carry a magnitude.** `basis=UNKNOWN` with `value=0.0` raises, so
   "not known" can never be published as "none available".
3. **Every estimate is `role=DERIVED` and carries provenance.** Nothing in this phase writes a
   measurement.

Direction semantics are pinned by the contract rather than by a comment, because getting them
backwards would let a caller shed load by increasing it:

- `DOWNWARD` means net demand **decreases**; `UPWARD` means it **increases**.
- `FLEXIBLE_LOAD_SHIFT` is a signed load setpoint, so `as_signed_setpoint()` on a downward
  5 kW estimate returns `-5.0`.
- `NodePower` treats load as negative, so the same estimate's `as_node_power_delta()` returns
  `+5.0`.

`FlexibilityEnvelope` is the block form: per-row, per-horizon magnitudes with one basis and
one authority for the whole block. Its `to_dict()` serialises `is_dispatchable` explicitly, so
a deserialised envelope cannot recover a control authority it was never allowed to hold.

The domain layer imports no numeric library. `tests/test_package.py` enforces that, and it
caught a real violation during this build: the magnitudes are read by a small pure-Python
structural reader (`_as_float_grid`) rather than by coercing through numpy.

---

## 4. Why the estimate is a *behavioural* envelope

The only thing this dataset supports is a statement about behaviour:

> how far has this customer's demand historically moved from its own expected profile, at this
> time of day, at this horizon?

That is a real, useful, measurable quantity. It is not flexibility in the sense a planner
needs, and the artifact says so in its `limitations`, its `basis`, and its `authority`.

**Demand reconstruction.** The panel carries targets only at `t+1`, `t+4` and `t+96`, which
cannot characterise a day. The full series is rebuilt from the per-unit array and the row's
rated kW, then checked against the targets the panel *does* carry. The check is exact: max
absolute gap **0.0 kW over all 359,040 rows**. A reconstruction that was merely close would put
a systematic error into every deviation and therefore into every published magnitude, so the
run stops unless it reproduces the panel exactly.

**Chronological partition.** `CAL_FIT` (89,760 rows) fits the baseline and the envelope,
`CAL_CONF` (89,760 rows) selects the configuration and solves the calibration scale, and the
sealed test split (179,520 rows) is read **once**. The split-parity table is reported rather
than assumed - the fixed ensemble's MAE is 0.451/0.800/1.382 kW on the fit half and
0.415/0.806/1.553 kW on the conformity half, so the halves are close enough to be
interchangeable and the sizes of the differences are in the artifact.

---

## 5. Three estimator decisions that were wrong first

Each of these was a real defect found by a measurement, not by review, and each is now pinned
by a named regression test in `tests/ml/test_flexibility_offline.py`.

### 5.1 The band must be fitted per horizon

The first implementation pooled every horizon's deviations into one cell per
`(series, slot)` and took a single quantile, on the reasoning that pooling buys sample size.
That produced **the same 4.1622 kW band at h=1, h=4 and h=96**, with coverage 0.949 / 0.853 /
0.700. Deviation at h=96 is roughly four times deviation at h=1, so a shared cell must be
wide enough for h=96, and the h=1 band over-covered by five points. No downstream calibration
can fix a band that is ten times too wide at one horizon because it was fitted at another.

Pooling is fine for the *baseline* - a time-of-day profile is the same profile at every
horizon - and wrong for the *magnitude*. Cells are now fitted per horizon, and widths came
out 2.73 / 5.57 / 14.12 kW.

### 5.2 The miss probability must be split between the tails

The quantile was taken as `np.quantile(sample, 1 - alpha)` upward and
`np.quantile(sample, alpha)` downward with `alpha = 1 - 0.90`. That is a 90th and a 10th
percentile, which is an **80%** band, labelled 90%. Measured in-sample coverage was 0.6485 -
lower still, because an interpolated empirical quantile from a handful of observations sits
*between* observed points and therefore under-covers.

The fix is two-part: `alpha / 2` per tail, and conformal order statistics
(`ceil((n+1)(1-tail))`) instead of interpolation, so a cell guarantees at least nominal
coverage however small it is and degrades to its sample extreme when it is too small to
promise it.

### 5.3 The fitted band needs a calibration scale

Even correctly fitted, the band over-covered the later split: **+2.71pp at h=1 and +3.39pp at
h=4** on 179,520 rows, roughly 47 binomial standard errors. That is a real shift between the
calibration period and the test period, not sampling error, and re-fitting the quantiles
cannot fix it.

This is what the second half of the calibration split is for. One symmetric scale per horizon
is solved on `CAL_CONF` - rows the band was not fitted on - and applied unchanged to test:

| horizon | scale | conformity coverage before | after | sealed-test coverage | error | calibrated |
|---|---|---|---|---|---|---|
| 1 | 0.9184 | 0.9150 | 0.9000 | 0.9140 | +0.0140 | no |
| 4 | 0.9063 | 0.9237 | 0.9000 | 0.9110 | +0.0110 | no |
| 96 | 0.9572 | 0.9109 | 0.9000 | 0.8925 | -0.0075 | **yes** |

The residual miss at h=1 and h=4 is one to four tenths of a percentage point beyond the
tolerance, and it is reported rather than tuned away: a phase that kept rescaling until the
test split passed would be fitting to the test split.

### 5.4 Two more, caught by checks rather than by inspection

- **The calendar baseline was indexed by the wrong axis.** `fit_baseline` passed panel *row
  positions* where the fitter expected *series indices*, indexing a 40-series demand matrix
  with values up to 89,760. It raised rather than returning a wrong number, which is the only
  reason it was caught at all.
- **The fixed-ensemble weight matrix was transposed.** It is built `[expert, horizon]` and was
  multiplied with Phase 8's `"he"` einsum. The product is a valid array of plausible numbers
  that reproduces nobody's forecast. It was caught only because the run refuses to start
  unless the rebuilt ensemble reproduces Phase 7's published MAE to 1e-6 - measured maximum
  relative difference **0.0%**.

---

## 6. What was selected, and what it says

Four configurations were fitted on `CAL_FIT` (`calendar` at three calendar granularities plus
`persistence`), each evaluated on `CAL_CONF`, and the lowest weighted interval score among
configurations inside the coverage tolerance was selected. The criterion is the weighted
interval score rather than width because width alone rewards a band that covers nothing.

**`persistence` at the flat `horizon` granularity won.** The calendar variants all
under-covered badly - 0.43 to 0.66 - and, instructively, `calendar|horizon` had a band 4.4
times *wider* than the selected one and still covered **less** (0.6579 against 0.9150). A
mis-centred band is worse than a well-centred narrow one: conditioning on a calendar slot
while centring on a pooled median produces a wide band around the wrong number.

That is the most useful methodological finding in the phase, and it is a reason to distrust
naive-summer-time-day pooling, not a reason to distrust calendar conditioning in general - the
finer granularities were thinnest (median ~11-23 observations per cell) and their bands were
narrower still, which on this dataset is indistinguishable from being less well estimated.

| horizon | coverage | error | mean width kW | width/MAE | WIS kW | up cov | down cov |
|---|---|---|---|---|---|---|---|
| 1 | 0.9140 | +0.0140 | 2.7258 | 6.742 | 3.9853 | 0.9079 | 0.9048 |
| 4 | 0.9110 | +0.0110 | 5.5658 | 6.256 | 7.7263 | 0.9103 | 0.9101 |
| 96 | 0.8925 | -0.0075 | 14.1165 | 5.799 | 18.0102 | 0.8969 | 0.8819 |

**The band is about six times wider than the mean absolute deviation it bounds.** That is not
a defect to be tuned away - it is what 90% two-sided coverage costs on data whose deviations
are close to symmetric and heavy-tailed - but it does mean the envelope is a *containment*
statement, not a planning resolution. Anyone using it should read "within ±6 kW" as "we were
surprised outside ±6 kW about 10% of the time", not as "we can move 6 kW".

---

## 7. The uncertainty coupling: measured, supported, and still withheld

The coupling discounts promised flexibility by predictive uncertainty:

```text
reliance = clip(causal_trailing_median(normalised_width) / normalised_width, 0.25, 1.0)
normalised_width = Phase 8 interval width / the row's rated kW
```

The reference is **strictly trailing**, never centred. A reference that included the current
row would make the discount a function of the answer. The floor is 0.25 rather than 0.0
because a floor of zero would let a very wide interval imply *zero* flexibility, which is a
claim about capability rather than about confidence.

The coupling is **supported** on this data - Phase 8's width genuinely predicts realised
deviation magnitude, which is a real finding:

| horizon | Spearman(width, abs deviation) | supported | width that withholding the discount would have removed |
|---|---|---|---|
| 1 | +0.3928 | yes | +30.88% |
| 4 | +0.2460 | yes | +12.77% |
| 96 | +0.3115 | yes | +22.80% |

**It is nevertheless not applied to the published envelope**, for two reasons, both recorded
as decisions rather than left to chance:

1. The factor was computed and tested **post hoc on the sealed split**. Applying it would
   reuse test information to construct the quantity being evaluated. The test split was read
   once, and this keeps that promise intact.
2. A discount worth 12-31% of the band's width is not a refinement. It would change what the
   envelope *means*, and doing that on the strength of one post-hoc correlation would be
   exactly the kind of inference this phase exists to refuse.

What the phase therefore publishes is the raw behavioural envelope, plus the measured
coupling as evidence that Phase 8's uncertainty does carry information about flexibility. A
later phase that wants to apply it must calibrate the factor on data disjoint from its
evaluation.

---

## 8. Aggregation: measured, and explicitly not capability

The group envelope is reported beside the **naive sum of individual envelopes**, with the
measured pairwise correlation that explains the gap:

| horizon | aggregate up kW | naive sum kW | diversification | mean pairwise corr | aggregate coverage |
|---|---|---|---|---|---|
| 1 | 0.527 | 57.713 | 0.9909 | -0.0000 | 0.8212 |
| 4 | 1.351 | 115.491 | 0.9883 | -0.0000 | 0.8353 |
| 96 | 1.837 | 271.691 | 0.9932 | -0.0000 | 0.7894 |

Two things are worth stating plainly, because the table invites the wrong reading:

- **Diversification of 99% is a statement about independence, not about headroom.** These
  customers' deviations are almost perfectly uncorrelated, so summing their individual bands
  produces a fleet figure roughly 100 times the pooled band. That is a real property of the
  data. It is not evidence that the fleet can move 271 kW, and the envelope carries
  `authority=NOT_CONTROLLABLE` and `basis=STATISTICAL_PROXY` for exactly that reason.
- **The pooled band covers worse than the individual ones** (0.79-0.84 against 0.89-0.91).
  Near-zero correlation is *why* pooling is defensible for a containment statement and *why*
  it is not a capability: the pooled number is small precisely because the members move
  independently, which is the same fact that makes commanding them all at once impossible.

`AggregationResult.describe()` carries a `warning` string for the same reason, so a consumer
who reads only the aggregate still sees it.

---

## 9. Scenario mode is quarantined

`ScenarioAssumptions` exists so that a future phase can demonstrate a control workflow. Its
values are **assumptions**, and they are kept out of measured results by construction:

- `assert_real_data_mode(scenario)` raises `ScenarioViolation` if a scenario object reaches a
  real-data code path. Every real-data entry point calls it.
- `assumed_battery_usable_kwh` and `assumed_battery_max_kw` must be supplied **together** -
  usable energy without a power limit, or the reverse, is not a battery. That validation lives
  in the scenario object itself, so a demo cannot invent a battery with unlimited power.
- Anything derived from a scenario is tagged `basis=ASSUMED`, which the domain contract
  permits to carry an advisory authority and nothing more.

No scenario values appear in this phase's results.

---

## 10. What this phase does not claim

- **No physical flexibility**, for any asset, on any dimension. The audit establishes it.
- **No dispatchable resource.** Historical variation is not the ability to be commanded.
- **No guaranteed demand response.** No demand-response programme exists in this dataset.
- **No aggregation benefit.** Diversification is measured historical correlation.
- **No uncertainty discount.** The coupling was measured, found supportive, and withheld.
- **No optimisation.** Nothing here solves a dispatch problem; that is Phase 10's job and it
  needs physical limits this dataset does not contain.

## 11. What Phase 10 must do first

Phase 10 cannot plan against a statistical proxy. Before it optimises anything it needs, from
an external source:

1. battery usable energy **and** rated power per site (G-01),
2. per-node power-flow limits with losses modelled (G-02, G-06),
3. an EV fleet with charging windows, SOC and departure times (G-07),
4. thermostat setpoints and indoor temperature for any HVAC flexibility (G-08),
5. an identified interruptible or shiftable load with its limits (G-09),
6. a demand-response programme with enrolment and event history (G-10).

Until those exist, any dispatch figure would be arithmetic performed on a number this phase
has shown to be a proxy for behaviour and nothing more.

---

## Reproducing

```bash
energy-intel flexibility audit      # the capability table, fitting nothing
energy-intel flexibility config     # the resolved experiment configuration
energy-intel flexibility run        # the full phase, ~7 minutes
energy-intel flexibility experiments   # the recorded summary
```

Inputs, all read and never regenerated:

| input | what it provides |
|---|---|
| `artifacts/phase7/experts.npz` | panel geometry, per-row rated kW, origins, per-expert forecasts |
| `artifacts/phase7/router-main/result.json` | the fixed-ensemble weights Phase 9's point forecast is rebuilt from |
| `artifacts/phase8/uncertainty-main/probabilistic_forecasts.npz` | the published interval widths the coupling consumes |
| `configs/flexibility.toml` | the task inherited from Phases 5/7/8, validated rather than trusted |

Outputs, under `artifacts/phase9/flexibility-main/`:

| file | contents |
|---|---|
| `result.json` | every number in this document |
| `summary.md` | the rendered report |
| `capability.json` | the audit |
| `flexibility_envelope.npz` | per-row per-horizon magnitudes, the reference band, and the fallback flags |
| `flexibility_envelope.json` | the envelope's contract fields, naming the arrays |
| `run.log` | the run's own trace |

The run is **deterministic**: calendar medians, empirical order statistics and a bisection,
with no sampling anywhere. `ExperimentRecord.seed` is recorded as `None` rather than a number
that was never used, because claiming a seed for a procedure that ignores it is a small lie
about reproducibility.