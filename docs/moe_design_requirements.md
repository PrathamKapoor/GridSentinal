# Energy-MoE Design Requirements

What a Mixture-of-Experts model for energy demand would require, and what Phases 5, 6
and 7 actually measured. **This document has been superseded as the plan.** It was the
input to Phase 7; Phase 7 answered it, and the architecture it proposed was not built.

The short version is in §0. Sections 1-9 are the original measurement and reasoning, kept
because Phase 7 was designed against them and its results should be readable against them.

---

## 0. Outcome: the plan changed, and this is what replaced it

The original plan was a **Qwen-internal MoE**: sparse expert FFN layers added to
`Qwen3-1.7B-Base`, with a load-balancing loss and capacity factors, because Phase 6 measured
that Qwen's frozen readout is nearly rank-2 (participation ratio 2.1 of 2,048 dimensions,
§3). The reasoning was that the shared trunk needed more capacity.

**That architecture was not built.** Phase 6's own numbers said the experts were already
winning every regime: persistence in low-ramp periods, the GBM in both demand terciles, the
TCN at high ramp and at the longer horizons, and Qwen in *none* of them (§1). Building
experts inside a backbone that wins no regime, whose readout is rank-2, would have added
capacity where the measurements said the problem was elsewhere.

Phase 7 instead implemented a **heterogeneous Energy Expert Router** over the three models
that demonstrably work:

| | Planned (superseded) | Built |
|---|---|---|
| Experts | Sparse FFN layers inside Qwen3-1.7B | Persistence, Phase 4's HistGBM, Phase 5's TCN |
| Gate | Load-balancing loss over internal experts | Softmax over heterogeneous model forecasts |
| Shared trunk | Qwen's 1.72 B-parameter residual stream | A 1,097-parameter MLP |
| Motivation | The trunk's rank-2 readout | The best expert changes with horizon and regime |
| Result | not run | **PARTIALLY** - beats the best single expert at 2 of 3 horizons, a fixed weighted ensemble at **0 of 3** |

The full design, the leakage protections, and the measured result are in
`docs/expert_router_design.md`. Recorded in `decisions.md` D-091 and D-092.

**What carried over unchanged.** Every requirement in §6 that could apply still does, and
§9's minimum bar was the checklist the implementation was held to:

| §9 requirement | How Phase 7 met it |
|---|---|
| Gate inputs all observable at the origin | 24 columns in six named groups; a poisoning test over whole series histories rebuilds them from corrupted data and requires bit-identical output |
| No future-derived feature | The regime terciles of §1 are used **only** as diagnostics; `past_error` lags by exactly `h + 1` steps, hand-checked in a test |
| Leakage test automated, covering the gate's inputs | `assert_router_features_are_causal`, plus a test that injects a leaking builder and asserts the guard fires |
| Control on the same rows in the same run | Uniform, best-single, fixed-weight and oracle baselines, all built from one cached forecast block |
| Per-regime error against the best single model | Reported per regime; the router wins one of six and is **9.4x worse** than persistence in the low-ramp tercile |
| Budget stated in Phase 7's units | Router is 0.11% of expert inference and ~475,000x cheaper per row than Phase 6's frozen backbone. Compute was never the constraint |
| Fallback acceptable | **Taken, and it is the result.** The fixed weighted ensemble is recorded as the better architecture at every horizon |

The one requirement that became moot is §6.3, "evidence that the shared backbone is not the
bottleneck" - there is no shared backbone.

---

## 1. The precondition that is still true: measured regime heterogeneity

At the shortest horizon, kW MAE on the Phase 6 test subsample (10,000 rows, identical
for every model):

| Demand tercile | n | GBM | persistence | Phase 5 TCN | Qwen probe |
|---|---|---|---|---|---|
| low | 3,334 | **0.0834** | 0.0690 | 0.0800 | 0.1197 |
| typical | 3,333 | **0.2548** | 0.2626 | 0.2599 | 0.2985 |
| high | 3,333 | **1.1456** | **0.8541** | 0.8845 | 1.2176 |

| Ramp tercile | GBM | persistence | Phase 5 TCN | Qwen probe |
|---|---|---|---|---|
| low | 0.0803 | **0.0034** | 0.0343 | 0.1022 |
| typical | 0.2008 | 0.0602 | **0.1035** | 0.2132 |
| high | 1.2028 | 1.1220 | **1.0867** | 1.3203 |

The Phase 5 conclusions hold and are now measured twice:

* Error concentrates. The high demand tercile costs **14x** the low tercile and is
  roughly 73% of total error; the high ramp tercile is roughly 84%.
* **No single model wins everywhere.** Persistence is untouchable in low-ramp periods
  (0.0034 kW, an order of magnitude better than anything else). The Phase 5 TCN wins
  high-ramp periods. The GBM wins both demand terciles.
* Qwen is **never** the best model in any regime, including the difficult ones. It
  does not carve out a niche; it is uniformly fourth.

That last row matters for Phase 7 planning. A router over experts only pays if some
regime is better served by a specialist. On this evidence, the specialists to route
*between* are persistence, the GBM and the TCN - not Qwen variants.

**Phase 7 confirmed the diversity premise and refuted the exploitation of it.** On the
full test split the experts disagree on **59-75% of rows**, their error correlation falls from
0.90 at 15 minutes to 0.69 at 24 hours, and the oracle gain available over the best single
expert is **30-38%**. The pool is genuinely complementary.

The router did not exploit it. A fixed weighted ensemble is better at every horizon; the
router's only regime win is the high-ramp tercile; it assigned persistence a mean weight of
under 0.001 and is **9.4x worse than persistence in the low-ramp tercile** that §6.7 warned
about. Capture was -2% to 13% of the available headroom. **The single-model-plus-regime-router
fallback that §9 listed as acceptable is the outcome.**

---

## 2. The precondition that is NOT met: a deployable gate

**The terciles above are computed from the demand being predicted.** They are
diagnostics, not a routing signal. At forecast time the future demand level and ramp
are unknown by construction, so a router cannot use them. Any design that routes on
them has leaked the target, and Phase 4's leakage guard would correctly reject it.

A deployable gate must run on features observable **at or before the forecast origin**:

| Candidate gate feature | Observable? | Note |
|---|---|---|
| Trailing demand level over recent hours | yes | the natural proxy for the demand tercile |
| Trailing ramp magnitude | yes | the natural proxy for the ramp tercile |
| Hour of day, day of week | yes | but see §4: Phase 6 found the cyclic channels actively harmful |
| Predicted level for the target horizon | only if a level model runs first | implies a two-stage design |
| **Future demand level or ramp** | **no** | that is the target |
| **Future weather** | **no** | not in this dataset at all |

Phase 7 must state which of these its gate uses and demonstrate that every gate input
is available at the origin. Building a *usable* regime proxy means labelling regions
by a trailing-window rule and accepting that the proxy is imperfect; Phase 7 must
report how well any proxy separates the terciles in §1 rather than assuming it does.

---

## 3. The new finding: the representation is the bottleneck, not the routing

Phase 6's central measurement is that a frozen Qwen3-1.7B backbone hands its head an
almost rank-2 signal for this task.

Participation ratio of the covariance spectrum, `(sum lambda)^2 / sum lambda^2`, where
the effective number of active dimensions is the value itself. Measured on the final
layer's last-token state:

| Backbone | Layer 8 | Layer 16 | Layer 24 | Layer 28 |
|---|---|---|---|---|
| Qwen3-1.7B pretrained | 1.0 | 1.0 | 1.1 | **2.1** |
| random, same architecture | 7.1 | 8.4 | 9.6 | **10.0** |

Read carefully, because it cuts two ways:

* The pretrained residual stream is **far more collapsed** than a random one at every
  depth - participation 1.0 at layers 8 and 16 means those states are effectively a
  single direction. Whatever Qwen's final state computes for a numeric window, it is
  not 2048-dimensional structure.
* Pretraining nonetheless **helps**: on identical rows with an identical probe,
  pretrained beats random by 33.9% at h=1 and 22.9% at h=96, and ties at h=4. So
  there is real transfer in there. It is simply not enough to reach the baselines.

Two further measurements support the same reading:

* **Layer choice matters enormously.** Reading layer 16 instead of layer 28 gives an
  MAE of 23.33 kW at h=1 against 1.33 kW at layer 28 - a 17x degradation from choosing
  the wrong depth. Whatever survives to the final layer is not what an intermediate
  layer holds.
* **The linear probe was under-parameterised.** A two-layer head cut h=1 error from
  1.3324 to 0.5452 kW. The best Qwen configuration is therefore **not** the shipped
  linear one, and even that best arm still loses to the GBM by 10.2% at h=1 and by
  65-70% at the longer horizons.

**Implication for Phase 7.** Structural modification of Qwen is being considered on
the premise that its dense form is a good representation that merely needs
specialisation. Phase 6 measured that premise and did not support it. Before adding
experts, the honest question is whether the *shared* backbone can be made to carry more
dimensional structure for numeric windows - not whether to route around a collapsed
one. An MoE over a rank-2 shared trunk inherits the collapse.

---

## 4. What the cyclic-channel ablation settles

Phase 5 found the sin/cos hour and day-of-year channels were not earning their place
under a convolution, and kept them anyway rather than tune against the test set
(D-077). Phase 6 re-tested them under a transformer, where the motivation is stronger:
Qwen's RoPE encodes only **relative** position, so absolute time-of-day has to come
from the input channels.

Measured, identical rows, identical probe:

| Horizon | with cyclic | without cyclic | effect of removing |
|---|---|---|---|
| h=1 | 1.3324 | **0.8432** | **36.7% better** |
| h=4 | 1.8264 | **1.2174** | **33.3% better** |
| h=96 | 2.9956 | **3.0695** | 2.5% worse |

Removing them helps substantially at the two short horizons even under RoPE, and costs
almost nothing at 24 hours. The Phase 5 finding **transfers across architecture**:
absolute-time channels are not the missing ingredient, and on a transformer they are
mildly harmful at short range. A Phase 7 expert should not assume a cyclic channel is
required just because the backbone uses RoPE.

---

## 5. Candidate architectures

| Design | Assessment |
|---|---|
| **Horizon experts** (one per horizon) | Weak. Horizons are already separate heads over a shared trunk, and Phase 5's `single_horizon` ablation showed sharing *helps* by 38.5%. Splitting them discards a measured win. |
| **Regime experts** (stable vs ramp, low vs high) | The best-supported routing option, conditional on §2's gate problem being solved. But note the specialists that currently win are persistence, GBM and TCN - not Qwen variants. |
| **Lookback experts** (1 day vs 1 week) | Supported by Phase 5's measurement that context length helps monotonically. Increases memory sharply, since each expert needs its own window. |
| **Model-family experts** (TCN vs GBM vs persistence) | Effectively a learned stacking ensemble. Not an MoE: the gate would select among different model classes. Measured to be worth building - §1 shows the best model is regime-dependent. |
| **Per-customer experts** | Rejected. Error concentrates by *regime*, not by customer size; per-customer experts would multiply parameters by 40 for no measured gain. |
| **Qwen experts over a shared Qwen trunk** | **Not supported by Phase 6.** The shared trunk's readout is near rank-2 (§3), and Qwen wins no regime (§1). Specialising a collapsed representation does not obviously produce a better one. |

---

## 6. Requirements any Phase 7 proposal must meet

1. **A gate over observable features only**, with each feature's availability at the
   forecast origin stated explicitly (§2).
2. **A leakage test** covering the gate's inputs, not just the sequence window.
3. **Evidence that the shared backbone is not the bottleneck.** Phase 6 measured a
   participation ratio of 2.1 at the readout. A proposal that adds experts while the
   shared trunk stays at that rank must say why more capacity helps.
4. **Load-balancing loss**, chosen and justified. Without it an MoE collapses onto one
   expert and becomes a single slow model.
5. **Capacity factors reported per expert.** An MoE whose experts each handle 3% of rows
   has multiplied parameters without buying coverage.
6. **A no-MoE control at matched budget**, measured on the same rows in the same run.
7. **Per-regime error against the best single model**, not only against the average. The
   low-ramp regime, where persistence reaches 0.0034 kW, is the one a router is most
   likely to damage.
8. **Compute accounting.** Phase 6's measured cost: 0.248 s/sample for a frozen
   1.7B-parameter forward pass at 21 tokens on CPU. A multi-expert forward pass is
   several times that.

---

## 7. Compute reality

| Measurement | Value |
|---|---|
| GPU | RTX 4050 laptop, 6,141 MiB total, ~2,400 MiB already used by the desktop |
| Torch build installed | **CPU-only** (`torch 2.14.1+cpu`), CUDA unavailable |
| Frozen forward pass, 21 tokens | **0.248 s/sample** measured over 25,000 rows |
| bf16 vs fp32 on this CPU | bf16 **4.8x slower** (emulated, no native bf16 in AVX2) |
| Full Phase 5 test population | ~11.8 h of forward passes at 21 tokens |
| Feature extraction actually run | 3 arms x 25,000 rows = **5.4 h** |
| Head training (34,627 params) | **5.6 s** |

The head is free; the backbone is everything. Any Phase 7 design should be costed as a
multiple of the frozen forward pass, and should expect that a trainable backbone - let
alone an MoE forward pass - is out of reach on this machine at the full population.

---

## 8. What is deliberately NOT claimed

* Not that an MoE would beat the TCN or the GBM. §1 and §3 give reasons to doubt it.
* Not that the regime terciles are available at inference. §2 shows they are not.
* Not that ablation budgets support absolute comparisons. Phase 6's ablations share one
  25,000-row subsample and one extraction pass, so comparisons **within** them are
  valid; absolute values are subsample values and are not the published Phase 4/5
  figures.
* Not that a 2-layer head is the best head. It beat the linear probe by a lot, which
  means the probe search was shallow, not that it has converged.
* Not that patch granularity has been settled. `patch_size = 1` was not run: at 168
  tokens it would cost roughly 16 h per extraction pass.
* Not that a trainable backbone would not help. Phase 6 measured a *frozen* one. Full
  or partial unfreezing was costed, not run, and remains untested.

---

## 9. Minimum bar for starting Phase 7

*Superseded - this is the bar Phase 7 was held to, and §0 records how each item was met.
Kept unchanged so the checklist and its outcome stay together.*

| Criterion | Requirement |
|---|---|
| Gate inputs | Every one listed with its availability at the origin; no future-derived feature |
| Gate quality | The trailing-window proxy's separation of the §1 terciles measured and reported, before any MoE is trained |
| Leakage test | Automated, covering the gate's inputs |
| Representation check | The shared trunk's participation ratio re-measured, with an argument for why added capacity beats a better readout |
| Control | TCN and GBM on the same rows in the same run |
| Budget | Stated in the units of §7 |
| Fallback | "One model, plus a regime router over persistence/GBM/TCN, is enough" is a recorded, acceptable outcome |

Two items resolved differently than the table anticipates, and both are findings rather
than omissions. **Gate quality** was measured by the §7 regime tables in
`docs/expert_router_design.md` §7.7, which show the router beating the best single expert in
the high-demand tercile (0.8808 against 0.9197 kW) while losing badly in the low-ramp
tercile (0.0188 against 0.0051 kW) - the trailing-window proxy separates the terciles
adequately where the error is concentrated and badly where persistence is already exact.
**Representation check** became moot when the shared trunk was removed; the premise it
questioned, whether the frozen backbone could carry more dimensional structure, was never
adopted.
