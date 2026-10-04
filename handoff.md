# Handoff — Phase 6

**Project:** Energy Intelligence — adaptive, self-verifying energy management
**Event:** Yuva Yodha Energy Tech Hackathon 2026, Schneider Electric
**Current phase:** **Phase 6 of 20 — Qwen3-1.7B-Base energy specialization — COMPLETE (negative result)**
**Previous phases:** Phases 1–5 complete and preserved, zero regressions
**Date:** 2026-10-04
**Author of all work:** PrathamKapoor <prathamkapoor027@gmail.com>

---

## 1. The answer, first

> Can a pretrained Qwen3-1.7B checkpoint be meaningfully adapted to the real SMART-DS
> energy forecasting problem established in earlier phases?

**No — and the negative result is the phase's product.**

A frozen Qwen3-1.7B trunk with a trained head is **worse than the classical gradient
boosting model at every horizon**, worse than Phase 5's temporal TCN at every horizon,
and worse than persistence at 15 minutes. Every comparison is significant under a paired
bootstrap on identical rows. Qwen wins **no** demand or ramp regime.

The bar was pre-registered in `docs/qwen_energy_requirements.md` §1 **before** any of
this ran: beat **1.5648 kW at 24 hours**. Qwen's best h=96 result is **2.7400 kW**.

What *is* real: the pretrained representation beats a randomly initialised backbone of
identical architecture by **33.9% at h=1** and **22.9% at h=96**. So there is genuine
transfer. It is simply nowhere near sufficient.

---

## 2. Checkpoint: exact identity and verification

Verified directly against the authoritative model repository, not from the audit's
summary.

| Property | Verified value |
|---|---|
| Model ID | `Qwen/Qwen3-1.7B-Base` |
| **Revision (pinned)** | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| Architecture | `Qwen3ForCausalLM`, `model_type: qwen3` |
| Layers / hidden | 28 / 2048 |
| Attention | 16 query heads, **8 KV heads** (GQA), head_dim 128 |
| FFN | 6144 (3× hidden), SiLU |
| Embedding | 151,936 rows, **tied** — no stored `lm_head` |
| Context | 32,768 positions |
| Positional | RoPE θ = 1e6, no scaling, **relative only** |
| Layer types | `full_attention` × 28 — uniform, no special layer |
| License | **Apache-2.0** (full text present) |
| Gated | **False** |
| Weights | one `model.safetensors`, **3,441,185,608 bytes** |
| Measured from the file | **310 tensors, all BF16, 1,720,574,976 parameters** |

**License:** matches the audit classification (Apache-2.0), so no stop condition was
triggered. Recorded, not interpreted — no legal claim is made here.

**Base-model status confirmed:** the id contains `Base` and no `Instruct`/`Chat`
identifier; no instruct, chat, community fine-tune, merged, quantized, LoRA or
distilled artifact was used.

Post-download verification (`energy-intel qwen verify`): files present, byte count exact,
`config.json` agrees with the pinned facts on ten fields, dtype BF16 throughout, tied
embeddings confirmed by the absence of an `lm_head` tensor, model loads (8.2 s), forward
pass produces `logits [2,168,151936]` and 29 hidden states of `[2,168,2048]`.

---

## 3. Compute, measured before anything was designed

| Measurement | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 (laptop), **6,141 MiB total**, ~2,372 MiB already used by the desktop |
| CPU | AMD Ryzen 7 7435HS, 8 cores / 16 threads, 3.10 GHz |
| RAM | 23.7 GB total |
| Disk free at start | 31.5 GB |
| PyTorch | **2.14.1+cpu** — `torch.cuda.is_available() == False` |
| Transformers | 5.18.0 (added this phase) |

**The installed torch cannot use the GPU that is present.** That single fact, plus
3.44 GB of weights against ~3.7 GB of free VRAM, is what decided the adaptation
strategy.

Measured forward-pass cost for the frozen trunk, 21 tokens:

| dtype / threads | s/sample | note |
|---|---|---|
| bf16, 8 threads, batch 8 | 2.568 | bf16 is **emulated** on this CPU |
| fp32, 8 threads, batch 8 | 0.533 | **4.8× faster than bf16** |
| fp32, 16 threads, batch 64 | **0.236** | shipped configuration |
| real extraction, 25,000 rows | 0.248–0.278 | three arms, 5.4 h total |

Extrapolations at the full Phase 5 scale: **15.6 h** for training origins, **11.8 h**
for test origins, per arm.

**Cost actually spent:** 3 feature extractions × 25,000 rows = **5.4 h**; five probe
experiments = **~75 min**, of which the head training itself is **5.6 s**; 782 MiB of
feature cache per arm.

---

## 4. Adaptation strategy, and why not something else

**Shipped: frozen backbone + trainable head.** 1,720,574,976 backbone parameters,
**0 trainable**; head 1,115 parameters (linear probe) or 34,627 (two-layer).

* **Full fine-tuning** — impossible. Weights alone are 6.9 GB in fp32; no optimiser
  state fits in the free VRAM, and a CPU backward pass through 1.7 B parameters over
  25,000 rows is not a measurement, it is a week.
* **LoRA / QLoRA / partial unfreezing** — **not run, and recorded as untested.** A
  backward pass through the frozen trunk is ~2–3× the forward cost, so even a few
  hundred steps would take hours and could not be evaluated on a meaningful population.
  This is an open question, **not** a claim that they would fail.
* **Continued pretraining on serialized energy states** — rejected on the merits. Phase 5
  already measured that attention lost to convolution here, and Phase 6 measured that
  the frozen readout is near rank-2; more text-shaped exposure to the same data is
  unlikely to change a geometric property of the activations.

**The representation bridge** (documented in full, `ml/qwen/representation.py`):

```text
per-unit channels, 168 sampled positions (Phase 5's exact channels)
  -> patches of 8 positions  ->  21 tokens covering the whole week
  -> LayerNorm + FIXED linear projection (100,448 params, seeded)
  -> inputs_embeds [B, 21, 2048] into Qwen's residual stream
  -> frozen trunk -> last-token state at layer 28
  -> PCA to 256 dims, fitted on TRAINING ROWS ONLY
  -> concat per-series embedding (8 dims) -> Linear -> 3 per-unit deltas
  -> yhat(t+h) = y(t) + delta
```

* **No text tokenisation.** Measured: `"load=0.5 lag_1=0.4"` becomes 12 unrelated BPE
  fragments. The vocabulary embedding and the LM head are unused — `inputs_embeds`
  replaces `input_ids` entirely.
* **The projector is fixed, not learned** — with a frozen trunk there is no gradient path
  to it. That is the random-features regime and it is stated as a limitation (D-083).
* **The per-series embedding is required, not decorative.** All inputs are per-unit, so
  the magnitude separating a 1.4 kW household from a 210 kW shop has been divided out.
  Phase 5's TCN carried the same embedding; without it the comparison would measure the
  handicap rather than the representation.

---

## 5. Metrics, on identical rows

10,000 test rows from a deterministic per-series balanced subsample; every model scored
on exactly those rows. kW MAE, lower is better.

| Model | h=1 (15 min) | h=4 (1 h) | h=96 (24 h) |
|---|---|---|---|
| persistence | **0.3952** | 0.8925 | 2.4427 |
| Phase 5 TCN (real checkpoint) | 0.4081 | **0.8224** | 1.6522 |
| classical GBM (refitted on the subsample) | 0.4946 | 0.9749 | **1.6168** |
| **Qwen probe, 2-layer head** | 0.5452 | 1.6060 | 2.7400 |
| Qwen probe, linear head | 1.3324 | 1.8264 | 2.9956 |

Consistency checks that the setup is sound: Phase 5's TCN reads 0.4081 kW here against
its published full-population 0.4178, and the refitted GBM reads 0.4946 against
Phase 4's 0.4242 — worse, as expected from training on 10,000 rows instead of 950,400.

**Paired bootstrap, probe vs GBM** (2,000 resamples, per-row MAE difference):

| Horizon | difference | 95% CI | excludes 0 |
|---|---|---|---|
| h=1 | +0.0506 kW | [+0.0350, +0.0652] | yes |
| h=4 | +0.6310 kW | [+0.5860, +0.6785] | yes |
| h=96 | +1.1232 kW | [+1.0095, +1.2407] | yes |

**Absolute MAE on a subsample is not the published Phase 4/5 figure**, and every
artifact says so. The paired comparison is the result.

---

## 6. The two measurements that explain the failure

### Pretraining helps — measurably, and not enough

Identical rows, identical projector, identical probe, identical head:

| Horizon | pretrained | random backbone | pretrained better by |
|---|---|---|---|
| h=1 | 1.3324 | 2.0154 | **33.9%** |
| h=4 | 1.8264 | 1.8375 | 0.6% (tie) |
| h=96 | 2.9956 | 3.8840 | **22.9%** |

So the answer to "does a pretrained transformer representation transfer to energy
dynamics" is **yes, partially**. It is not nothing, and it is not enough.

### The representation the head receives is nearly rank-2

Participation ratio of the covariance spectrum — `(Σλ)²/Σλ²` — which equals the
effective number of active dimensions out of 2048:

| Backbone | layer 8 | layer 16 | layer 24 | layer 28 |
|---|---|---|---|---|
| Qwen3-1.7B pretrained | 1.0 | 1.0 | 1.1 | **2.1** |
| random, same architecture | 7.1 | 8.4 | 9.6 | **10.0** |

The pretrained residual stream is **more** collapsed than a random one at every depth.
Whatever the final state computes for a numeric window, it is not 2048-dimensional
structure a head can read. Mean absolute pairwise cosine between windows is also higher
for the pretrained trunk (0.898 vs 0.809 at layer 28): its windows look more alike.

This is a measured geometric property of the activations, not a metaphor, and it is the
single most important input to Phase 7.

---

## 7. Regime and error results

At h=1, kW MAE on the same rows:

| Demand tercile | GBM | persistence | Phase 5 TCN | Qwen |
|---|---|---|---|---|
| low | **0.0834** | 0.0690 | 0.0800 | 0.1197 |
| typical | **0.2548** | 0.2626 | 0.2599 | 0.2985 |
| high | **1.1456** | **0.8541** | 0.8845 | 1.2176 |

| Ramp tercile | GBM | persistence | Phase 5 TCN | Qwen |
|---|---|---|---|---|
| low | 0.0803 | **0.0034** | 0.0343 | 0.1022 |
| typical | 0.2008 | 0.0602 | **0.1035** | 0.2132 |
| high | 1.2028 | 1.1220 | **1.0867** | 1.3203 |

* Error concentrates: the high demand tercile costs **14×** the low tercile and is ~73%
  of total error; the high ramp tercile is ~84%.
* **Qwen is fourth in every regime**, including the difficult ones. It does not carve
  out a niche where a specialist would help.
* **The best model is regime-dependent**: persistence is untouchable in low-ramp periods
  (0.0034 kW), the TCN wins high-ramp, the GBM wins demand. This is the strongest
  evidence *for* some form of routing — between **persistence / GBM / TCN**, not
  between Qwen variants.

Error distribution at h=1 (kW):

| Model | median | p90 | p99 | p99/median | >1 kW |
|---|---|---|---|---|---|
| persistence | 0.0476 | 0.973 | 6.170 | 130 | 9.74% |
| Phase 5 TCN | 0.0696 | 0.998 | 5.532 | 79 | 9.98% |
| classical GBM | 0.0982 | 1.222 | 5.702 | 58 | 12.88% |
| Qwen probe | 0.1327 | 1.256 | 6.918 | **52** | 13.49% |

Qwen has the **flattest** relative tail of the four and the **worst median**: it is
consistently mediocre rather than occasionally catastrophic. That is a real, if
unattractive, property — it does not fail spectacularly, it just does not help.

---

## 8. Ablations

| Ablation | Question | Answer |
|---|---|---|
| `random_backbone` | do pretrained weights beat a random deep feature map? | **yes, partly** — 33.9% / 0.6% / 22.9% better |
| `probe_depth` | was the linear probe under-parameterised? | **yes** — h=1 improves 1.3324 → 0.5452 kW (1,115 → 34,627 params) |
| `no_cyclic` | do absolute-time channels matter under RoPE? | **no** — removing them is 36.7% / 33.3% *better*, 2.5% worse at h=96 |
| `layer_16` | is an intermediate layer a better readout? | **no** — 23.33 kW at h=1 vs 1.33 kW at layer 28 |

**`no_cyclic` confirms Phase 5's D-077 across architectures.** Phase 5 found the sin/cos
channels were not earning their place under a convolution; Phase 6 tested the stronger
motivation (RoPE is relative-only, so absolute time must come from the channels) and
found them mildly *harmful* at short range.

**Deliberately NOT run, and said so:** `patch_size = 1` (168 tokens) would cost ~16 h per
extraction pass; `aggregate = "mean"` needs its own pass. Both are open questions, not
omissions.

---

## 9. Failures and defects found

Three were mine, in the baseline path, and **all three initially made Qwen look better**:

1. **GBM trained on kW targets against per-unit features.** Train MAE 14.63 kW where
   Phase 4 reported 0.42 kW. Phase 4 fits every baseline in per-unit space and scales
   only when reporting.
2. **TCN baseline compared per-unit predictions against kW actuals.** 13.28 kW where
   Phase 5 reported 0.4178 kW. `predict_series` returns per-unit, not kW.
3. **`series_index` omitted from the feature builder.** Its default assumes Phase 4's row
   layout ("each series contributes the same origins in turn"); the Phase 6 subsample is
   ordered by series, so every row was paired with **another customer's demand**. The
   symptom was a GBM ten times worse than persistence.

Together these produced a fake "Qwen beats GBM by 81%" result. They were caught by
checking the baselines against Phase 4/5's *published* figures rather than trusting
them, which is the only reason the real result is trustworthy.

Other defects found and fixed:

4. **A fresh random projector per batch** in the first extraction implementation, which
   would have injected batch-dependent noise. Now one seeded projector per extraction,
   asserted deterministic by test.
5. **The probe had no per-series embedding.** Added, because per-unit inputs make
   customer identity unobservable otherwise and Phase 5's TCN had one. It did **not**
   rescue the result — h=1 went 1.2608 → 1.3324 kW, i.e. marginally worse — so the
   probe was not merely handicapped by a missing identity signal. The fix was still
   correct: without it the comparison would have been unfair.
6. **`np.ascontiguousarray(..., dtype=torch.float32)`** — numpy needs a numpy dtype.
7. **`SequenceConfig` rejected a list of horizons** where it wanted a tuple, so the
   `single_horizon`-style TOML config crashed. Normalised, with a regression test.

---

## 10. Architectural inspection for Phase 7

Recorded in `artifacts/phase6/` and asserted by `tests/ml/test_qwen_checkpoint.py`,
which is the regression baseline Phase 7 needs **before** any structural change:

| Site | Shape | Phase 7 note |
|---|---|---|
| decoder layers | 28 × `Qwen3DecoderLayer` | uniform; no layer is special |
| attention | `Qwen3Attention` | `q_proj [2048,2048]`, `k_proj [1024,2048]`, `v_proj [1024,2048]`, `o_proj [2048,2048]`, `q_norm`/`k_norm [128]` |
| FFN | `Qwen3MLP` | `gate/up [6144,2048]`, `down [2048,6144]` — the natural expert-swap unit |
| norms | `Qwen3RMSNorm`, eps 1e-6 | **pre-norm**: both norms precede their sublayer |
| residuals | attention and FFN, two paths | |
| embedding | `[151936, 2048]`, tied | **unused by Phase 6** |
| positional | RoPE, θ=1e6, max 32,768 | relative only; no absolute position table |

`assert_architecture_intact()` raises on any deviation, with the consequence named, and
tests confirm it fires on a changed hidden width and on a deleted layer. **Nothing was
modified in Phase 6.**

---

## 11. Blocked or out of scope

| Item | Status |
|---|---|
| LoRA / QLoRA / partial unfreezing | **Not run** — no VRAM for optimiser state, CPU backward infeasible at scale. Untested, not refuted. |
| Trainable backbone | Same |
| Full-population evaluation | ~11.8 h per arm; scored on a documented subsample instead |
| `patch_size = 1`, `mean` pooling | ~16 h per pass each |
| Multiple seeds | One seed (`20260101`). Paired bootstrap gives row-level uncertainty; **seed** uncertainty is not quantified. |
| Action-conditioned dynamics | Still unavailable — `docs/world_model_requirements.md` unchanged |
| Feeder PV, net load, battery, EV, wind | Still unavailable (Phase 3/4 gaps) |
| Phase 3 balance | Still intentionally failing: 20.4506 kW vs 0.5 kW tolerance |

---

## 12. Phase 7 requirements and recommendation

`docs/moe_design_requirements.md` has been **rewritten from Phase 6's measurements**.

The honest recommendation: **Phase 7 as originally conceived is not justified by this
evidence.** Structural MoE surgery on Qwen assumes its dense form is a good
representation needing specialisation. Phase 6 measured that premise and found a
near-rank-2 readout and no regime where Qwen leads.

If Phase 7 proceeds, the requirements are:

1. **Evidence the shared backbone is not the bottleneck** — it currently reads at
   participation ratio 2.1. Adding experts on top of a collapsed trunk inherits the
   collapse. This must be argued and measured, not assumed.
2. **A gate over origin-observable features only.** The terciles in §7 are computed from
   the demand being predicted and **cannot** be used by a router.
3. **A leakage test covering the gate's inputs**, not just the sequence window.
4. **Load-balancing loss**, per-expert capacity factors reported.
5. **A no-MoE control at matched budget**, same rows, same run.
6. **Per-regime reporting against the best single model** — the low-ramp regime, where
   persistence reaches 0.0034 kW, is the one a router is most likely to damage.
7. **Costed as a multiple of 0.248 s/sample.** A multi-expert forward pass is several
   times a frozen one.

**Better-supported alternative for Phase 7:** build the **regime router over the models
that already win regimes** — persistence for low-ramp, the TCN for high-ramp, the GBM for
demand level. §7 shows the best model is regime-dependent and that Qwen is never it.
That is a measurable, cheap win, and it is the routing question Phase 6 actually
answered.

---

## 13. Critical context

* **The task was inherited unchanged.** Sequence index version **`sq-437c16b5d925`** —
  byte-identical to Phase 5's, asserted by test. Phase 6 changes the model, not the task.
* **Never invent data.** No synthetic energy values, weather, PV, battery, EV, wind or
  topology. Unsupported targets are refused with evidence.
* **The chronological split is the contract.** Unshuffled; validation and test origins at
  stride 1; scaling fitted on training rows only, including the PCA basis.
* **Absolute subsample MAE is not the published figure.** Always compare paired, on
  identical rows.
* **Check the baselines against published figures before believing a result.** Three
  baseline bugs made Qwen look better than it is.
* **One seed, no significance claims** beyond the row-level paired bootstrap.
* **Do not run heavy commands concurrently with training.** CPU contention invalidated a
  Phase 5 run; 5.4 h of extraction went wrong once already.
* **fp32, not bf16, on this CPU** — 4.8× measured difference.
* **The checkpoint is never committed.** `models/*` is git-ignored; re-download with
  `huggingface_hub.snapshot_download` at the pinned revision.
* **Commit identity:** `PrathamKapoor <prathamkapoor027@gmail.com>`. No attribution lines,
  no AI or bot collaborators, on any commit, PR, tag or release note.

---

## 14. Repository state

```text
src/energy_intelligence/ml/qwen/checkpoint.py      identity, verification, Phase 7 invariants
src/energy_intelligence/ml/qwen/representation.py  the numeric bridge
src/energy_intelligence/ml/qwen/features.py        subsample + feature cache
src/energy_intelligence/ml/qwen/probe.py           PCA basis, head, training
src/energy_intelligence/ml/qwen/baselines.py       baselines on identical rows, bootstrap
src/energy_intelligence/ml/qwen/experiment.py      runner, diagnostics, verdict
src/energy_intelligence/config/qwen.py             strict experiment config
configs/qwen.toml                                  the shipped experiment
scripts/phase6_extract.py                          backbone extraction (expensive)
scripts/phase6_probe.py                            probe experiments (cheap)
tests/ml/test_qwen_checkpoint.py                   21 tests, incl. the real 1.7B load
tests/ml/test_qwen_probe.py                        30 tests: bridge, leakage, training
tests/ml/test_qwen_experiment.py                   36 tests: config, subsample, statistics
tests/ml/test_qwen_runner.py                       16 tests: end-to-end runner + feature cache
```

Artifacts (git-ignored, regenerable): `artifacts/phase6/features/*.npz` (782 MiB per
arm), `artifacts/phase6/qwen-*/result.json` and `summary.md`,
`artifacts/phase6/experiments.json`, `artifacts/phase6/load_test.json`,
`artifacts/phase6/forward_cost.json`, `artifacts/phase6/extraction.log`.

**Tests: 913 pass** (was 810). The qwen package: `representation.py` 97%,
`checkpoint.py` 95%, `probe.py` 94%, `experiment.py` 91%, `features.py` 78%. Health
7/7. The runner and the feature cache had almost no coverage until
`test_qwen_runner.py` drove them end to end on a synthetic series with a stub
backbone - without it, `experiment.py` stood at 24%.

Dependencies added: **`transformers>=4.51.0`** (floor is the checkpoint's own declared
minimum) plus its transitive `safetensors`, `tokenizers`, `huggingface_hub`, `regex`,
`pyyaml`, `tqdm`. **pandas remains deliberately absent.** Phase 4's three dependencies
are unchanged.

Reproduce:

```powershell
uv run python scripts/phase6_extract.py full pretrained random nocyclic   # ~5.4 h
uv run python scripts/phase6_probe.py probe random nocyclic layer_16 probe_depth
energy-intel qwen verify
energy-intel qwen inspect
uv run pytest tests/ml -q
```
