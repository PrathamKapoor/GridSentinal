# Decision Record

Every meaningful architectural and implementation decision, with the options
considered, what was selected, why, and what the consequence is.

Format: **D-NNN — Title**, then *Choices*, *Selected*, *Rationale*,
*Consequence*.

Status values: `Accepted`, `Superseded by D-NNN`, `Revoked`.

---

## D-001 — Start from an empty repository

**Context:** Phase 1 Step 1 requires inspecting the repository before changing
anything. `C:\Projects\Schneider` was completely empty: 0 files, 0
subdirectories, and not a git repository. `decisions.md`, `flow.md` and
`handoff.md` did not exist. No prior architectural decision existed to
contradict, and no work could be destroyed.

**Choices:** Adopt the existing structure (none existed) / define a fresh one.

**Selected:** Define a fresh structure, and record this as greenfield rather
than a migration.

**Rationale:** There is no legacy layout to preserve and nothing to migrate, so
no restructuring risk exists. This is recorded explicitly so a later reader does
not assume a restructuring decision was made and subsequently lost.

**Consequence:** All Phase 1 directories are new. No compatibility shims are
needed.

**Status:** Accepted

---

## D-002 — Target Python 3.12

**Context:** Phases 6-7 plan to customise `Qwen3-1.7B-Base` and implement a
custom Mixture-of-Experts. That work typically needs quantization and kernel
wheels (`bitsandbytes`, `flash-attn`, `xformers`, `deepspeed`, `megatron-core`),
whose releases historically lag the newest Python. The machine's default
interpreter was Python 3.13.14; 3.12 and 3.10 were also installed.

**Choices:** 3.13 (newest, already local) / 3.12 (widest ML wheel coverage) /
3.10 (maximum compatibility, oldest).

**Selected:** **Python 3.12**, pinned in `.python-version` and enforced by
`requires-python = ">=3.12,<3.13"`. `uv` provisioned CPython 3.12.5; the
machine's system interpreter was not modified.

**Rationale:** The ML dependency set for the planned MoE work is the dominant
risk in this project, and 3.12 is the version where that set is most reliably
complete. The newest interpreter is the one most likely to lack a wheel when
Phase 6 begins.

**Consequence:** The environment uses a downloaded 3.12 rather than the local
3.13. `requires-python` uses an upper bound so a 3.13-only feature cannot be
introduced accidentally. Re-evaluate at Phase 6 if the ecosystem has converged.

**Status:** Accepted

---

## D-003 — Manage the environment with `uv`

**Context:** Phase 1 needs a reproducible Python environment. `uv` 0.12.5 was
already installed on the machine; `pip` 26.1.2 was also available.

**Choices:** `uv` with PEP 621 `pyproject.toml` / stdlib `venv` + `pip` /
separate `requirements*.txt` files.

**Selected:** `uv` with a single PEP 621 `pyproject.toml`.

**Rationale:** `uv` was already present (no new tooling assumption), creates the
venv itself, and produces an exact `uv.lock` so every teammate and the future
Jenkins pipeline reproduces the same environment from one file. `pip` alone
offers no lockfile, so runs would not be bit-reproducible. Split requirement
files duplicate dependency metadata and drift.

**Consequence:** `uv.lock` is committed and is the source of truth for exact
versions; `pyproject.toml` holds the version ranges. Contributors must use `uv
sync` rather than `pip install`. Added a build-backend dependency
(`hatchling`) to produce the editable install.

**Status:** Accepted

---

## D-004 — Zero runtime dependencies in Phase 1

**Context:** The brief requires the smallest reasonable dependency set and
forbids installing packages "because they might be useful later". The Phase 1
feature set is configuration, logging, tests and a health check, all of which
Python's standard library can implement.

**Choices:** `pydantic` for config validation / `PyYAML` + config files /
stdlib only.

**Selected:** **`dependencies = []`** in `pyproject.toml`. Everything Phase 1
needs comes from the standard library (`dataclasses`, `tomllib`, `logging`,
`argparse`, `pathlib`, `json`).

**Rationale:** No scientific or ML code exists yet, so any ML dependency would
be speculative. Starting at zero keeps the environment reproducible and makes
each later dependency an explicit, justified addition rather than inherited
baggage.

**Consequence:** Validation logic is hand-written instead of schema-generated.
This is more code, but every rule is visible and directly testable. The cost is
accepted in exchange for zero third-party surface area. Guarded by
`test_runtime_dependencies_are_empty` and
`test_no_ai_or_ml_dependencies_are_imported_at_import_time` in
`tests/test_package.py`, which will fail loudly if a later phase breaks this
without a recorded decision.

**Status:** Accepted

---

## D-005 — Development dependencies: `pytest` and `pytest-cov` only

**Context:** The brief requires test infrastructure and forbids creating tests
that claim to test non-existent AI behaviour.

**Choices:**

```text
Package:    pytest
Why needed: Test runner. The brief mandates a test suite; Python has no built-in runner.
Alternatives considered:
  - unittest (stdlib): ships with Python, but no fixtures, no parametrisation
    and no plugin ecosystem. The suite needs tmp_path fixtures and parametrised
    validation cases.
  - nose/nose2: effectively unmaintained.
  - doctest only: cannot cover configuration rejection or health-check output.
Why selected: Industry standard, first-class fixtures and parametrisation,
best-in-class failure output, and native pytest-cov support.
Consequence: One dev dependency. Tests use fixtures and parametrisation heavily.

Package:    pytest-cov
Why needed: Measures which Phase 1 code the suite actually exercises. Without it
            "tests pass" says nothing about whether the code was tested, which is
            precisely the self-verification property this project is about.
Alternatives considered:
  - coverage.py directly: needs a separate command and does not integrate with
    the pytest reporting.
  - No measurement: unquantifiable coverage; regressions in untested branches
    go unnoticed.
Why selected: Standard coverage plugin; reports within the same test run.
Consequence: `uv run pytest --cov` reports coverage. Current figure: 94%
statement coverage (see handoff.md).
```

**Selected:** `dev = ["pytest>=8.3", "pytest-cov>=6.0"]`.

**Consequence:** Dev-only, never a runtime dependency. `colorama` and `pygments`
are installed transitively by pytest on Windows and are not direct choices.

**Status:** Accepted

---

## D-006 — Configuration: frozen dataclasses + stdlib `tomllib`

**Context:** Future experiments must be reproducible and configurable without
editing source code. The brief names the required knobs (random seed, device,
model path, dataset path, experiment name, logging level, output directory) and
forbids hard-coding architecture-level choices such as MoE expert count or
optimizer choice.

**Choices:**

1. **Schema mechanism**
   - `pydantic` v2 — schema plus validation from one declarative object; less
     boilerplate and better error messages, but adds a runtime dependency.
   - `PyYAML` + dataclasses — familiar in ML research, but needs a dependency
     and YAML has ambiguous typing (e.g. `seed: 007`).
   - **frozen dataclasses + hand-written validators** — selected.

2. **File format**
   - JSON — no comments, poor for documented config.
   - YAML — ambiguous typing, extra dependency.
   - **TOML via stdlib `tomllib`** — selected.

3. **Mutability** — mutable dataclass / **frozen dataclass** / dict.

**Selected:** `AppConfig` is a `frozen=True, slots=True` dataclass validated by
explicit functions in `src/energy_intelligence/config/schema.py`, populated from
TOML via `tomllib` in `config/loader.py`.

**Rationale:** Option 1 keeps the zero-dependency stance of D-004 and makes each
validation rule individually testable — 84 config tests assert specific rules,
which a schema library would have generated opaquely. TOML is a
standard-library format since Python 3.11, supports comments (essential for
documenting *why* a value exists), and has no type ambiguity. Frozen dataclasses
make mutation impossible, so a config cannot change mid-run and silently break
reproducibility; type hints carry editor and type-checker support.

**Consequence:** More code than a schema library, accepted deliberately. TOML
writes are read-only here, so there is no round-trip concern. Future phases
adding a config field must add it to `AppConfig`, `CONFIG_FIELDS`, the coercion
function and `configs/default.toml` — the unknown-key rejection (D-007) turns a
missed step into a test failure rather than a silently ignored setting.

**Status:** Accepted

---

## D-007 — Configuration keys are strict: unknown keys are rejected

**Context:** Reproducibility depends on knowing every setting that affected a
run. Silently ignoring an unrecognised key means a run used a value the author
did not intend, and the mistake is invisible.

**Choices:** Ignore unknown keys / **reject them** / warn only.

**Selected:** Reject. `coerce_app_config` collects an error for every key not in
`CONFIG_FIELDS`, and `load_config` rejects unknown keys in layer files and in
CLI overrides.

**Rationale:** A silent default is the most expensive kind of bug in an
experiment: the run completes, looks plausible, and is not reproducible. Failing
fast costs one line of TOML to fix.

**Consequence:** Renaming or removing a config key is a breaking change that
must update every existing config file. This is intended. Tests confirm that
`experimant_name` (a plausible typo) is rejected rather than ignored.

**Status:** Accepted

---

## D-008 — All configuration errors are reported together

**Context:** A user editing a TOML file with several mistakes should fix them in
one pass, not discover them one run at a time.

**Choices:** Fail on the first error / collect and report all errors.

**Selected:** Collect all errors into `ConfigValidationError.errors` and render
them as a numbered list.

**Rationale:** Config errors are cheap to fix and expensive to rediscover one at
a time. Collecting them costs little and removes an entire class of frustration.

**Consequence:** `ConfigValidationError` carries a tuple of messages rather than
a single string. Callers that want only the first error must index it
explicitly.

**Status:** Accepted

---

## D-009 — Configured paths are resolved to absolute, and not required to exist

**Context:** Relative paths are ambiguous when the working directory changes.
But `model_path` and `dataset_path` point at artifacts that later phases create
— Phase 3 owns datasets and Phase 6 owns the model — so in Phase 1 they
legitimately do not exist yet.

**Choices:** Require all paths to exist / resolve and defer existence checks /
keep paths relative.

**Selected:** Resolve relative paths against the detected project root into
absolute `pathlib.Path` values; do **not** require existence at load time.
Existence is a health-check concern, not a load-time concern.

**Rationale:** Absolute resolution removes working-directory sensitivity and
makes paths loggable and comparable across runs. Requiring existence would make
a valid Phase 1 configuration unbuildable, because the artifacts it names are
out of scope for this phase.

**Consequence:** A config naming a nonexistent model path loads successfully and
logs an absolute path pointing nowhere. That is intentional and covered by
`test_paths_are_not_required_to_exist`. Existence of the *canonical project
directories* is checked separately by the health check.

**Status:** Accepted

---

## D-010 — `experiment_name` is restricted to a slug character set

**Context:** The experiment name becomes a log file name
(`logs/<name>.jsonl`) and an experiment directory name
(`experiments/<name>`). An unvalidated name containing `/` or `..` would place
files outside the intended directory.

**Choices:** Accept any string and sanitise on use / accept any string / **validate
against `^[a-z0-9][a-z0-9._-]{0,64}$` at load time**.

**Selected:** Validate at load time: lowercase alphanumerics plus `.`, `_`, `-`,
must start alphanumeric, maximum 64 characters.

**Rationale:** Failing at configuration load is far better than failing when a
run tries to write a file. A strict, stable character set also keeps experiment
identifiers portable across Windows, Linux and the future Jenkins filesystem
(`:` and `\` are invalid on Windows, `\` is a path separator on POSIX).

**Consequence:** Uppercase and spaces are rejected in experiment names. This is
intentional and tested (`test_invalid_experiment_name_is_rejected`).

**Status:** Accepted

---

## D-011 — Config resolution order: default file, explicit file, env var, CLI

**Context:** Experiments need per-experiment configuration without editing
`default.toml`, and the future CI pipeline needs to select configuration without
rewriting commands.

**Choices:** Single config file / single file plus CLI flags / **layered
resolution** / hierarchical config with `extends` and environment interpolation.

**Selected:** Layered resolution, lowest priority first:

1. `configs/default.toml`
2. the file passed via `--config PATH`
3. the file named by `$ENERGY_INTEL_CONFIG`, when `--config` was not given
4. CLI overrides `--seed`, `--log-level`, `--experiment-name`

`--config` takes precedence over the environment variable so an explicit
invocation always wins. `None` values from absent CLI flags are ignored rather
than written, so an unspecified flag cannot blank a real value.

**Rationale:** Layers the minimum machinery needed for reproducibility without
introducing inheritance chains or interpolation, which have no consumer until
Phase 15 adds CI. Keeps every setting overridable without touching source code.

**Consequence:** No `extends` chains and no `${ENV_VAR}` interpolation. If a
later phase needs them, that is a new decision. The environment variable
`ENERGY_INTEL_CONFIG` is read but never written, so loading is side-effect free
(covered by `test_environment_is_not_mutated`).

**Status:** Accepted

---

## D-012 — Structured logging: console + JSON Lines, stdlib `logging`

**Context:** Logs must eventually answer what happened, when, which component,
which experiment and which configuration. The brief forbids building an
elaborate observability platform.

**Choices:** Plain print statements / stdlib `logging` with plain formatting /
**stdlib `logging` with structured formatters** / a third-party library such as
`structlog` / an observability backend.

**Selected:** Stdlib `logging` with two sinks: a human-readable console
formatter and a JSON Lines file formatter, in
`src/energy_intelligence/logging_setup.py`.

**Rationale:** Satisfies every requirement with zero dependencies (D-004) and
gives standard `logging` behaviour, level filtering, handler composition and
`exc_info` handling for free. JSON Lines is the right sink for later ingestion:
one object per line streams cleanly, is trivially greppable, and is the common
currency of log pipelines. `structlog` would add a dependency and wrap a
facility the standard library already provides adequately at this scale.

**Consequence:** Run identity (`experiment`, `config_id`, `run_id`) is attached
by a `logging.Filter` rather than by call sites, so no call site can forget it.
Log files are opened in append mode, so a visible INFO run-start marker
(`event: logging_initialised`) delimits runs within a long-lived file. Fields
set via `extra=` become top-level JSON keys automatically, which is how later
phases will attach domain fields such as `horizon_hours` or `regime`.

**Status:** Accepted

---

## D-013 — Logging owns only the handlers it installs

**Context:** The idempotency guard in `initialize_logging` originally treated
*any* handler present on the `energy_intelligence` logger as "already
initialised". In practice this silently disabled all logging setup whenever a
host had attached its own handler — pytest's `caplog` handler during the test
run, and potentially a Jupyter kernel or an embedding application later.

**Choices:** Count all handlers / **tag and track only our own** / configure the
root logger.

**Selected:** Handlers created here are tagged with
`_energy_intelligence_managed`; idempotency, reset and shutdown consider only
tagged handlers.

**Rationale:** A library must not remove handlers it does not own — doing so
breaks the host's logging — and must not be disabled by them. Tagging makes
ownership explicit and both failure modes are now regression-tested
(`test_host_handlers_do_not_disable_initialisation`,
`test_reset_logging_leaves_host_handlers_alone`).

**Consequence:** `reset_logging()` no longer empties the handler list outright;
it detaches only ours. The project logger also sets `propagate = False` so
records are not duplicated by the root logger, and the root logger is never
configured — importing this package cannot change a host application's logging.

**Status:** Accepted

---

## D-014 — `src/` layout with a `pyproject.toml`-installed package

**Context:** The project needs an importable, installable package rather than a
directory of loose scripts, so that tests, the CLI and later phases all import
the same code.

**Choices:** Flat layout with `energy_intelligence/` at the repository root /
**`src/` layout with an editable install** / namespace packages.

**Selected:** `src/energy_intelligence/`, installed editable via
`hatchling`, with `[tool.hatch.build.targets.wheel] packages = ["src/energy_intelligence"]`.

**Rationale:** The `src/` layout makes it impossible to accidentally import the
package from the working directory, which would let a test pass against
uninstalled code. Editable install means edits take effect without reinstalling
during development.

**Consequence:** `uv sync` must be run after cloning before tests pass. The
health check reports a warning if the package resolves to `site-packages`
instead of the checkout, so the failure mode is visible. `[tool.pytest.ini_options]
pythonpath = ["src"]` additionally lets a raw `pytest` run work without an
install.

**Status:** Accepted

---

## D-015 — Only Phase 1 directories are created

**Context:** The brief's conceptual structure lists `data/`, `models/`,
`forecasting/`, `moe/`, `optimization/`, `simulation/`, `assurance/`,
`agents/` and `mlops/` subpackages. It explicitly instructs not to create every
directory merely because it appears in the list, and not to pretend an
empty directory contains functionality.

**Choices:** Create the full conceptual tree / create only what Phase 1
implements / create the full tree with a reserved-for-later README in each.

**Selected:** Only directories with a Phase 1 responsibility were created:

```text
Created (implement something):  src/energy_intelligence/{config,cli,health,
                                logging_setup,paths}, tests/, configs/, docs/
Created (Phase 1 conventions):   data/{raw,processed,external}/, models/,
                                experiments/, artifacts/, logs/
NOT created (Phase 2+):          src/energy_intelligence/{data,models,
                                forecasting,moe,optimization,simulation,
                                assurance,agents,mlops,utils}/
```

The reserved directories each contain a `.gitkeep` so the convention survives a
fresh clone, and the full Phase 2+ map is recorded here and in `README.md` as a
plan rather than as code.

**Rationale:** An empty package is worse than an absent one: `import
energy_intelligence.moe` succeeding would falsely imply a Mixture-of-Experts
implementation exists. Recording intent in documentation satisfies the brief's
"document that rather than pretending" requirement without shipping hollow
structure.

**Consequence:** `energy_intelligence.utils` was created during scaffolding and
then deliberately removed as unused. Later phases add their subpackage when they
implement it. `paths.py` holds the data/model/experiment conventions as
constants so later phases do not invent their own.

**Status:** Accepted

---

## D-016 — Data, model, experiment and artifact contents are git-ignored

**Context:** The brief requires directory conventions for data, models and
experiments, but model checkpoints and datasets are large binaries and logs are
regenerable outputs.

**Choices:** Commit everything / commit only code and configuration / **ignore
contents, track the directories via `.gitkeep`**.

**Selected:** `data/{raw,processed,external}/`, `models/`, `experiments/` and
`artifacts/` have their contents ignored with `!*/.gitkeep` negation.
`logs/` is ignored entirely, including its `.gitkeep` is tracked.

**Rationale:** Keeps the repository light and clonable while guaranteeing a
fresh clone still has the correct layout. The layout is code-level knowledge
enforced by `ProjectPaths.required_directories` and asserted by
`test_real_repository_has_the_layout`, so it is verified on every run rather
than merely hoped for.

**Consequence:** Model weights and datasets are never version-controlled. Phase
3 must document how they are obtained and versioned (likely externally, with a
checksum recorded in configuration). The ignore patterns are declared now so a
later phase cannot accidentally commit a multi-gigabyte checkpoint.

**Status:** Accepted

---

## D-017 — The health check is a set of independent checks with exit codes

**Context:** The brief requires a command that verifies the Phase 1 environment
and reports a clear result, without inventing a command that was never run.

**Choices:** A single boolean script / **a set of named checks returning an
aggregate report** / pytest tests only.

**Selected:** `check_health()` in `src/energy_intelligence/health.py` runs six
independent checks — `python_version`, `package_import`, `config`,
`directories`, `test_infrastructure`, `logging` — each returning a `CheckResult`
with status `ok` / `warn` / `fail`, detail and a remedy. `HealthReport.exit_code`
is 0 only when nothing failed. Surfaced as `energy-intel health` (text) and
`energy-intel health --json`.

**Rationale:** Independent checks mean one broken foundation never masks
another, and each result carries its own remedy so a failure is actionable. Exit
codes make the command usable as a CI gate directly, which is what Phase 15's
Jenkins pipeline will need. Warnings do not fail the report, so an advisory
observation (such as importing from `site-packages`) does not block work.

**Consequence:** Six checks exist, covering exactly the five foundations the
brief names plus the console script. Tests cover both the passing state and
every failure branch, so the check is not vacuous. `--json` output is stable
enough for machine consumption. Individual checks accept an optional `paths`
argument so tests can exercise failure branches against a temporary directory
instead of mutating the real checkout.

**Status:** Accepted

---

## D-018 — Register of deliberately undecided items

Items explicitly left open, per the brief's instruction to write `TBD` rather
than guess. **None of these are implemented.** Items marked *resolved* were
settled by a later decision; the "Resolved in" column says which.

| Item | Status | Decided in |
|---|---|---|
| Which energy system is modelled (grid, microgrid, building, DER fleet) | **resolved** — renewable-integrated distribution DER environment | D-022 |
| Network topology fidelity | **resolved** — topology-aware, node-granular, physics deferred | D-023 |
| Temporal resolution and horizon | **resolved** — 15 min native, H=96 day-ahead | D-024 |
| Control authority over flexible assets | **resolved** — tiered per-asset authority | D-028 |
| Dataset representability anchor | **resolved** — SMART-DS (anchor only, not a download) | D-038 |
| Concrete dataset selection, size, cadence, licensing | TBD | Phase 3 |
| Tariff model and currency unit | TBD | Phase 3 |
| Grid emission factors | TBD | Phase 3 |
| Baseline forecasting models | TBD | Phase 4 |
| Energy World Model architecture | TBD | Phase 5 |
| Whether the ~100M edge model shares weights with the ~1.7B model | TBD | Phase 5-6 |
| Energy-MoE expert count, routing objective, regime definitions | TBD | Phase 7 |
| Uncertainty estimation method (quantile / ensemble / conformal) | TBD — container exists | Phase 8 |
| Flexibility quantification method | TBD — representation exists | Phase 9 |
| Optimizer (MPC / LP / MILP / NLP / RL) — must follow the problem formulation | TBD | Phase 10 |
| Digital twin fidelity and simulator; power-flow solver | TBD | Phase 11 |
| Red Team scenario library | TBD | Phase 12 |
| Decision-assurance scoring formula and thresholds | TBD | Phase 13 |
| Final MLOps agent roster (matrix is a candidate list, not a commitment) | TBD | Phase 14 |
| Jenkins pipeline definition | TBD | Phase 15 |
| Ramp limits and minimum up/down times | TBD — needs measured data | Phase 3 |
| UI framework and reference interface | TBD | Phase 19 |


---

## D-019 — Deliberately not built in Phase 1

Per the brief, and verified by the absence of both code and dependencies:

* No AI model, no training, no inference.
* No `Qwen3-1.7B-Base` download or customisation.
* No Energy-MoE or router.
* No optimizer or decision engine.
* No digital twin or simulator.
* No red-team or adversarial engine.
* No MLOps agents, Jenkins pipeline or CI configuration.
* No UI, frontend, dashboard, charts or authentication.
* No database, cloud service, API, queue, Docker or Kubernetes.
* No dataset download.

Consequence: the repository contains 558 statements of foundation code and no
energy-domain logic whatsoever. The package docstring states this explicitly so
the absence is intentional and visible rather than looking like unfinished work.

**Status:** Accepted

---

## D-020 — Commit policy for research artifacts

**Context:** The brief instructs not to commit `paper/`, `research/` or `*.tex`
unless the user changes that requirement.

**Choices:** Commit research artifacts / **do not commit them**.

**Selected:** Do not commit `paper/`, `research/` or `*.tex`. None exist in
Phase 1, so no ignore rule has been added yet — an unused ignore rule would be
noise.

**Rationale:** Explicit instruction, and no current content to act on.

**Consequence:** When a paper directory is created in a later phase, an ignore
rule must be added at that time. Flagged in `handoff.md` so it is not forgotten.

**Status:** Accepted

---

## D-021 — Normalise line endings via `.gitattributes`

**Context:** The working tree is Windows, where `core.autocrlf` converts LF to
CRLF in the working copy. A future Jenkins or CI runner will be Linux and
expects LF. Without normalisation the same file differs between contributors,
and every diff that touches line endings is noise that hides real changes.

**Choices:** Leave it to per-machine `core.autocrlf` / set
`core.autocrlf` locally only / **commit a `.gitattributes`**.

**Selected:** `.gitattributes` with `* text=auto eol=lf`, plus explicit
`eol=crlf` for `.bat`, `.cmd` and `.ps1` (Windows tooling mishandles LF there)
and `binary` for model and data formats.

**Rationale:** Line-ending normalisation belongs in the repository, not in each
contributor's global git config, because reproducibility across machines is a
stated project requirement. Committing it makes the guarantee travel with the
code.

**Consequence:** Text files are stored as LF and checked out as LF. Windows
contributors see LF in their editor, which every modern editor and IDE handles
correctly. Future Phase 3 data files and Phase 6 model checkpoints are already
declared `binary` so a later phase cannot accidentally normalise a checkpoint.

**Status:** Accepted

---

# Phase 2 — Energy system definition and data model (D-022 … D-039)

Phase 1 (D-001 … D-021) established *how* the software is organised. Phase 2
establishes *what energy system is modelled*. The decisions below are the
contract every later phase depends on.

## Source-repository findings (context for D-022 … D-024)

Both source repositories were cloned and inspected directly rather than assumed.

| Finding | Evidence |
|---|---|
| **SAT-SA is a cybersecurity SOC supervision project, with zero energy content** | `battery`, `feeder`, `voltage`, `storage` → 0 hits in `src/` |
| **Guard is a forecasting-MLOps project on aggregate bulk data**, not distribution DERs | RTS-GMLC, 2020, hourly, 8,784 rows, targets LOAD/WIND/PV; `battery`, `EV`, `HVAC`, `curtail`, `feeder`, `voltage`, `storage` → **0 hits** in `src/` |
| **Neither models uncertainty** | Guard: `quantile`, `conformal`, `ensemble`, `aleatoric`, `epistemic` → 0 hits |
| **Neither has a red team, digital twin or optimizer** | `red.?team`, `adversar`, `digital.?twin`, `simulat`, `optimiz`, `MILP`, `MPC`, `cvxpy` → 0 hits in Guard |
| **SAT-SA's advertised 13-field agent contract is not implemented** | `analysis_period` has 2 hits, both in docstrings; no dataclass/Protocol with those fields exists |
| **Guard's agent contract and capability firewall are real** | `agents/schemas.py` `AgentOutput`/`OUTPUT_CONTRACT_FIELDS`; `agents/firewall.py` allow-list vs block-list, unknown blocked by default |
| **SAT-SA's provenance layer is strong** | `SourceRecord`, `ProvenanceRecord`, hash-chained ledger, canonical-form module |

**Conclusion driving D-022:** the energy domain has to be built here. What is
reusable is architectural *pattern*, not code. The split is recorded in
`docs/agent_responsibility_matrix.md`.

---

## D-022 — Model a renewable-integrated distribution-level DER environment

**Context:** The brief fixes the boundary as a distribution-level environment with
DERs and forbids inventing a specific number of buses, buildings, batteries, solar
plants or EVs. Both source repositories model something else entirely.

**Choices:** Match a source repository's scope (aggregate bulk forecasting) /
single lumped node with no network / **node-granular distribution network with DERs
attached** / full AC power flow from Phase 2.

**Selected:** The boundary given in the brief, implemented as a configurable node
graph with typed DER assets. Full topology in `docs/energy_system_spec.md` §1.

**Rationale:** Only this boundary yields a genuine *decision* problem rather than a
forecasting exercise — which the brief requires explicitly. It also makes the
listed target applications (renewable integration, grid reliability, flexibility,
curtailment, resilience) simultaneously addressable.

**Consequence:** RTS-GMLC, which the source forecasting repository used, **cannot
be this project's dataset**: it has no batteries, EVs, flexible loads or topology.
Phase 3 must not select it. No asset or node count is hard-coded anywhere.

**Status:** Accepted

---

## D-023 — Topology-aware, node-granular, physics deferred

**Context:** How faithful must the network be? This determines whether
voltage/current/thermal variables exist at all and what Phase 10 can constrain.
The brief lists voltage, current, line loading and transformer loading as
candidate grid-state variables.

**Choices:** Nodal with mandatory network physics and enforced voltage/thermal
constraints / single-bus aggregate with no network / **topology-aware and
node-granular with physics deferred**.

**Selected:** Real `Node` and `NetworkElement` entities; per-node power balance
enforced; network elements carry ratings and parameters as descriptive metadata;
voltage and thermal loading are *observation* and *constraint* categories only.
`Constraint.needs_power_flow` flags the two physics-dependent categories.

**Rationale:** Satisfies the brief's requirement that topology be a first-class,
configurable representation and that the digital twin have something to simulate,
without committing Phase 2 to a power-flow solver. Claiming enforceable voltage
constraints without a solver would be a false capability.

**Consequence:** `NetworkTopology.is_aggregate` is surfaced so a lumped system is
never mistaken for a feeder, and `configs/domain.toml` declares `topology_kind`
explicitly. Node voltage and current are **observations**, never derived values.
A single lumped node remains a valid configuration, so the same types serve both.

**Status:** Accepted

---

## D-024 — 15-minute native timestep, 96-step day-ahead horizon

**Context:** The brief forbids inventing the timestep or horizon and requires the
choice to be justified. The source forecasting repository used hourly RTS-GMLC.

**Choices:** Hourly H=24 / 5-minute H=288 / **15-minute H=96 with hourly derived** /
15-minute H=672 week-ahead.

**Selected:** `timestep_minutes = 15`, `horizon_steps = 96` (24 h), hourly
derivable by integer aggregation (4 steps/hour). Configurable, not hard-coded.

**Rationale:** Battery cycling and EV charging are only meaningfully schedulable at
sub-hourly resolution — a 1-hour battery cycle cannot express most useful
schedules, so hourly under-resolves the very flexibility this project studies. PV
ramp within an hour is materially non-representative on a distribution feeder.
15 minutes is also the resolution used by distribution operators and by the
representability anchor. The source repo's hourly choice was never constrained by
storage or EVs, because it has none.

**Consequence:** `TimeBase` is built from configuration, so 5-minute or hourly
requires no code change. The permitted timestep set is constrained to values that
aggregate cleanly. Off-grid timestamps are rejected, not rounded — silent rounding
would shift the whole system by a partial step.

**Status:** Accepted

---

## D-025 — Domain errors are collected and reported together

**Context:** Extends Phase 1's D-008 to the domain layer.

**Choices:** Fail on first error / collect all errors / log and continue.

**Selected:** `DomainValidationError` carries every violation found, in a
`DomainValidationError.errors` tuple. Every domain object validates in
`__post_init__`.

**Rationale:** Validation at construction means an invalid energy state cannot
exist, so no later phase has to defend against one. Collecting errors means a
malformed object is fixable in one pass.

**Consequence:** Construction is slightly more expensive than assignment, which is
irrelevant at this scale. Every object is frozen and slotted, so validation cannot
be bypassed by later mutation either.

**Status:** Accepted

---

## D-026 — Explicit sign conventions and unit enforcement

**Context:** In an energy system a bare float is a correctness hazard: 0.5 means
half a megawatt or half a state of charge depending on context, and confusing them
is silent.

**Choices:** Bare floats / unit-annotated floats / **`Quantity` objects carrying a
closed `Unit` enum**.

**Selected:** Every numeric domain value is a `Quantity(value, Unit)`. Units are a
closed `StrEnum`. Mixing units raises rather than coercing.

**Rationale:** Makes unit confusion a construction-time error instead of a
plausible-looking wrong number. Rejects NaN and infinity, which would otherwise
propagate silently through every downstream calculation.

**Consequence:** Phase 2 performs **no unit conversion**. Consequences that are
correct rather than convenient: `EnergySystem.controllable_capacity_kw` skips
non-kW ratings; `Battery.nameplate_duration_hours` returns `None` unless the
energy/power unit pair is dimensionally compatible (`kWh`/`kW`, `MWh`/`MW`). The
unit enum also needed `KILOVOLT` and `MEGAWATT_HOURS` added during implementation —
distribution nominal voltage is naturally in kV.

**Status:** Accepted

---

## D-027 — Typed identifiers with per-kind prefixes

**Context:** Cross-entity references must resolve. A bare string cannot prevent
`NodeId("asset-7")`.

**Choices:** Bare strings / a single `str` alias / **frozen validated classes with
per-kind prefix patterns**.

**Selected:** `AssetId`, `NodeId`, `NetworkElementId`, `ConstraintId`,
`ObjectiveId`, `ActionId`, `ForecastId`, `SystemId`, each matching
`^<prefix>-[a-z0-9][a-z0-9_-]{0,63}$`.

**Rationale:** Makes a cross-kind reference impossible to write rather than
something a later validation has to catch. Prefix patterns make serialized JSON and
logs self-describing.

**Consequence:** Adding a new entity kind requires a new identifier class. This
immediately paid off: `Constraint` and `ObservationRecord` originally stored asset
references as bare `str`, and because `AssetId` is not a `str` subclass **every
cross-reference silently failed to resolve**. Both now use typed identifiers and
the mismatch is a construction-time error.

**Status:** Accepted

---

## D-028 — Tiered per-asset control authority, carried on the action

**Context:** Does the system own the flexible assets it models? The brief's
execution loop requires an "execute" step; the source forecasting repository's
firewall says agents may never execute anything.

**Choices:** Full authority over all flexible assets / advisory only / **tiered
per-asset authority**.

**Selected:** `AuthorityLevel` of `DISPATCHABLE`, `SCHEDULEABLE`, `CURTAILABLE`,
`ADVISORY_ONLY` or `NOT_CONTROLLABLE`, declared per asset. An `Action` carries the
authority it was issued under and is rejected at construction if that authority is
not in the sufficiency set for its action type.

**Rationale:** A distribution operator genuinely cannot dispatch a customer's EV or
HVAC unilaterally, so blanket authority would be scientifically invalid for a
project claiming real energy relevance. It also makes Phase 13's adaptive autonomy
meaningful: high autonomy for self-owned assets, advisory where authority is absent.

**Sufficiency is an explicit set per action type, not a ranking.** An earlier draft
encoded the levels as a lattice with `DISPATCHABLE > SCHEDULEABLE > CURTAILABLE`,
which silently asserted that scheduling authority implies authority to curtail
generation. That implication is false — curtailment is a direct output reduction,
not a shift in time. Listing sufficiency explicitly makes each implication a stated
decision. Relatedly, `AuthorityLevel` is a `StrEnum`, so `>=` on it would order
members *alphabetically* (`"curtailable" < "scheduleable"`); a membership test is
used instead.

**Consequence:** `ADVISORY_ONLY` assets are fully modelled and reasoned about but
can never be dispatched — the domain supports "we know we could shed this load but
we have no authority to". Battery discharge requires `DISPATCHABLE`, which is the
correctly strict requirement.

**Status:** Accepted

---

## D-029 — Data quality is representable but never detected in Phase 2

**Context:** The brief asks for data-quality metadata but forbids implementing
anomaly detection.

**Choices:** Implement detection now / **define the vocabulary and invariants only**.

**Selected:** `DataQuality` with `frozenset` of flags: `ok`, `missing`, `stale`,
`outlier`, `invalid_range`, `sensor_error`, `estimated`, `interpolated`. Invariants:
≥1 flag required; `ok` cannot combine with anything; flags may co-occur.

`is_usable_for_decision` is strict — **any** adverse flag, including `stale`,
disqualifies a value from backing an action.

**Rationale:** Staleness is included deliberately: an out-of-date state of charge is
exactly the input that produces a confidently wrong dispatch, and "it was fine a few
steps ago" is not a safe basis for acting on a time-coupled system. `stale` was
initially *omitted* from the blocking set during implementation and a test caught
it — a genuine gap, not a style choice.

This is the one area where **SAT-SA is a negative reference**: it has no staleness
detector, no sensor-error flag and no estimated/imputed marker, and its own
research notes acknowledge that gap. Those are exactly the omissions worth closing.

**Consequence:** Detection is Phase 14 (Data Quality and Anomaly agents).
`is_usable_for_decision` may be relaxed per-component in Phase 13 under a tightened
assurance level; it is the strict default.

**Status:** Accepted

---

## D-030 — Constraints are declared, never enforced or formulated

**Context:** The brief forbids finalising mathematical constraints that depend on an
unselected topology, and places optimizer selection in Phase 10.

**Choices:** Write the formulation now / declare categories with no algebra /
defer entirely.

**Selected:** Twelve `ConstraintCategory` values with natural units and a
`CONSTRAINT_UNITS` mapping. `Constraint` binds them to real assets and nodes.
`Constraint.is_enforceable_in_phase_2` returns `False` for every constraint.

**Rationale:** A formulation written in Phase 2 would prejudge the Phase 10
optimizer choice, which the brief explicitly forbids. Declaring the categories
still constrains later work: a constraint naming a phantom resource is rejected
here rather than surfacing later as a wrongly-feasible optimisation.

**Consequence:** No constraint is enforced in Phase 2 and nothing pretends
otherwise. The one invariant Phase 2 *does* enforce is per-node power balance,
because it is structural rather than an optimisation choice (D-031).

**Status:** Accepted

---

## D-031 — Per-node power balance is enforced structurally

**Context:** The chosen topology is node-granular (D-023). What must hold at a node?

**Selected:** `generation_kw + storage_kw + grid_kw − load_kw + unaccounted_kw ≈ 0`
per node, within `POWER_BALANCE_TOLERANCE` = 0.5 kW. Sign convention: generation
and grid import positive, load and grid export negative, storage a signed
injection. `unaccounted_kw` records an explicit residual rather than hiding it.

**Rationale:** This is what distinguishes a distribution model from a lumped one,
and it catches the failure modes that would otherwise be invisible: a sign error, a
double count, a missing asset. 0.5 kW is generous enough for unmodelled feeder
losses while still catching those.

**Consequence:** `EnergyState` construction fails on an unbalanced node — the check
is in `__post_init__`, not deferred to a validation pass. Phase 11 may tighten the
tolerance once a loss model exists. `EnergySystem.validate_state` additionally
requires the state to cover *exactly* the system's nodes and to report no asset the
system does not own as that asset type.

**Status:** Accepted

---

## D-032 — Provenance is mandatory on every domain object

**Context:** The brief requires traceability and cites SAT-SA's principle that data
must have provenance rather than appearing magically inside the system.

**Choices:** Optional provenance / mandatory `Provenance` / full provenance ledger
with content digests.

**Selected:** A mandatory `Provenance` on every observation, state, forecast,
action, constraint and objective. `Provenance` is unconstructible without a
`SourceReference` and at least one `ProvenanceEvent`.

**Rationale:** Mandatory-by-construction is what stops provenance silently
disappearing downstream, which is the specific failure mode SAT-SA documented and
guarded against. The full ledger machinery SAT-SA uses (hash chaining, cryptographic
signing, canonical-form modules) is deliberately **not** reproduced: Phase 2 adopts
the contract, not the crypto. Whether later phases add digests is their decision.

**Consequence:** `SourceReference.checksum` is optional for now (no dataset
ingested yet) and should become mandatory in Phase 3. `ProcessingStep.version` is
required so a result stays reproducible after code changes.

**Status:** Accepted

---

## D-033 — Uncertainty is a container with the mathematics deferred

**Context:** Uncertainty is first-class in the project's architecture, and no source
repository models it at all.

**Choices:** Ignore uncertainty until Phase 8 / model it now / **define the
container, defer the method**.

**Selected:** `UncertaintyEstimate` with `kind` (measurement / model / parametric /
combined / **unknown**), `unit`, `method` defaulting to `"TBD"`, and exactly one of
`lower_bound` / `upper_bound` / `standard_deviation` / `relative_std`.

**Rationale:** `unknown` is a legal kind so a model can honestly declare that
uncertainty is uncharacterised rather than omitting the field or defaulting to
zero — the same "None means not-applicable, not zero" discipline SAT-SA uses in
`ConfidenceVector`. Supplying more than one numeric summary is rejected because how
an interval relates to a standard deviation depends on a distributional assumption
Phase 8 has not made.

**Consequence:** `Forecast` carries uncertainty optionally and reports
`is_quantified_uncertainty`, so a consumer can refuse to act on an unquantified
forecast. `Forecast` rejects `issued_at >= target_time`, because a record stamped at
or after the moment it describes is hindsight and would contaminate evaluation.

**Status:** Accepted

---

## D-034 — Action/observation separation enforced at construction

**Context:** The brief requires every variable to be classified and forbids the same
variable becoming both an uncontrolled observation and an action.

**Choices:** Convention only / runtime validation / **construction-time rejection in
both directions**.

**Selected:** `VariableRole` (`OBSERVATION`, `ACTION`, `DERIVED`, `CONSTRAINT`,
`OBJECTIVE`). `ObservationRecord` accepts only `OBSERVATION`/`DERIVED`.
`Action` accepts only `ACTION`. `Forecast` accepts only `OBSERVATION`.

**Rationale:** A mix-up here is silent and severe — commanding a battery by setting
its state of charge, or treating a setpoint as a measurement. Enforcing it at
construction means the error cannot exist.

**Consequence:** `ObservationRecord` must be scoped to an asset or node; an
unscoped observation cannot be attributed and is rejected. Adding a new role
requires deciding which containers accept it.

**Status:** Accepted

---

## D-035 — `StateOrigin` makes observed / simulated / predicted distinct

**Context:** The brief requires these to be non-interchangeable so the twin can be
compared against reality.

**Selected:** Mandatory `StateOrigin` on `EnergyState` with `OBSERVED`, `SIMULATED`,
`PREDICTED`, `ASSIMILATED`. Never inferred.

**Rationale:** The project's central research signal is comparing a learned model's
prediction against a simulation against reality. With origin implicit that
comparison is meaningless.

**Consequence:** The same `EnergyState` type carries all four origins, so comparing
observed against simulated needs no change to the representation — only a check
that the origins differ.

**Status:** Accepted

---

## D-036 — Deterministic versioned JSON serialization, no database

**Context:** The brief requires serializable domain objects that are deterministic,
explicitly schematised, free of hidden fields, round-trippable and versionable, and
forbids introducing a database without a requirement.

**Choices:** Database / pickle / **deterministic JSON with an explicit schema
version**.

**Selected:** `encode()` emits compact, key-sorted JSON with a
`_schema_version` field (`"2.0.0-phase2"`); `decode()` refuses an unknown version.
Encoding walks `dataclasses.fields`, so the emitted key set *is* the field set —
no hidden attribute can be smuggled through.

**Rationale:** JSON is inspectable, dependency-free and diffable in git, which
matters when reproducibility depends on seeing exactly what changed. Hand-written
decoders keep the schema explicit rather than inferred from annotations.

**Consequence:** Round-trip and determinism are enforced by tests for twelve
concrete object kinds, not assumed. Dispatch in `from_dict` is by discriminating
field and **order matters**: an asset payload also carries `node_id`, so the asset
check must precede the bare-node check — this was a real bug found during
implementation. `_schema_version` must be bumped when the shape changes.

**Status:** Accepted

---

## D-037 — Domain configuration is a separate module, not a wider `AppConfig`

**Context:** The brief permits extending the Phase 1 configuration only where
required, with categories `energy_system`, `assets`, `time`, `data`, `simulation`.

**Choices:** Widen `AppConfig` with nested tables / **a separate `config/domain.py`
and `configs/domain.toml`**.

**Selected:** `DomainConfig` with `TimeConfig`, `EnergySystemConfig` and
`data_anchor`, loaded by `load_domain_config()`. `AppConfig` is untouched.

**Rationale:** `AppConfig` describes *a run* (seed, device, paths, logging); domain
config describes *the environment under study*. Keeping them apart means the 160
Phase 1 tests remain meaningful and Phase 3 can load a dataset config without a run
config and vice versa. Only fields Phase 2 actually uses are present — no dataset
paths, no model paths, no simulation settings.

**Consequence:** Two config files and two loaders, which is a small cost for a
clear separation of concerns. `data_anchor` must be written **before** the first
`[table]` header in TOML, since a key after a header belongs to that table — an
error this project made once and has tests for. The permitted timestep set is
constrained to `{1, 5, 15, 30, 60}` so coarser views aggregate by integer.

**Status:** Accepted

---

## D-038 — SMART-DS as representability anchor, model stays dataset-agnostic

**Context:** The brief forbids inventing a dataset and asks that the Phase 2 model be
checked for representability.

**Choices:** RTS-GMLC to match the source repo / IEEE PES test feeders / no anchor /
**SMART-DS as a representability anchor with the model left dataset-agnostic**.

**Selected:** `data_anchor = "smart-ds"` in `configs/domain.toml`, recorded as a
*representability anchor*, not a download instruction. Phase 3 selects and fetches.

**Rationale:** SMART-DS is the only open dataset supporting **both** topology
(substations, feeders, lines, transformers, regulators in OpenDSS/CYME) and the full
DER action space (PV, storage, EV charging, demand response, ZIP loads, 15-minute
profiles). That makes the Phase 2 domain model falsifiable rather than abstract.
Its stated limitation — timeseries battery dispatch is not included — must be
handled by Phase 3, likely by deriving dispatch from state of charge.

**Consequence:** RTS-GMLC is ruled out (D-022). The anchor is recorded so a later
claim of representability is traceable rather than assumed, and so a future
representability test has something to test against.

**Status:** Accepted

---

## D-039 — No unit conversion and no aggregated energy/power helper

**Context:** Should the domain convert between units?

**Selected:** No conversion. `Quantity.require_same_unit` raises on mismatch.
`Battery.nameplate_duration_hours` uses an explicit dimensionally-compatible pair
table (`kWh`/`kW`, `MWh`/`MW`) and returns `None` otherwise.

**Rationale:** Silent conversion in a data model hides the unit decisions that later
phases legitimately make differently (per-unit tariffs, emissions factors). Failing
loudly at the point of use is better than a wrong number.

**Consequence:** Some helpers decline rather than guess:
`EnergySystem.controllable_capacity_kw` skips non-kW ratings. This is why an
implementation bug had to be caught by a test — the first version compared
`usable_energy.unit != rated_power.unit`, which is *always* true (`kWh` is never
equal to `kW`), making the property dead code.

**Status:** Accepted
