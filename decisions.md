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

---

# Phase 3 — SMART-DS ingestion, normalization and reality validation (D-040 … D-048)

Phase 2 defined the domain contract as a hypothesis. Phase 3 tested that
hypothesis against real data and recorded what broke.

## Source-research findings driving these decisions

Everything below was established by querying the authoritative source and
reading the real files. Nothing is taken on trust.

| Finding | Evidence |
|---|---|
| SMART-DS v1.0 is on the OEDI data lake as a **public, unauthenticated S3 bucket** | `https://oedi-data-lake.s3.amazonaws.com/?list-type=2&prefix=SMART-DS/` lists `v0.9/` and `v1.0/` |
| Version is **1.0** | `SMART-DS_version.txt` inside a scenario folder contains exactly `1.0` |
| **Native resolution is 15 minutes**, matching the Phase 2 grid exactly | Confirmed three ways: `Master.dss` `Solve mode=yearly stepsize=15m number=35040`; `LoadShapes` `npts=35040 interval=0.25`; profile files hold exactly 35040 values |
| Load power is `kW(t) = rating x profile_pu(t)`, profiles being **per-unit** | User Guide rule, then verified: residential reconstructed 12380.5126 kW vs 12380.5126 kW published, **error 0.0000 kW** |
| Centre-tap loads must be **summed**, not de-duplicated | Dedup gave -39.3%; summing gave -0.14%. All 1819 pairs carry identical kW |
| Battery dispatch **does not exist and is underivable** | 93/93 Storage objects have `State=IDLING`, one `kWhStored` scalar, no SOC series, no storage loadshape |
| EV exists as **placements only** | User Guide says "charging locations"; no EV timeseries folder exists |
| **No wind** at all | No wind folder; `metrics.csv` has PV and Battery columns only |
| Feeder losses are **3.2175 %** | `Summary_data.csv`: 501.672 kW losses at 15090.718 kW customer power |
| `Buscoords.dss` is **not** a `.dss` directive file | Its content is bare `<bus> <lon> <lat>` lines; the generic parser yields **zero** elements |
| Percent-prefixed parameters exist | `%Cutout`, `%Cutin`, `%EffCharge`, `%EffDischarge` on every PVSystem and Storage |

## D-040 — Acquire from the authoritative OEDI bucket, not a mirror

**Choices:** Kaggle or community mirror / an archived subset / **the OEDI S3
bucket over plain HTTPS**.

**Selected:** `https://oedi-data-lake.s3.amazonaws.com/SMART-DS/v1.0/`, fetched
with plain HTTPS GET and the S3 `list-type=2` API.

**Rationale:** the source is public and unauthenticated, so a mirror adds no
convenience and introduces an untrustworthy provenance chain. No S3 client
library was installed, preserving D-004.

**Consequence:** acquisition needs no credentials, so the manifest has **no
secrets to record** (asserted by a test). Licence is recorded as `UNKNOWN`
because the User Guide does not state it, and must be resolved before
publication (G-11).

**Status:** Accepted

---

## D-041 — Mirror the authoritative layout under `data/raw/smart_ds`

**Choices:** flatten the acquired files / store them in an archive /
**mirror the bucket layout at the same relative paths**.

**Selected:** `data/raw/smart_ds/v1.0/2018/AUS/P1U/...`, mirroring the bucket.

**Rationale:** a provenance locator that reads like the authoritative path is
auditable against the source without a lookup table, and an unchanged upstream
layout means a future file can be dropped in without touching the adapter.

**Consequence:** raw data stays git-ignored (D-016); 228.5 MiB is never
committed. Re-fetching plus digest verification is the reproducibility contract.

**Status:** Accepted

---

## D-042 — A checksummed manifest, with acquisition time isolated

**Choices:** record nothing / record everything including wall-clock time /
**a manifest whose content digest excludes acquisition time**.

**Selected:** `DatasetManifest` recording S3 key, size and a **computed** SHA-256
per file, with `acquisition.acquired_at` isolated in `AcquisitionMetadata`.

**Rationale:** reproducibility needs "which bytes produced this result", but two
acquisitions of identical bytes must compare equal. Separating the one
nondeterministic field makes `manifest.digest()` a pure function of the content.

**Consequence:** a test asserts the digest is unchanged when only
`acquired_at` differs. No checksum is *claimed* unless calculated or published.

**Status:** Accepted

---

## D-043 — Verify the derivation empirically, not from documentation

**Context:** the User Guide's centre-tap wording is easy to misread, and the
suggested "multiply by 0.5" applies to driving an OpenDSS solve, not to
reconstructing customer demand.

**Choices:** follow the documentation literally / follow the parenthetical "half
of the total" wording / **test both readings against the dataset's own published
figures and adopt whichever reproduces them**.

**Selected:** sum both members of each `_1`/`_2` pair. Dedup was 39.3 % low;
summing matched residential to eight decimal places.

**Rationale:** the dataset publishes its own peak-time customer power, giving an
independent oracle. Where documentation and data disagree, the data decides.

**Consequence:** the mapping record cites the numeric evidence rather than the
prose, and `test_centre_tap_mapping_records_the_empirical_evidence` fails if
that evidence is removed.

**Status:** Accepted

---

## D-044 — No resampling: the native grid already matches the domain

**Choices:** resample 15 min to 15 min anyway / resample to hourly and
re-derive / **use the native grid unchanged**.

**Selected:** native 15-minute, 35,040-point grid used directly. No resampling.

**Rationale:** confirmed three independent ways. Phase 2 chose 15 minutes
*because* battery, EV and PV ramp are not meaningfully schedulable at hourly
resolution (D-024); SMART-DS happens to be 15-minute native, so the two align
and no information is lost.

**Consequence:** `TimeAxis.matches_domain_grid` is True and is asserted.
Resampling rules for a future coarser or irregular dataset belong to whichever
phase first needs them; Phase 3 defines none because none is justified.

**Status:** Accepted

---

## D-045 — Timestamps: anchor the synthetic clock to UTC, explicitly

**Context:** SMART-DS carries **no timezone** — no offset, no local time, no
daylight-saving handling. Phase 2 rejects naive datetimes. Both requirements
cannot be satisfied without a choice.

**Choices:** store naive timestamps (violates the domain) / invent a local
timezone (would be wrong and unverifiable) / **anchor to
`<year>-01-01T00:00:00+00:00` and treat as UTC, recording the assumption**.

**Selected:** the third, with `TIMEZONE_ASSUMPTION` exported, embedded in the
time-axis description and on every provenance chain.

**Rationale:** the dataset is internally consistent on one synthetic clock, so
anchoring it anywhere preserves every **relative** time relationship — which is
what forecasting, flexibility and balance validation depend on.

**Consequence:** true local wall-clock time is **not** preserved (Austin is
CST/CDT). Recorded as G-09. Any future claim about local solar noon or tariff
windows is valid only up to that fixed offset.

**Status:** Accepted

---

## D-046 — Battery dispatch is UNAVAILABLE, and no `StorageState` is built

**Context:** Phase 2's handoff stated dispatch "may need to be derived from state
of charge". The Phase 3 brief required verifying that claim rather than
assuming it.

**Choices:** derive dispatch from a chosen SOC trajectory (fabrication) / run
OpenDSS with a storage control strategy (produces *our* dispatch, not the
dataset's) / **report UNAVAILABLE and construct no `StorageState`**.

**Selected:** the third.

**Rationale:** verification showed the suggested route is itself unavailable —
deriving power needs SOC(t) and SOC(t+1), and SMART-DS supplies a *single*
`kWhStored` scalar. There is nothing to difference. Producing any dispatch series
would have meant inventing it, which rules A, C and D all forbid.

**Consequence:** battery *ratings* are mapped (`DIRECT`), but
`storage.charge_power_kw`/`discharge_power_kw`/`state_of_charge` series are
`UNKNOWN`. Recorded as G-01, and the Phase 2 handoff statement is corrected.
Battery flexibility must come from capacity (Phase 9) and from explicitly
`SIMULATED` augmentation if desired.

**Status:** Accepted

---

## D-047 — Report the 0.5 kW failure; do not adjust the tolerance

**Context:** Phase 2 set a 0.5 kW per-node target (D-031). Real data does not meet
it.

**Choices:** widen the tolerance to pass / drop failing nodes / smooth the data /
**report the failure with its cause**.

**Selected:** the fourth. `TOLERANCE_KW` is a module constant with no setter, and
a test asserts it equals 0.5.

**Rationale:** the failure is informative. Two distinct causes were identified
with evidence:

1. **Real distribution losses** (501.67 kW, 3.2175 %) that Phase 2 deferred
   physics for (D-023). A physics gap, not an ingestion error.
2. **An unexplained 20.45 kW commercial reconstruction shortfall**, cause
   `UNKNOWN`, recorded as G-07.

Residential demand matches the published value exactly, which isolates the
problem to the commercial subset.

**Consequence:** the balance report distinguishes *reconstruction fidelity*
(our ingestion) from *physical closure* (the network's losses), because conflating
them would hide which one is failing. Per-node balance over time is reported as
`NOT EVALUATED` with the reason, since grid flow is absent (G-02). `energy-intel
data validate` exits non-zero on failure, which is a legitimate reportable
outcome, not a crash.

**Status:** Accepted

---

## D-048 — Fixtures extracted mechanically from real files

**Context:** the brief forbids requiring the full dataset for unit tests, and
forbids fictitious schemas.

**Choices:** hand-write miniature SMART-DS files / use the full dataset for
every test / **extract verbatim excerpts with a script and document each
truncation**.

**Selected:** `scripts/extract_fixtures.py` cuts real excerpts into
`tests/data/fixtures/`, writing `FIXTURE_PROVENANCE.json` that names every file
and every truncation.

**Rationale:** a hand-written fixture is a guess at SMART-DS's format, and a
guess is exactly what Phase 3 exists to eliminate. Two fixtures are truncated and
say so: the profile fixture is a genuine 96-value prefix (so its maximum is
**not** 1.0, and a test asserts it is not normalised rather than pretending), and
the placement JSON keeps 2 of 95 real keys.

**Consequence:** unit tests run on a clean checkout with no dataset;
integration tests use the full dataset and skip when it is absent
(`real_dataset_available`). Truncations are documented rather than hidden.

**Status:** Accepted

---

## D-049 — Parser bug found by real data: percent-prefixed parameters

**Context:** OpenDSS uses percent-prefixed parameter names (`%Cutout`,
`%Cutin`, `%EffCharge`, `%EffDischarge`). The first parser pattern required a key
to start with a letter or underscore.

**Consequence:** every one of the 1,216 `PVSystem` and 93 `Storage` objects had
its `Pmpp`, `kWhStored` and efficiency values silently corrupted by swallowing
the following parameters into the previous value — for example
`Pmpp` = `'5.0 %Cutout=0.1 %Cutin=0.1'`. The failure was silent: the value was a
string that happened to parse as far as the caller.

Fixed by allowing an optional leading `%`. Regression-tested by
`test_actual_full_scenario_asset_counts`, which asserts `%EffCharge` is `95.0` on
real files.

**This is the clearest argument in the project for testing against real data
rather than hand-written fixtures**: the fixture excerpt would have had to
contain the bug for a unit test to catch it.

**Status:** Accepted (fixed)

---

## D-050 - Unknown is representable: nameplate is not usable capacity

**Context:** Phase 2's `Battery` required `usable_energy`, `min_soc`, `max_soc`
and `round_trip_efficiency`, and treated `kWhRated` as `usable_energy` when
mapping SMART-DS. Those are different quantities. A nameplate rating is what the
manufacturer fitted; usable energy additionally needs a depth-of-discharge limit,
which SMART-DS never states. Mapping one to the other inflates flexibility, and
mapping a single `kWRated` to *both* `max_charge_power` and `max_discharge_power`
invents a directional split the dataset does not contain.

**Alternatives:**

1. Keep the required fields and fill them with plausible values - rejected. This
   is the invention the Phase 3 brief forbids, and it is invisible downstream.
2. Omit batteries from the domain object - rejected. The batteries exist; dropping
   93 real assets loses information the dataset genuinely provides.
3. Widen the domain so a quantity the source does not state can be `None`.

**Decision:** option 3. `usable_energy`, `min_soc`, `max_soc` and
`round_trip_efficiency` may be `None`, meaning *not known*; a new
`nameplate_energy` carries what the source actually stated. `None` never means
zero. `capacity_known` and `usable_soc_window` report the difference rather than
letting a caller infer it from a bare `None`.

The fields keep their position and arity, so every Phase 2 call site and test is
unaffected, and `SCHEMA_VERSION` moves `2.0.0-phase2 -> 2.1.0-phase3` because the
emitted shape changed. A 2.0.0 payload still decodes.

**Consequence:** Phase 9 must obtain usable capacity from another source, or
declare an assumption with a decision of its own. Until then no battery
flexibility can be computed, which is now visible in the data rather than hidden
inside a plausible number.

**Status:** Accepted

---

## D-051 - The point of common coupling is read, never guessed

**Context:** `NetworkTopology` requires exactly one grid-connection node. The
first implementation derived it from the feeder *folder name*
(`p1uhs0_1247--p1udt12703` -> `p1udt12703`) and, when that bus was not in the
discovered graph, fell back to marking the lexicographically first node as the
PCC.

Both are wrong. `Master.dss` declares the source bus directly:
`New Circuit.feeder_p1udt12703-p1uhs0_1247x bus1=p1udt12703-p1uhs0_1247x`. A
folder name is a label; a bus that happens to sort first is not a point of common
coupling.

**Decision:** read `Circuit.bus1`. If it is absent, or absent from the discovered
graph, raise `UnresolvedBoundaryError` instead of constructing a system. A
boundary the data does not state cannot be represented honestly, and failing loudly
beats a plausible-looking wrong answer.

The fixture extraction now appends the real line incident to that source bus, so
the truncated excerpt has a resolvable boundary and the mapper's success path is
testable without the 228 MiB dataset.

**Consequence:** one honest failure mode instead of one silent wrong answer. A
future feeder whose `Master.dss` lacks `bus1` will fail loudly at ingestion, which
is the correct place to find that out.

**Status:** Accepted

---

## D-052 - Customers are mapped individually, not aggregated

**Context:** the first mapper created a single `FlexibleLoad` asset with
`asset_type=LOAD` and the summed rated power of all 1,871 customers. That discards
per-customer ratings and connection buses, and it declared flexibility fields
(`min_power`, `max_power`, `shiftable_energy`, `priority`, `is_hvac`) that
SMART-DS never states. It also used the wrong class: `AssetType.LOAD` is
implemented by the base `Asset`, so a `FlexibleLoad` carrying `asset_type=LOAD`
would not have survived a serialization round trip.

**Decision:** one `Asset` per customer, attached to its own connection bus, with
its members and centre-tap status in metadata and an explicit
`flexibility: NOT STATED` note. Because the base `Asset` carries no flexibility
fields, none is asserted.

**Consequence:** 1,871 load assets instead of 1, so `energy_system.json` grows and
the asset catalogue no longer collapses the network's load structure. Phase 9 and
Phase 10 get the granularity they need without re-deriving it from the raw files.

**Status:** Accepted

---

## D-053 - Every discovered element bus becomes a node

**Context:** `adapter.topology()` collected buses from `Lines.dss` and `Loads.dss`
only. A `Storage` or `PVSystem` on a bus with no incident line therefore produced
an asset referencing a node the topology never declared, which `EnergySystem`
rejects at construction. It surfaced as a test failure with a *different scenario's*
`Storage.dss`, which is exactly the case the real PV and battery scenarios
reproduce.

**Decision:** collect `bus1` from every discovered element - loads, PV systems and
storage - so the graph covers every asset point.

**Consequence:** node counts can exceed line endpoints, which is correct: a device
behind a service transformer is still a point in the network. Line-element count
stays the measure of graph edges.

**Status:** Accepted

---

## D-054 - Ingestion evidence, not ingestion convenience

**Context:** Phase 3 delivered a manifest, a mapping table, a quality report, a
balance report and an instantiated `EnergySystem`, all reproducible from
`data/raw/smart_ds`. None of that is load-bearing on its own: without a recorded
provenance chain, a later phase cannot tell which bytes produced a number.

**Decision:** provenance is emitted, not implied. Every mapped object carries a
`SourceReference` to the real SMART-DS path plus a `ProcessingStep` naming the
transform. The reconstructed demand is emitted as an `ObservationRecord` with
`VariableRole.DERIVED`, its real timestamp and its quality flags; the dataset's own
published figure is emitted separately as `VariableRole.OBSERVATION`. Manifest
remote keys carry the full `SMART-DS/<version>/` prefix and the recorded source is
the authoritative URL, because a locator that cannot be fetched is not a locator.

**Consequence:** `MappingReport` grew an `unknown_fields` list naming domain fields
the source does not state, and `DomainMappingResult` now carries the observations.
A reader of `artifacts/data/` can reconstruct which claims are measured, which are
derived, and which are absent.

**Status:** Accepted

---

## D-055 - Phase 4 adds exactly three runtime dependencies

**Context:** D-004 and D-005 committed to a standard-library-only project, and a
Phase 1 test enforced `dependencies == []`. Phase 4 is the first phase that
genuinely needs a numerical stack, so the commitment has to be revisited rather
than quietly broken.

**Alternatives:**

1. Keep zero dependencies and implement arrays, a linear solver, tree boosting and
   gradient descent by hand. Rejected: reimplementing a gradient-boosted tree is
   months of work and would be less trustworthy than a maintained one.
2. Add a full data-science stack. Rejected: pandas, matplotlib and joblib would be
   added "just in case". The whole dataset here is a numeric matrix, so pandas buys
   nothing, and no figure is produced by this phase.
3. Add the smallest set that covers what is actually used.

**Decision:** option 3 - `numpy`, `scikit-learn`, `torch` (CPU build only). The
Phase 1 test is **kept and tightened** rather than deleted: it now asserts the exact
set, and two new tests assert that pandas is never imported and that
`energy_intelligence.domain` imports no ML library at all, so the Phase 2 contract
stays a dependency-free vocabulary.

The CPU index is declared explicitly in `pyproject.toml` (`pytorch-cpu`). The
default Windows wheel bundles CUDA and is several hundred megabytes; a hackathon
prototype on CPU does not need it, and the smaller wheel installed in 53 s.

**Consequence:** `uv sync` now pulls ~250 MB of wheels. The Phase 2 guarantee that
the domain layer is framework-free is now actively tested rather than merely
intended.

**Status:** Accepted

---

## D-056 - Baselines are evaluated, not tuned

**Context:** it is possible to make a baseline look good by searching
hyperparameters until it wins, at which point it is no longer a baseline and the
comparison with a future Energy-MoE becomes meaningless.

**Decision:** hyperparameters are fixed at stated defaults, declared in the model
module so the registry records them, and recorded verbatim in
`experiments/registry.jsonl`. Ridge uses scikit-learn's own `alpha=1.0`. Gradient
boosting uses `max_iter=200`, `early_stopping=False` - disabled deliberately, so
training duration is deterministic and does not depend on the split. The neural MLP
uses two small hidden layers, 40 epochs, L1 loss.

**Consequence:** the neural baseline's failure at the 24-hour horizon (below) is
reported as measured. Tuning it would have produced a better number and a worse
experiment.

**Status:** Accepted

---

## D-057 - The adapter gains a customer filter; the feeder total is accumulated per profile

**Context:** Phase 4 needs whole-year series for a declared subset of customers.
`SmartDsAdapter.loads()` had no way to select them, and Phase 4 must not
re-implement SMART-DS semantics - that is exactly what Phase 3 established.

**Decision (two parts):**

1. `loads()` gains an optional `names` filter. It selects customers and changes no
   arithmetic, so the derivation remains single-sourced.
2. The feeder total is accumulated **per profile** rather than per customer: 341
   profile files, each weighted by the summed kW of every Load object that uses it.
   Materialising 1,871 whole-year series costs about 525 MB of tuples; the profile
   accumulation costs about 50 MB and takes the same time.

**The evidence that part 2 is correct:** at the published peak (grid index 19553)
the accumulated feeder series equals **15070.267606731577 kW**, while Phase 3's
per-customer sum gives **15070.267606731582 kW** - agreement to 5e-12 kW. The test
`test_feeder_extraction_matches_the_phase3_reconstruction` asserts this, so a future
divergence between the two paths is a test failure rather than a quiet divergence.

**Consequence:** cross-phase consistency is now machine-checked.

**Status:** Accepted

---

## D-058 - Target selection: three tasks, four refusals

**Context:** the phase brief listed load, solar, wind, net demand, battery state
and EV demand as candidates. Only three are real in SMART-DS.

**Decision:** `ml/targets.py` declares every candidate with the fields the brief
requires (source, unit, resolution, coverage, missingness, granularity,
feasibility, status, evidence) and the runner **refuses** an `UNSUPPORTED` target by
name, with the evidence, rather than producing an empty dataset.

| Target | Status | Basis |
|---|---|---|
| `customer_load` | SUPPORTED | 1,871 customers x 35,040 points, no missing values |
| `feeder_load` | SUPPORTED | sum over all 3,690 Load objects |
| `pv_generation` | PARTIALLY_SUPPORTED | one shipped 1000 kW array; **not** the feeder PV fleet (G-03) |
| `net_load` | UNSUPPORTED | needs feeder PV, whose irradiance shapes are absent |
| `battery_soc`, `battery_power` | UNSUPPORTED | no SOC(t) exists at all (G-01) |
| `ev_demand` | UNSUPPORTED | 2,991 adoption sites, no charging series (G-04) |
| `wind_generation` | UNSUPPORTED | SMART-DS v1.0 has no wind asset (G-05) |

`pv_generation` is `PARTIALLY_SUPPORTED` rather than `SUPPORTED` because honesty
demands the distinction: it is one real array at one real location, while the 1,216
feeder PV systems reference 20 irradiance loadshapes
(`AUS_30.4196_-97.8095_<tilt>_<azimuth>`) that **are not present anywhere in the
dataset**, and the one `solar_data` file that is present is a different location
(30.3459 N). Feeder PV output remains underivable (G-03).

**Consequence:** three forecasting tasks, on three declared targets, with four
absences recorded rather than filled.

**Status:** Accepted

---

## D-059 - The experiment shape is a fourth configuration file

**Context:** `configs/default.toml` describes a run, `configs/domain.toml` the
environment, `configs/data.toml` the dataset. Phase 4 needed a fourth thing: which
target, lookback, horizons, split and baselines.

**Decision:** `configs/ml.toml`, following D-037's separation. The dataset selection
is a fact about what was downloaded; the lookback and horizon are choices about a
question. They change for different reasons and at different times, so conflating
them would mean editing the acquisition record to change an experiment.
Unknown keys are rejected like the other three.

Two further configs exist because the PV experiments differ from each other in a way
that **is** the scientific question: `configs/ml_pv_nowcast.toml` (15 minutes, real
weather enabled) and `configs/ml_pv_horizon.toml` (1/2/4/96 steps, weather off).

**Status:** Accepted

---

## D-060 - Lookback: 672 steps (7 days)

**Context:** 1 hour, 6 hours, 24 hours and 7 days were all defensible.

**Decision:** 672 steps = 7 days, because it is the shortest window containing a
full weekly cycle, and the dataset demonstrably has weekly structure: the
seasonal-naive-week rule reaches MAE 1.83 kW at a 24-hour horizon where the
seasonal-naive-day rule reaches 2.43 kW. Without a week of history a model cannot
express "same time last week", which is the strongest seasonal signal here.

A 96-step lookback was tested and is insufficient: `lag_672` and
`roll_mean_same_hour_7d` are withdrawn by the feature builder, with the reason
recorded in the dataset manifest. The withdrawal mechanism is itself tested.

**Computational effect:** 672 steps is the dominant memory term of a window. It is
affordable here only because the dataset is stored as a flat feature matrix rather
than as dense windows (see D-065).

**Status:** Accepted

---

## D-061 - Horizons: 15 minutes, 1 hour and 24 hours; split chronologically

**Context:** the brief asks for a horizon that the data supports, that is useful to
the eventual decision problem, that can feed optimization, and that is practical.

**Decision:** horizons of 1, 4 and 96 steps (15 min, 1 h, 24 h). 15 minutes and
1 hour serve nowcasting and intraday control; 24 hours is the horizon an
optimiser actually needs, because a day-ahead dispatch decision is made at one
horizon. All three are supported by the native 15-minute grid with **no
resampling** (D-044).

The split is **chronological, contiguous, by grid index**, at 70 / 15 / 15. The
test span is the last 15% of the year, roughly 54 days, and it contains the
dataset's own published peak timepoint (grid index 19553) - so peak-period error is
measured on data no model has seen. Nothing is shuffled anywhere in the pipeline.

Ramp error is reported as `n/a` unless two **consecutive** horizons are declared,
because a ramp is a change between consecutive steps and computing it across a
3-step gap would report a number that is not a ramp. The PV horizon config declares
1 and 2 precisely so the metric exists there.

**Status:** Accepted

---

## D-062 - Scale: a stratified 40-customer sample, stride 1

**Context:** all 1,871 customers for a full year is 65.6M values. The full matrix is
a scale problem, not a baseline problem.

**Decision:** a **stratified** sample of 40 customers (25% commercial, against a
population of 34 commercial out of 1,871, so the minority class is deliberately
over-represented), drawn deterministically from `numpy.default_rng(seed)` across
rated-power deciles, with the composition recorded in the dataset manifest. Sampling
the first 40 names in sorted order would have taken a contiguous run of one
neighbourhood and biased both class mix and magnitude.

Stride is 1: every 15-minute origin is kept, giving 1,309,440 samples. Stride was
available for a cheaper run and is recorded in the dataset version, so a strided
dataset is a *different version*, never a silent substitution.

**Consequence:** the load result is a statement about 40 real customers plus the
feeder aggregate, not about all 1,871. Widening it is a configuration change, and
the configuration is versioned.

**Status:** Accepted

---

## D-063 - Per-unit normalisation by rated kW, and weather withdrawn beyond 15 minutes

**Context:** two problems, both found by measurement rather than anticipated.

1. Customer ratings span 1.37 kW to 387.6 kW. A single pooled model minimising
   squared error is dominated by the largest series, and at the first attempt
   classical ML was *worse* than the mean predictor (R2 -0.59) while persistence
   scored 0.27 kW.
2. The solar file ships **actual** weather observations. Using them at a 4-hour
   horizon means using the future, because no weather forecast exists in the data.

**Decision (1):** the dataset stores every target and lag **per unit of the
series' own scale**, which is the series' rated kW for load and the 1000 kW array
rating for PV. This is a physical quantity known for the whole forecast horizon,
not a fitted statistic, so it leaks nothing - and it is how SMART-DS itself stores
its profiles. Metrics are converted back to kW for reporting, so both numbers are
available. Without it, the classical baselines were reading a scale gap as signal.

**Decision (2):** weather features are withdrawn automatically whenever any declared
horizon exceeds one step, and the withdrawal **and its reason** are recorded in the
dataset manifest. `WEATHER_AVAILABLE_THROUGH_STEP = 1`. This is why the PV task
exists in two configurations: a nowcast with weather, and a multi-horizon run
without it.

**Status:** Accepted

---

## D-064 - The feature set must include what the baseline sees

**Context:** the first feature set contained `lag_1 = y[t-1]` but **not** `y[t]`. The
persistence baseline reads `y[t]`. A model that cannot see the current measurement
starts from a worse position than the baseline it is measured against - the
comparison was rigged in the baseline's favour, and it showed: ridge scored 0.48 kW
against persistence's 0.27 kW at a 15-minute horizon.

**Decision:** `value_at_origin = y[t]` is a catalogue feature, with the reason
recorded in the feature spec itself. After adding it, classical ML beat persistence
where it genuinely can (24-hour horizon) and lost where persistence is genuinely
near-optimal (15 minutes). The leakage guard was adjusted in the same change: it
poisons strictly **after** the origin, because `y[t]` is the current reading and is
knowable at prediction time.

**Status:** Accepted

---

## D-065 - Flat feature matrices, not nested windows

**Context:** the brief asks for a representation that is "easy to test and later
transform into model tensors".

**Decision:** one flat row per `(series, forecast origin)`: `features (n, f)`,
`targets (n, horizons)`, plus `origin_index`, `series_index` and `split` arrays. A
dense `(n, lookback)` window tensor for this dataset would be gigabytes, could not be
`assert`-ed against, and would hard-code one model's input format.

**Consequence:** every temporal invariant is directly checkable, and the GRU
baseline reassembles its sequence from the lag columns it is entitled to.

**Status:** Accepted

---

## D-066 - Split-dependent features are created at fit time, not stored in the dataset

**Context:** a pooled model has no way to tell a 1.4 kW household from a 388 kW shop,
so calendar terms end up absorbing cross-customer differences.

**Decision:** three per-series context columns (`series_index`, `series_level`,
`series_volatility`) are computed **inside the experiment**, from the training rows
only, and are never written into the dataset artifact. A dataset containing
split-dependent columns could not be reused under a different split, which would
defeat the purpose of versioning it.

The feature scaler is handled the same way: `FeatureScaler` records how many rows it
was fitted on, and `assert_train_only_fit` raises if that is not exactly the training
row count.

**Status:** Accepted

---

## D-067 - Metrics: MAE and RMSE in kW, sMAPE, no MAPE, plus energy-specific measures

**Context:** 52.2% of the PV target is **exactly zero** (18,304 of 35,040 steps).
MAPE divides by the actual value and is undefined at every one of them.

**Decision:** MAE in the target's own unit is the headline, because that is what a
capacity decision is made against. RMSE is reported alongside because it is the
sensitive one. sMAPE is reported with `0/0 = 0` defined as a perfect prediction.
MAPE is not used anywhere.

Energy-specific measures, each only where it has a definition: peak-decile error,
ramp error (consecutive horizons only), zero-period error and daytime error. The
last two are only meaningful for a target with exact zeros, and they earned their
keep: at night the ridge model errs by 3.085 kW where persistence errs by 0.929 kW -
a tree-free linear model cannot represent "output is exactly zero at night".

**No composite score is produced.** One number weighting MAE against peak error
would be an invented criterion.

**Status:** Accepted

---

## D-068 - Regime analysis measures whether regimes exist; it does not assume them

**Context:** the future Energy-MoE assumes some conditions deserve a specialist.

**Decision:** derive regimes from the data - demand terciles, ramp terciles, and for
PV the generation derivative's sign (night / morning ramp / midday / evening ramp) -
and report per-regime error and the ratio between the worst and best regime. No
expert is created, named or hinted at.

**Result:** the premise holds. On the load task the worst-to-best regime error ratio
is 12.8-18.5 for every family: error is concentrated in high-demand and high-ramp
periods, roughly an order of magnitude above low-demand periods. On the PV task the
ratio across solar phases is 15.8 for gradient boosting. That is evidence a
regime-specialised model has something to specialise on, which is the first
data-driven argument for the Phase 7 architecture.

**Status:** Accepted

---

## D-069 - One fixed seed; no significance claims

**Context:** the brief asks for explicit seeds and warns against claiming
statistical significance from a single run.

**Decision:** one seed, `20260101`, set for numpy and torch and recorded in every
registry record. `verify_reproducible` re-runs a model and asserts identical
metrics, so determinism is tested rather than assumed. **No confidence intervals,
no multi-seed significance tests and no p-values are reported**, because one run
cannot support them. Re-running with more seeds is the right follow-up before any
"significantly better" claim is made about the future models.

**Status:** Accepted

---

## D-070 - Experiment registry: committed metadata, regenerable artifacts

**Context:** §32 and §34 require a reproducible record per experiment and forbid
committing large generated artifacts.

**Decision:** one JSON Lines record per experiment in `experiments/registry.jsonl`,
carrying dataset version, dataset SHA-256, feature names, lookback, horizons, split
fractions and counts, model, family, hyperparameters, seed, durations, metrics,
artifact paths, host and library versions, and notes. Re-running an experiment
**replaces** its record rather than appending a duplicate, so the history cannot
accumulate fake trend.

`experiments/registry.jsonl` is the single file added to the `.gitignore` exception
list; the dataset arrays, reports and checkpoints stay ignored because they are
regenerable from the configuration.

**Status:** Accepted

---

## D-071 - Phase 5 architecture: dilated causal TCN as the primary model

**Context:** Phase 4 showed that both neural baselines (MLP, GRU) lost to a classical
histogram gradient-boosting model at every horizon, and that an MLP degraded badly at
24 h (6.4850 kW). The brief asks for one architecture chosen on evidence rather than
on fashion. Nothing in the project measured an attention model, so the choice had to
be made on the properties of the problem rather than on preference.

**Decision:** the primary model is a **dilated causal temporal convolutional network**
with three levels of residual blocks, kernel size 3, dilations `1, 2, 4, 8, 16, 32,
64`, `48` channels and a per-series learned embedding of dimension `8`. A compact
patch Transformer is implemented as a **secondary** model and is measured as an
ablation, not as a claim.

Reasoning recorded at the time of the decision: the receptive field spans the full
`672`-step window without recurrence, which keeps the training cost bounded; the
model is translation-equivariant, which matches a signal whose statistics do not
change with the absolute time index; and it has few enough parameters (`74,099`) that
a CPU budget is realistic. The Transformer was not dismissed on principle - it is
measured, and the result is recorded in `artifacts/phase5/ablations.md`.

**Status:** Accepted

---

## D-072 - Ordered windows as input, with the token stride applied only to inputs

**Context:** Phase 4 fed models a flat vector of lags, which discards order: the same
set of lags in a different order would be identical to the model. Phase 5's premise
is that order carries information that representation throws away. Full native
resolution for a week is `672` steps, which is expensive.

**Decision:** the input is an ordered sequence of tokens; the token stride (`4`) is
applied **only to the input window**, while the forecast targets stay at native 15-minute
resolution. A week's context therefore costs `168` tokens instead of `672`, and the
model still predicts every horizon in native steps.

Sequence construction is centralised in one place (`ml/sequence.py`) with a causality
guard that refuses any window whose origin would read a value at or after its own
forecast timestamps.

**Status:** Accepted

---

## D-073 - Chronology and split identity with Phase 4

**Context:** a Phase 5 claim is meaningless unless it is measured on the same rows as
the Phase 4 baseline it is compared against. Striding training origins changes the
training population, which would break that identity if it were applied to evaluation.

**Decision:** training origins are strided (`train_origin_stride = 4`) purely for
compute. Validation and test origins keep stride `1`, so the evaluated population is
**identical** to Phase 4's `179,520` test rows. The artifact records both counts
(`237,600` training rows against `179,520` validation and test rows) so the difference
is visible rather than implied. No shuffling anywhere.

**Status:** Accepted

---

## D-074 - Delta target from the value at the forecast origin

**Context:** persistence is within 3% of the best 15-minute result, which means most of
the difficulty at short horizons is not in the history at all. A model that must
reconstruct the level from a window is solving a harder problem than the one being
scored.

**Decision:** the model predicts the **change from the value at the forecast origin**,
and the level is reconstructed for scoring. This is measured, not assumed: the
`level_target` ablation costs **+253% at 15 min, +52% at 1 h and +20% at 24 h** against
the same configuration. It is the largest single design effect found in Phase 5.

**Status:** Accepted

---

## D-075 - Equal horizon weighting, L1 loss, no tuning against the test split

**Context:** h=96 is where the classical model is strongest. A design that up-weighted
the long horizon could produce a flattering average, and tuning the mixture weight
against the test split would make the comparison meaningless.

**Decision:** L1 loss with equal weights across the three horizon heads. Validation MAE
in per-unit terms is the checkpoint selection criterion; early stopping is patience-based
on validation only. The test split is evaluated **once**, after selection. `max_seconds`
is a compute guard that stops training and evaluates the best checkpoint so far - it
never selects anything, and the run that hit it was discarded as invalid rather than
reported.

**Status:** Accepted

---

## D-076 - GroupNorm inside the window, and why it is not leakage

**Context:** normalisation over a batch mixes information across rows. Two questions
follow: does it leak the test split, and does normalising across the time axis read the
future?

**Decision:** `GroupNorm` is applied across the channel and batch dimensions of a
window, i.e. **within** each row's own past, never across time. It therefore cannot
read a future value, and the batch contains training rows only during fitting. The
choice is recorded because "batch normalisation looks like leakage" is a reasonable
objection that deserves an explicit answer rather than a reassurance.

**Status:** Accepted

---

## D-077 - Cyclic channels retained in the main model despite the ablation

**Context:** the `no_cyclic` ablation was **better** than the reference at all three
horizons (`-3.6%` at 15 min, `-28.7%` at 1 h, `-18.6%` at 24 h). On that evidence the
sin/cos hour and day-of-year channels were not earning their place, and the honest
move would have been to drop them from the shipped configuration.

**Decision:** they are **kept** in the main run, and the negative result is reported
prominently rather than quietly fixed. Two reasons. First, the ablation suite trains at
`2` epochs and a coarse training stride, where its reference is far worse than the main
run (`0.5435` against `0.4178` kW at 15 min); a component can look useless precisely
when the model is undertrained. Second, re-running the main configuration after seeing
the test result would be tuning against the test split, which D-075 forbids.

The consequence is recorded honestly: **the shipped main configuration is not the best
configuration this study found**, and removing the cyclic channels is the first thing
Phase 6 should try at full budget. Deciding that requires a fresh, pre-registered run -
not a re-run chosen because it scored better.

**Status:** Accepted, with an explicit known deficiency

---

## D-078 - Regime analysis reports every horizon separately

**Context:** the first implementation rebuilt the error arrays inside the per-horizon
metrics loop, so the regime table was labelled with the **shortest** horizon's regime
axes while carrying the **longest** horizon's errors. Per-regime MAEs came out roughly
four times the headline figure, which is exactly the kind of error a reader would have
taken at face value. No headline metric was affected, so nothing else would have caught it.

**Decision:** analysis is factored into one shared function (`analyse_predictions`) that
takes the horizon explicitly and emits a regime table **per horizon**, plus a headline
naming which horizon it describes. Two regression tests assert that each horizon is
judged on its own errors. A checkpoint replay path was added at the same time: it
rebuilds the analysis from a saved checkpoint, **raises if the recomputed MAE differs
from the recorded one**, and in doing so verified that the shipped checkpoint still
reproduces its own metrics.

**Status:** Accepted

---

## D-079 - Phase 5 conclusion recorded as mixed, not as a win

**Context:** on the shared test split the TCN reaches `0.4178`, `0.8307` and `1.6510`
kW against the Phase 4 GBM's `0.4242`, `0.8579` and `1.5648`. It is better at 15 min
(`+1.5%`) and 1 h (`+3.2%`) and **worse at 24 h (`-5.5%`)**, while beating persistence
at 1 h and 24 h and losing to it at 15 min (`-3.3%`). Selecting the headline "a temporal
model beats the classical baseline" would be a partial reading of the model's own table.

**Decision:** the recorded verdict is **mixed / partially improves**. Two horizons of
three beat gradient boosting; the longest does not, and the longest is the one an
operator would most want to improve. `h=96` remains the open problem, and the honest
summary includes the fact that the best configuration found was not the shipped one
(D-077).

**Status:** Accepted

---

## D-080 - Phase 6 foundation checkpoint: Qwen3-1.7B-Base at a pinned revision

**Context:** the phase brief selected Qwen3-1.7B-Base provisionally, from a separate
foundation-model audit, and required the authoritative model repository to be the
source of truth for the checkpoint itself rather than for the audit's conclusions.

**Decision:** use `Qwen/Qwen3-1.7B-Base`, pinned to revision
**`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`**. Verified directly against the Hub:
architecture `Qwen3ForCausalLM`, 28 layers, hidden 2048, 16 query heads / 8 KV heads
(GQA), head_dim 128, FFN 6144, vocab 151,936 with **tied** embeddings, context 32,768,
RoPE theta 1e6 with no scaling, published dtype bfloat16, **Apache-2.0**, **not gated**.

Measured from the downloaded file rather than the model card: 3,441,185,608 bytes in a
single `model.safetensors`, **310 tensors, all BF16, 1,720,574,976 parameters**, and no
stored `lm_head` tensor because the embeddings are tied.

The license matches the audit classification, so no stop condition was triggered. The
revision is pinned rather than tracked because a moving tag would make the experiment
irreproducible, and the tokenizer's `len` (151,669) is smaller than the embedding rows
(151,936) - a padding detail that matters for anyone who later tries to use the
vocabulary head.

**Status:** Accepted

---

## D-081 - The forecasting task is inherited unchanged; only the model differs

**Context:** every claim in this phase is a comparison. A comparison needs an identical
population, an identical target and identical horizons, or it measures the setup rather
than the model.

**Decision:** Phase 6 reuses Phase 5's `SequenceConfig` exactly - same target
(`customer_load`), same 40 customers, same seed, same chronological 70/15/15 split,
same horizons `1, 4, 96`, same 672-step window at token stride 4, same delta target. The
resulting sequence index version is **`sq-437c16b5d925`**, byte-identical to the version
Phase 5 committed its metrics under, and a test asserts that equality. The Phase 5 TCN
checkpoint is then applied directly to Phase 6's rows, so the neural comparison point
is that model rather than a retrained approximation.

The configuration loader **refuses** to load a Phase 6 config that changes the target,
reorders or changes the horizons, or alters the loss. Silently redefining the task
would make every comparison in the phase meaningless.

**Status:** Accepted

---

## D-082 - Numerical projection into the residual stream; no text tokenisation

**Context:** the brief asked how to connect a language model to numerical time series,
and named three candidate routes: numerical projection, structured serialisation, and a
hybrid. Phase 5's own requirements document had already ruled out text on the merits -
the target is a smooth physical quantity, and BPE fragments floats lossy. Measured here:
`"load=0.5 lag_1=0.4"` becomes 12 tokens of unrelated subword pieces.

**Decision:** the bridge is a **fixed linear projection of numeric patches**. Per-unit
channels (demand, dy/dt, sin/cos hour, sin/cos day-of-year) at Phase 5's token stride of
4 give 168 sampled positions; 8 consecutive positions form one patch of 48 numbers,
LayerNormed per token and projected into Qwen's 2048-wide residual stream, giving **21
tokens covering the whole week**. The vocabulary embedding table and the
language-model head are both unused.

Both omissions are documented consequences rather than oversights. 151,936 embedding
rows encode token frequencies in text, which carry no information about kilowatts; and
emitting one regression output through a 151,936-way softmax would be a 311 M-parameter
multiply per position for nothing.

**Status:** Accepted

---

## D-083 - The front-end projection is fixed and seeded, and that is stated as a limit

**Context:** with a frozen backbone there is no gradient path to the projector unless
gradients are propagated through 1.7 B parameters, which this hardware cannot afford.

**Decision:** the projector is **randomly initialised from a recorded seed**
(`projector_seed = 20260101`) and frozen - the standard random-features regime. It is
**100,448 parameters** against 1,720,574,976 frozen ones.

The consequence is stated wherever results are reported: a good probe result would show
the backbone's activations are linearly informative, and could not be attributed to the
projector. A first implementation drew a fresh projector per batch, which would have
injected batch-dependent noise into the features; the projector is now constructed once
per extraction and asserted deterministic by test.

**Status:** Accepted, with a stated limitation

---

## D-084 - Frozen backbone with a linear probe, because compute forbids anything else

**Context:** the brief lists full fine-tuning, LoRA, QLoRA, PEFT, continued pretraining
and SFT as candidates, and requires the choice to follow from measured compute rather
than popularity.

**Decision:** the shipped configuration is a **frozen backbone plus a trainable head**,
and the choice is forced by measurement rather than preferred. The installed torch is
**CPU-only** (`2.14.1+cpu`, `torch.cuda.is_available() == False`) and the only GPU is an
RTX 4050 laptop with 6,141 MiB total, of which roughly 2,400 MiB is already consumed by
the desktop. The weights alone are 3.44 GB in bf16 and 6.9 GB in fp32, so no optimiser
state fits in the remaining VRAM.

LoRA, QLoRA and partial unfreezing therefore remain **unrun and untested**. That is
recorded as an open question, not as a claim that they would not work: the frozen probe
measures whether the pretrained representation is *usable*, not whether adapting the
backbone would help.

The head carries an explicit **per-series embedding** (8 dimensions). This is not
decoration: every input is per-unit, so the magnitude distinguishing a 1.4 kW household
from a 210 kW shop has been deliberately divided out, and Phase 5's TCN carried the same
embedding. Without it the comparison would measure the handicap rather than the
representation.

**Status:** Accepted, with adaptation left explicitly open

---

## D-085 - fp32 for inference, because on this CPU bf16 is 4.8x slower

**Context:** the checkpoint is published in bfloat16, so bf16 is the obvious compute
dtype. It is also the wrong choice here.

**Decision:** run the backbone in **float32**. Measured on this machine at 21 tokens,
batch 8: bf16 **2.568 s/sample** against fp32 **0.533 s/sample** - a **4.8x** penalty,
because the CPU has no native bf16 arithmetic and torch emulates it. With 16 threads and
batch 64 the fp32 figure improves further to **0.236 s/sample**, and the three real
extraction passes measured **0.248-0.278 s/sample** over 25,000 rows each.

Published checkpoint dtype and fastest inference dtype are different questions. The
choice is recorded in every result's `compute` block so the 5.4 h extraction cost is
attributable.

**Status:** Accepted

---

## D-086 - A subsample, with every baseline re-measured on the identical rows

**Context:** at 0.248 s/sample the full Phase 5 population costs 15.6 h for training
origins and 11.8 h for test origins - per arm, and there are three arms. The brief
requires the same dataset split and the same metrics "where applicable".

**Decision:** score on a **deterministic, per-series balanced subsample** - 10,000
train / 5,000 validation / 10,000 test rows, evenly spaced in time inside each split,
5.57% of Phase 5's test population. Then, crucially, **re-measure every baseline on
exactly those rows**: persistence is free, Phase 4's gradient boosting is refitted on
the subsample's training rows using Phase 4's own feature definitions, and Phase 5's
actual checkpoint is applied to the same rows.

Absolute MAE on a subsample is not the published Phase 4/5 figure, and the artifacts say
so explicitly. The result is the **paired** comparison, with a paired bootstrap of the
per-row error difference (2,000 resamples) - which cancels the row-to-row variance
dominating absolute MAE and is far tighter than two independent intervals.

Two baseline bugs were found and fixed while building this, both of which had produced
confidently wrong numbers: the GBM was trained on kW targets against per-unit features
(train MAE 14.6 kW where Phase 4 reported 0.42 kW), and the TCN baseline compared
per-unit predictions against kW actuals (13.3 kW where Phase 5 reported 0.4178 kW). A
third was a missing `series_index`, which silently paired each row with another
customer's demand. All three produced a *better-looking* Qwen result before they were
caught, which is exactly why the baselines are asserted against Phase 4's published
figures in review rather than trusted.

**Status:** Accepted

---

## D-087 - The random-backbone control is mandatory, and it is the phase's key result

**Context:** "does a pretrained transformer representation transfer to energy dynamics"
is unanswerable without a control. A deep random feature map of the same architecture is
a serious baseline, and Phase 5's transformer ablation had already shown that attention
did not obviously beat convolution on this data.

**Decision:** every configuration can be run against a **randomly initialised Qwen3 of
identical architecture and parameter count**, through the identical extraction and the
identical probe. The control is a first-class arm in `configs/qwen.toml`, not an
afterthought, and the config loader refuses an ablation that does not declare the
question it answers (D-087's companion rule, enforced in `config/qwen.py`).

**Measured:** on identical rows with an identical probe, pretrained beats random by
**33.9% at h=1** and **22.9% at h=96**, and ties at h=4 (0.6%). So pretraining
contributes real, measurable signal - and nowhere near enough. Recorded in
`docs/moe_design_requirements.md` §3 along with the participation-ratio measurement that
explains why.

**Status:** Accepted

---

## D-088 - Cyclic channels removed from the best configuration, after Phase 6 measured it

**Context:** D-077 kept Phase 5's sin/cos hour and day-of-year channels even though the
ablation said they were not earning their place, because removing them after seeing the
test result would be tuning against the test split. Phase 6 gave the question a second,
architecture-independent test: Qwen uses RoPE, which encodes only **relative** position,
so under a transformer the absolute-time channels have a stronger justification than
they did under a convolution.

**Decision:** the test was run and it is reported. Removing the channels **improves** the
probe by 36.7% at h=1 and 33.3% at h=4, and costs 2.5% at h=96.

The channels nevertheless stay in the *shipped* `configs/qwen.toml`, for the same reason
as D-077: the improvement was measured on the test subsample, so acting on it now would
be exactly the test-set tuning that decision forbade. What changed is that the evidence
is now two-sided and from two architectures, so a future pre-registered run can drop them
with a genuine prior rather than a hunch. Recorded in
`docs/moe_design_requirements.md` §4.

**Status:** Accepted, deliberately unchanged pending a pre-registered run

---

## D-089 - Phase 6 verdict: Qwen loses, and the reason is measured rather than asserted

**Context:** the phase asks whether Qwen3-1.7B provides a measurable advantage over the
established baselines. The answer may be YES, NO or MIXED BY HORIZON.

**Decision:** the recorded verdict is **NO, and not narrowly.** The best Qwen
configuration (a two-layer head on layer 28) reaches 0.5452 / 1.6060 / 2.7400 kW against
the refitted GBM's 0.4946 / 0.9749 / 1.6168 and the Phase 5 TCN's 0.4081 / 0.8224 /
1.6522 - worse at every horizon, worse than persistence at h=1, and every comparison
significant under the paired bootstrap. Qwen wins no regime in §1's tables.

The verdict is accompanied by the measurement that explains it rather than a shrug: the
frozen backbone's final-layer readout has a **participation ratio of 2.1** across 2048
dimensions, versus 10.0 for a random backbone of the same architecture. The signal
handed to the head is nearly rank-2. Pretraining still helps relative to random (§D-087),
so the finding is "real transfer, insufficient transfer", not "no transfer".

**This is a negative result and it is the phase's product.** The bar was pre-registered
in `docs/qwen_energy_requirements.md` §1 before any of this ran: beat **1.5648 kW at 24
hours**. Qwen's h=96 result is 2.7400 kW.

**Status:** Accepted

---

## D-090 - Layer choice and probe depth are reported as findings, not tuned away

**Context:** two ablations produced large effects - reading layer 16 instead of layer 28
gives 23.33 kW at h=1 against 1.33 kW, and a two-layer head improves h=1 from 1.3324 to
0.5452 kW. Both mean the shipped configuration is not the best one found.

**Decision:** both are reported in full and neither is acted on in the shipped config.
Layer 28 is the final layer and needs no justification; the deeper head was found *after*
seeing test results and selecting it now would be tuning against the test split (D-075's
principle, carried forward). The consequence is stated plainly: **the best Qwen
configuration this study found is not the shipped one**, and even that best arm loses.
The probe search is therefore shallow, and "the probe was under-parameterised" remains an
open possibility - it is not a settled explanation.

**Status:** Accepted, with an explicit known deficiency

# Phase 7 - Heterogeneous energy expert routing (D-091 … D-098)

## D-091 - Phase 7 routes over heterogeneous models; the Qwen-internal MoE is not built

**Context:** this phase's plan, written into `docs/moe_design_requirements.md` after
Phase 6, was to add sparse expert FFN layers inside `Qwen3-1.7B-Base` with a
load-balancing loss and capacity factors. The motivation was Phase 6's measurement that
the frozen backbone's final-layer readout has a participation ratio of **2.1** across 2,048
dimensions - an almost rank-2 signal - on the reasoning that specialisation would help.

**Evidence:** Phase 6 also measured that Qwen wins **no regime** (§1's tables, measured
twice), loses to every existing baseline at every horizon, and that pretraining
nevertheless contributes real signal (33.9% better than a random backbone at h=1). The
premise being acted on - that the shared trunk's collapse is what is holding the model
back - was therefore untested, while the measurement that Qwen is fourth in every regime
was in hand.

**Decision:** build a **heterogeneous Energy Expert Router** over the three models that do
work: persistence, Phase 4's `classical_hist_gbm`, and Phase 5's `phase5_tcn`. The gate is
a softmax over their forecasts, not over internal layers.

**Alternatives:**

| Alternative | Why not |
|---|---|
| Qwen-internal MoE | The trunk wins no regime and its readout is rank-2. Adding experts inside it would have produced a larger model with a worse starting point, on a machine where Phase 6 already measured a frozen forward pass at 0.248 s/row |
| One model, the best single expert | The best expert changes with horizon (persistence at h=1, GBM at h=96) and with regime (persistence in low-ramp, GBM in high-ramp). A single choice is a compromise everywhere |
| Fixed weighted ensemble | A live baseline, not an answer. It captures the average benefit of combining but cannot respond to the state, which is the hypothesis being tested |

**Why:** the phase's own instruction was to determine whether heterogeneous expert
specialisation is a useful architectural principle, and the only way to find out is to
route between the specialists that exist. Specialising a backbone that wins nothing would
have tested a different question.

**Consequences:** the load-balancing loss and capacity-factor requirements of §6.4 and
§6.5 of the superseded plan do not apply; the router's weights are inspected directly
instead (capacity is reported as the share of rows each expert is selected for). Qwen
remains a documented negative result and is revisited only if a new experiment gives a
reason. Recorded in `docs/expert_router_design.md` §1.

**Status:** Accepted

---

## D-092 - The router is trained on validation rows, not on the experts' training rows

**Context:** the router's label is "which expert would have been most accurate for this
row". If those expert predictions came from a model that had already fitted the row's own
target, the expert would look better than it ever will at inference, the label would be
noise, and nothing in the loss would look wrong.

**Decision:** the learned experts are fitted on the **training** split (Phase 4's grid for
the GBM, Phase 5's committed checkpoint for the TCN); the router is fitted on a
chronological **70% of the validation split**, early-stopped on the last 30%, and evaluated
on **test** exactly once. The fixed-weight and best-single baselines select their parameters
on the same router-training rows, so no baseline is advantaged by having seen test.

**Measured, not asserted:** `offline/crossfit.py` halves the validation block in time,
fits one GBM on the first half and computes the oracle gain on rows it did and did not see.
In-sample labels give 21.57% / 19.31% / 18.39% against 21.12% / 29.80% / 37.49%
out-of-sample at h=1 / 4 / 96. The direction is not the naive worry: an expert that looks
better in-sample makes the pool look *more* interchangeable, so the oracle gain **shrinks**.
Routing labels from in-sample predictions would have understated the available headroom at
long horizons by roughly half, and a router chasing them would have reported a small honest
improvement as if it were the whole opportunity.

**Residual limitation, stated:** Phase 5's TCN used the validation split for checkpoint
selection, so validation rows are not entirely independent of the *experts*. This is
inherited from Phase 5 and is a much smaller effect than in-sample expert predictions.

**Status:** Accepted

---

## D-093 - Oracle analysis lives in a separate subpackage, enforced by tests

**Context:** the phase requires an offline oracle - the per-row best expert with hindsight -
as an upper bound, and requires that it never be reachable from production inference. The
oracle reads the realised target by definition, so a system that routed on it would be
scoring itself against the answer.

**Decision:** the oracle and every analysis that legitimately reads the target live in
`ml/router/offline/`. Deployable modules (`experts`, `features`, `model`, `routing`,
`training`, `ensembles`, `cache`) are one level up and import none of it. Two tests enforce
the boundary: one parses each deployable module's AST and asserts no import resolves into
`offline`, `oracle` or `evaluation`; the other asserts no deployable module *names* an
oracle in code, while allowing prose that explains why it is unreachable.

**Why a subpackage rather than a docstring:** "the oracle is offline-only" written in a
comment is a promise. Import structure plus an AST test is a check, and a check is what
keeps the property after the person who wrote it has moved on.

**Status:** Accepted

---

## D-094 - The router is trained as a residual on the state-independent optimum

**Context:** initialised near uniform, an L1 objective through a softmax drives the logits
apart and the router settles on a single expert within a few epochs. Measured: mean weights
became (0.000, 0.000, 0.999) at h=4 and (0.000, 0.000, 1.000) at h=96, because the gradient
through the softmax vanishes as a vertex is approached. At h=1 the collapse was harmless; at
the longer horizons it cost more than a fixed weighted ensemble achieved, because a hedge
that reduces absolute error is precisely the solution a vertex cannot express.

**Decision:** train two variants from the same rows and the same protocol - **warm**,
initialised at the validation-fitted fixed-ensemble weights (head bias `log(w)`, head weights
zeroed, so epoch 0 *is* that mixture exactly), and **cold**, near uniform. Epoch 0 is scored
and eligible for selection, so a warm-started router can never regress below its starting
point on the selection rows. The variant is chosen **per horizon on the selection rows
only**.

**Why not simply always warm-start:** at h=1 the cold variant is better on the selection
rows (0.4233 against 0.4316) and the choice is made on validation, not on test.

**Recorded, because it is the phase's central negative result:** the warm-started router
selected **epoch 0 at every horizon**. Gradient descent found no state-dependent weighting
that improved on the constant one it started from, at any horizon. A router that returns its
own initialisation is not a router, and saying so is more useful than tuning until it does
not.

**Status:** Accepted

---

## D-095 - The verdict is the stricter of two comparisons, and a win needs 0.0001 kW

**Context:** the phase asks whether routing beats "the strongest individual expert", and
separately requires comparison against a fixed weighted ensemble. Reporting only the first
would let a router that learned nothing be called a success - which is exactly what an
earlier iteration of this phase produced, when a warm-started router reproduced the fixed
ensemble to fourteen decimal places and was scored YES.

**Decision:** report both, and take the verdict from the **stricter** one. A horizon counts
as beaten only when the router's test MAE is lower by more than **0.0001 kW** - a tenth of a
watt, far below anything physically meaningful and well above floating-point noise. Two
methods differing by 2e-6 kW are the same method.

This threshold is not cosmetic. An earlier iteration of this phase produced a router whose
MAE equalled the fixed ensemble's to fourteen decimal places at h=4, and a strict `less than`
test called that a win. The horizon is now recorded as a **tie**, which is what it is.

**Result:** versus the best single expert **PARTIALLY, 2 of 3** (wins h=4 by 4.70% and h=96
by 0.87%; loses h=1 by 0.64%). Versus a fixed weighted ensemble **NO, 0 of 3** (h=4 tied;
loses h=1 by 2.16% and h=96 by 1.31%). Phase verdict: **PARTIALLY**.

**Consequence:** the phase's conclusion is negative, and says the fixed weighted ensemble is
the better deployable combiner at every horizon.

**Status:** Accepted

---

## D-096 - The oracle mixture is a bound on fixed weighting, not on the router

**Context:** the phase's analysis initially documented `oracle_mixture` - the best convex
combination chosen per horizon *in hindsight* - as an upper bound any soft routing cannot
exceed.

**Evidence:** established on a constructed case and kept as an executable claim. Three
experts predicting the constants 0, 5 and 10 kW against a series alternating between 0 and 10:
any fixed combination is itself a constant, so it cannot beat 5 kW of mean error whatever the
weights are, while choosing the low expert on low rows and the high expert on high rows is
exact. A *state-dependent* rule can beat the best *constant* rule without seeing the target.

The deployed router happened not to beat it - 0.4069 against 0.3956 kW at h=1 - which is a
separate finding (§D-098) and not a contradiction. The relationship being corrected is about
what the quantity bounds, not about how this particular router performed.

**Decision:** the hard oracle remains an upper bound on **hard** routing, because it is a
per-row selection. `oracle_mixture` bounds **fixed** weighting only, and is the correct
baseline for the validation-fitted fixed ensemble rather than for the router.
`test_oracle_mixture_is_not_an_upper_bound_on_a_state_dependent_router` asserts the
relationship on a constructed case so the docstring cannot drift back.

**Status:** Accepted, correcting an earlier claim of this phase

---

## D-097 - Router features are lagged by exactly h + 1 steps, and the boundary is tested

**Context:** `past_error` - each expert's most recent realised error for the same horizon -
is the strongest single feature group (removing it costs 0.037 kW at h=1), and the only
group that is a function of the experts rather than of the demand. It is also where a leak
would be easiest to write.

**Decision:** at origin `t`, horizon `h`, the feature is
`|f_e(t - h - 1; h) - y(t - 1)| / rated_kw`. The delay is `h + 1` because the forecast issued
at `t - h - 1` is the newest one whose outcome is known at `t`. Using the forecast issued at
`t - 1` would read a target at `t + h - 1`, which has not happened.

**Normalised by rated kW.** A router weighting on raw kW error would learn one policy for a
1.4 kW household and another for a 388 kW shop, from 40 series.

**Boundary enforcement:** `assert_router_features_are_causal` poisons every value after a
sampled row's own origin in its series, rebuilds the features of **every row of that series
up to that origin**, and requires bit-identical output. The realised targets are recomputed
from the poisoned series rather than passed in, so a look-up reaching past the origin really
would change. `test_causality_check_actually_catches_a_leaking_builder` injects a
deliberately leaking builder and asserts the guard fires, because a guard that has never
rejected anything is not evidence of anything.

**Status:** Accepted

---

## D-098 - A learned, state-dependent router is deployed for h=1 only; the rest gets a fixed ensemble

**Context:** §6.7 of the superseded plan warned that "the low-ramp regime, where
persistence reaches 0.0034 kW, is the one a router is most likely to damage". It was
damaged: the router scores 0.0188 kW there against persistence's 0.0051, a factor of **3.7**,
on 39% of test rows. The router's wins are in the high-demand and high-ramp terciles, where
73% of total error lives.

**Measurement, on the sealed test split:**

| | h=1 | h=4 | h=96 |
|---|---|---|---|
| Router MAE | 0.4069 | 0.7917 | 1.5600 |
| Best single expert | **0.4043** | 0.8307 | 1.5737 |
| Fixed weighted ensemble | **0.3983** | **0.7917** | **1.5399** |
| Oracle | 0.2871 | 0.5217 | 0.9942 |

The router wins the high-ramp tercile (1.0929 against persistence's 1.1970 kW) and loses the
other five regimes. Selection accuracy is 23-39% against a three-way choice, and the router
selected persistence **13 times in 179,520 rows**.

**Decision:** the result is recorded as it is. No gate, no regime-conditional policy and no
low-volatility shortcut was added after seeing it, because each would be fitted on the test
split. What is recorded instead is the deployment recommendation: use a fixed weighted
ensemble, and treat volatility-gated routing as the direction for a **pre-registered**
experiment rather than as a fix derived from these results.

**Consequence:** the phase's principle, as measured rather than as planned:

> Different forecasting experts do have different strengths across energy conditions, and
> 30-38% of achievable accuracy is available between them. On this data a learned
> energy-aware router does not capture it: a fixed weighted ensemble is better at every
> horizon, and the router's advantage over any single expert comes from combining rather
> than from routing on state.

**Status:** Accepted, negative result recorded

## D-099 - A paired lower/upper bound is one uncertainty quantity, not two

**Context:** Phase 2's `UncertaintyEstimate` counted `lower_bound` and `upper_bound` as two
separate summaries and refused more than one, on the stated grounds that "how an interval
relates to a standard deviation depends on a distributional assumption that Phase 8 has not
yet made". The consequence was that a **prediction interval - the one representation Phase 8
exists to produce - was the one representation the domain could not hold.** The phase would
have had to invent a parallel type beside the domain model, which is how a domain stops being
a domain.

**Decision:** a *paired* interval is one quantity and is accepted. Pairing an interval with a
scalar summary (`standard_deviation`, `relative_std`) is still refused, because the stated
reason was about mixing the two and that reason survives untouched. Supplying one bound
without the other is refused - half a claim is not a claim. Recorded in
`docs/uncertainty_design.md` §8 and exercised by
`tests/domain/test_values.py::test_uncertainty_accepts_a_paired_prediction_interval`.

**Consequence:** `ProbabilisticForecast` converts to a Phase 2 `Forecast` with no parallel
type, and the optional `uncertainty` field that Phase 2 deferred is finally populated.

**Status:** Accepted

## D-100 - Validation is halved for calibration, not reused from the router's 70/30 split

**Context:** Phase 8 needs rows the point forecast did not fit (all of validation) and rows
the *width scale* did not fit (half of those, at minimum). Phase 7 already split validation
70/30 to fit and select its router, so reusing that split would put the scale model on rows
Phase 7's three weights per horizon were fitted on.

**Decision:** validation is halved chronologically - `CAL_FIT` (89,760 rows) fits the scale
functions and the quantile models, `CAL_CONF` (89,760 rows) supplies the conformity scores
and the band cut points. `CalibrationSplit` refuses overlapping halves, refuses any
calibration row inside test, and refuses any calibration row outside validation.

**Consequence:** the scale never depends on the residuals it is asked to normalise, which is
the structural precondition for the normalised conformal form to preserve its guarantee. The
overlap with Phase 7's weight fit that remains is not assumed away - it is measured and
reported (§2 of the design doc): +8.576% at h=1, -0.721% at h=4, -11.025% at h=96.

**Status:** Accepted

## D-101 - The split-parity number is reported even though it is inconvenient, and it predicts the headline negative result

**Context:** The fixed ensemble's own MAE differs by 8.6% (h=1) and 11.0% (h=96) between the
two halves of validation. Nothing can be done about it: the halves are chronological, demand
difficulty moves across a year, and the alternative - a random split - would put test-period
rows into the calibration set.

**Decision:** the gap is a headline number in the result and the summary, not a footnote. The
alternative - reporting only that "the split was chronological" - would leave the reader with
no way to judge whether the calibration transferred.

**Consequence:** it turns out to be the explanation for the phase's main negative result. The
default constant-width interval calibrated on `CAL_CONF` over-covers at h=1 (0.9836 against a
nominal 0.95) and under-covers at h=96 (0.9071), failing 11 of 12 level-horizon cells, and the
sign of each failure matches the sign of the gap at that horizon.

**Status:** Accepted, and the measurement earned its place

## D-102 - The conformal guarantee is reported as an empirical measurement, not invoked

**Context:** Split conformal gives `P(y_new in interval) ≥ 1 - α` when conformity scores are
exchangeable with future scores and the point predictor is fixed before calibration. The first
condition is load-bearing and it is not met: D-101's own numbers are a demonstration that
residuals from the two halves of a single validation split are not exchangeable, and across a
year they are less so.

**Decision:** the finite-sample correction is still computed the correct way
(`ceil((n+1)(1-α))`-th order statistic, capped at `n`) and the intervals are still called
conformal, but the string attached to every conformal interval states that coverage here is
an empirical measurement and that a time-ordered method would be required for a real
guarantee. `tests/ml/test_uncertainty_leakage.py` asserts that the wording survives.

**Consequence:** no document in this repository may be read as claiming a coverage guarantee
for the shipped intervals. A time-ordered conformal calibrator is recorded as the direction
that would earn one.

**Status:** Accepted

## D-103 - The published procedure is selected per horizon from measured calibration, and only among calibrated methods

**Context:** No single interval method was both calibrated and best-scoring at all three
horizons. At h=1 `quantile_regression` has the best weighted interval score (1.938 against
`conformal_state`'s 2.156) and covers 0.9164 at a nominal 0.95 - it fails. An earlier version
of this phase selected on interval score alone and published a method whose own
`calibration_status` read `FAIL`.

**Decision:** selection is per horizon, at the artifact's band level, among methods whose
measured coverage is inside the tolerance, by lowest weighted interval score. If nothing is
calibrated at a horizon, the fallback is published **with its FAIL status and a note saying it
is a measurement rather than a selection** - never silently narrowed. The batch's `method` is
one name per horizon for the same reason: a single name would credit a horizon with a method
that was not chosen for it.

**Consequence:** all three published horizons read `PASS`
(`conformal_state` at h=1 and h=4, `conformal_dispersion` at h=96). The verdict's
`is_worth_the_machinery` component compares the best *calibrated* method against the
baseline, and records the unconstrained winner alongside so the gap is visible.

**Status:** Accepted

## D-104 - Band cut points are per-horizon, and are terciles of the calibration split only

**Context:** The operational LOW/MEDIUM/HIGH band is what a downstream consumer reads. Two
things could have gone wrong and the first did. Taking terciles of the widths *pooled across
horizons* - whose means differ by 5x - labelled 61% of h=1 rows "low" and 43% of h=96 rows
"high", so the band reported the horizon rather than the row. Taking them from the test rows
would make an operational label a test statistic.

**Decision:** one tercile pair per horizon, computed from that horizon's **conformity** rows.
`ProbabilisticForecastBatch.band_cuts` requires one ascending pair per horizon and rejects a
single shared pair.

**Consequence:** the h=1 bands are 59,156 / 57,972 / 62,392, and the h=96 bands are
74,075 / 44,330 / 61,115 - a real spread, tracking a real width distribution, rather than a
horizon label.

**Status:** Accepted

## D-105 - The vocabulary is predictive uncertainty, model disagreement and data variability; no decomposition is claimed

**Context:** Two of the six methods are built on "the experts disagree", conventionally called
epistemic uncertainty, and the residual-based methods are conventionally called aleatoric.
The prompt's §23 hypothesis is phrased in the first vocabulary and the required output in the
second.

**Decision:** neither label is used. Separating the two requires assumptions about the
error-generating process that this data cannot support, and Phase 6 already measured that the
pretrained representation carries almost no usable structure. Where a method leans on
disagreement it says so, and `disagreement_is_predictive` in the verdict record is keyed by
the measure's own name rather than by a decomposition.

**Consequence:** the artifact carries `UncertaintyKind.COMBINED`, and a consumer can refuse an
epistemic interval where it needed an aleatoric one without the phase having pretended to
know which it produced.

**Status:** Accepted

## D-106 - Data-quality conditioning is reported as impossible, not simulated

**Context:** The phase's brief asks whether uncertainty rises under missing, derived,
interpolated or stale inputs. SMART-DS carries no per-timestep quality flags: Phase 3 recorded
`flags=['ok']` across all 35,040 steps and the `TargetSeries` `DataQuality` is clean for every
row.

**Decision:** the experiment is recorded as not run, with the reason and the available
metadata attached. No corruption is injected. Doing so would answer a question about the
injected corruption rather than about this dataset, and it would produce a number in a
report that a reader could mistake for a property of SMART-DS.

**Consequence:** for this dataset the equivalent question is answered by the horizon and
regime breakdowns, where difficulty genuinely varies - width tracks error in every regime at
every horizon, with width ratios of 4-16x against error ratios of 5-28x. A dataset with real
missing-data flags is where this experiment belongs.

**Status:** Accepted

## D-107 - Two-sided tails are mandatory; a nominal level is the mass the interval actually covers

**Context:** `quantile_levels` documented that a nominal 90% interval puts 5% below the lower
bound and 5% above the upper one, and returned `(1 - level, level)`. With those tails a
"nominal 90%" interval is a central 80% interval, and a nominal 50% interval is the single
median quantile with zero width. The symmetric methods in the phase are unaffected - they take
the `1 - α` quantile of `|residual|`, which was already correct - so only quantile regression
would have been silently mislabelled, and it would have looked like a method that simply
under-covers.

**Decision:** `quantile_levels` returns `(α/2, 1 - α/2)`. The quantile grid is built from it,
so every reported tail is fitted exactly; `quantile_for_tail` is strict by default and raises
on an off-grid tail rather than snapping, because "an interval labelled 90% that is really 89%
is a miscalibrated interval with a misleading name".

**Consequence:** quantile regression's coverage at a nominal 95% improved from 0.8438 to
0.9164 at h=1 purely from reporting the level it actually achieves. It still fails, for the
separate and honest reason in §5 of the design doc.

**Status:** Accepted

## D-108 - Crossing quantiles are counted at the pairs actually used, and reported

**Context:** Independently fitted pinball models cross: a lower quantile is predicted above a
higher one, which produces a negative-width interval. Two responses were possible - repair
silently, or count.

**Decision:** apply the standard monotone rearrangement so the interval stays valid, and
report the count **at the lower/upper pairs the phase actually publishes**. A grid-wide
count is 34-44% and is meaningless, because it counts adjacent fitted quantiles such as 0.90
against 0.95 that no reported interval uses.

**Consequence:** the published counts are 172 rows at h=1, 9 at h=4 and 0 at h=96, out of
359,040 - small enough to be a footnote and large enough to be stated. Quantile regression is
not published at any horizon, but the count is in the summary rather than buried in a note
string.

**Status:** Accepted

## D-109 - `ProbabilisticForecast` carries no decision, no confidence score and nothing derived from uncertainty and an objective

**Context:** The phase could have defined an action confidence score - "can I rely on this
forecast?" - from the interval plus a cost. It is the obvious next thing a reader wants.

**Decision:** the artifact stops at the forecast. `uncertainty_notes` records the calibration
sample and the band definition; it does not record an action. A decision-assurance layer needs
an objective, a cost of being wrong, and a measure of the alternatives, and defining a
confidence formula here would fix it on the evidence of one forecasting experiment.

**Consequence:** a future consumer composes the interval with its own objective. Phase 13's
decision-assurance layer starts from a measured, calibrated interval rather than from a
number this phase invented.

**Status:** Accepted


## D-110 - A statistical proxy may not claim a control authority, enforced in the constructor

**Context:** Phase 9 has to represent flexibility. The tempting shape is one number with a
magnitude, a direction and an authority, and the temptation is to let a behaviourally-derived
figure carry `SCHEDULEABLE` so a planner can use it directly. That single permission is the
difference between "demand has moved this much before" and "this much can be commanded", and
the second is not something observational data can support.

**Choices:** (a) one magnitude with an advisory flag callers are trusted to respect;
(b) a `FlexibilityBasis` enum with the authority rule enforced in `__post_init__`.

**Selected:** (b). `basis=STATISTICAL_PROXY` with any controllable authority raises
`DomainValidationError`. `basis=UNKNOWN` may not carry a magnitude at all, so "not known" can
never be published as "none available".

**Rationale:** a rule in a docstring is a rule a caller forgets under deadline. A rule in the
constructor is a rule that holds for every caller including future ones, and it fails loudly
and immediately rather than producing a plausible wrong number.

**Consequence:** Phase 9 publishes a behavioural envelope that cannot be mistaken for
capability by anyone holding the object. Phase 10 receives a `STATISTICAL_PROXY` /
`NOT_CONTROLLABLE` figure and must obtain physical limits externally before optimising.

**Status:** Accepted


## D-111 - The capability audit runs before any estimation and its "nothing is supported" result is a deliverable

**Context:** the phase could open by estimating an envelope and qualifying the caveat at the
end, or by first establishing which flexibility dimensions the dataset can support at all.

**Selected:** audit first, and treat "0 of 8 dimensions physically supported" as the phase's
headline finding rather than as a caveat. `energy-intel flexibility audit` exposes it as a
sub-command that fits nothing, in about a second.

**Rationale:** if the audit ran last, every number in between would already have been
interpreted as flexibility by whoever read the first table they saw. Running it first makes
"no physical flexibility is supported" the frame within which the behavioural numbers are
read, and separating it into its own command means the question can be answered without any
risk of it being mistaken for a fitted result.

**Consequence:** the phase's verdict is the weakest component, so it is NO. That is the correct
outcome for an honest characterisation of this dataset, and Phase 10's prerequisites are
listed explicitly rather than implied.

**Status:** Accepted


## D-112 - Envelope cells are fitted per horizon, not pooled across horizons

**Context:** a calendar slot aggregates roughly 90 days of observations, so pooling every
horizon into one cell per `(series, slot)` triples the effective sample. The first
implementation did exactly that.

**Selected:** one independent table per horizon. Cells are keyed `(horizon, series, slot)`,
and `magnitudes()` takes a horizon argument rather than defaulting.

**Rationale:** the first run returned **the same 4.1622 kW band at h=1, h=4 and h=96**, with
coverage 0.949 / 0.853 / 0.700. Deviation at h=96 is about four times deviation at h=1, so a
shared cell must be wide enough for h=96 and the h=1 band over-covers by five points. No
downstream calibration can repair a band that is ten times too wide at one horizon because it
was fitted at another. Pooling remains correct for the *baseline*, where a time-of-day profile
is the same profile at every horizon; it is wrong for the *magnitude*.

**Consequence:** widths are 2.73 / 5.57 / 14.12 kW at h=1/4/96, and per-horizon calibration is
possible at all.

**Status:** Accepted


## D-113 - The envelope uses split-conformal order statistics, with the miss probability split between the tails

**Context:** the band's tails could be empirical quantiles via interpolation, or order
statistics at a finite-sample-corrected rank.

**Selected:** `upper_index = min(ceil((n+1)(1 - tail)), n) - 1` and
`lower_index = max(floor((n+1)tail), 1) - 1`, with `tail = (1 - nominal_level) / 2`.

**Rationale:** two defects in one. Using the *total* miss probability as the per-tail quantile
gives a 90th and a 10th percentile, which is an 80% band labelled 90% - measured in-sample
coverage was 0.6485, and nothing crashes, the band is simply mislabelled. Separately, an
interpolated quantile from a handful of observations sits *between* observed points and
under-covers. The conformal rank guarantees at least nominal coverage however small the cell
is and degrades to the sample extreme when the cell is too small to promise it.

**Consequence:** the band delivers what it says at nominal 0.90, and `tail_at_saturation`
records which cells were too thin to reach the rank.

**Status:** Accepted


## D-114 - A conformal scale is solved on the conformity split and applied unchanged to test

**Context:** even correctly fitted, the band over-covered the later split by +2.71pp at h=1
and +3.39pp at h=4 on 179,520 rows - about 47 binomial standard errors. That is a real shift
between periods, not sampling error, and re-fitting quantiles cannot fix it.

**Choices:** (a) publish the over-covering band and report the miss; (b) solve one symmetric
scale per horizon on the conformity half; (c) re-fit quantiles on more data and re-read test.

**Selected:** (b). Coverage is monotone non-decreasing in the scale, so a bisection on the
conformity rows is exact. The scale is then applied to test without adjustment.

**Rationale:** (c) would read the sealed split a second time. (a) would publish a figure whose
stated level is wrong by three points. The conformity half of the split exists for exactly this
correction, and using it is what makes the design a three-way split rather than two.

**Consequence:** scales of 0.9184 / 0.9063 / 0.9572 bring sealed-test coverage to 0.9140 /
0.9110 / 0.8925. h=1 and h=4 remain 0.4-0.4pp outside a 1.0pp tolerance and are reported as
such: a phase that kept rescaling until the test split passed would be fitting to it.

**Status:** Accepted


## D-115 - The reliance coupling is measured and then withheld from the published envelope

**Context:** Phase 8's interval width was to condition the promised flexibility:
`reliance = clip(trailing_median(normalised_width) / normalised_width, 0.25, 1)`.

**Selected:** compute it, test it on the sealed split, publish the evidence, and **do not apply
it**. The published envelope is the raw behavioural one.

**Rationale:** two independent reasons. The coupling turned out to be *supported* -
Spearman +0.393 / +0.246 / +0.312 - so this is not a case of a null result being hidden. But
the factor was computed and judged post hoc on the sealed split, so applying it would reuse
test information to construct the quantity under evaluation. Separately, the discount is worth
12-31% of the band's width; that is not a refinement, it changes what the envelope means, and
doing that on one post-hoc correlation is the kind of inference this phase exists to refuse.

**Consequence:** Phase 9 publishes evidence that Phase 8's uncertainty does carry information
about flexibility, and a later phase that wants to apply the discount must calibrate it on
data disjoint from its evaluation. The trailing reference stays strictly causal, and the floor
stays at 0.25 because a floor of zero would let a wide interval imply zero flexibility - a
claim about capability rather than about confidence.

**Status:** Accepted


## D-116 - Phase 9 reads Phase 8's published artifact and proves the rows line up

**Context:** the coupling pairs each of Phase 8's widths with a deviation computed here. Those
two must describe the same row, and Phase 8's artifact covers test rows only while
`split_parity_check` needs the whole panel.

**Selected:** rebuild the full-panel fixed ensemble from Phase 7's cache and its recorded
weights; verify it against Phase 7's published MAE to 1e-6 before estimating anything; then
prove alignment by comparing Phase 8's artifact point forecast against the rebuilt one on the
test slice, requiring agreement to 1e-9 kW.

**Rationale:** row *counts* are not proof. A permutation preserves the count and the shape, so
every structural check passes and every interval width is then paired with another row's
deviation - which would make the coupling's entire verdict meaningless while looking healthy.
Rebuilding the ensemble also needed the weights in the order Phase 7 recorded them; a first
attempt multiplied an `[expert, horizon]` matrix with a `"he"` einsum and produced a
plausible-looking forecast from transposed weights, caught only because the MAE parity check
runs first (measured maximum relative difference on the corrected build: 0.0%).

**Consequence:** the coupling's verdict rests on a proved row correspondence, and the run
refuses to start if either check fails.

**Status:** Accepted


## D-117 - Stability is measured within a series, and a constant band counts as perfectly stable

**Context:** "is the envelope stable enough for a planner to consume" needs a statistic on the
band's own smoothness. The natural one is a lag-1 autocorrelation of its width.

**Selected:** compute it **within each series**, each series centred by its own mean and the
within-series products pooled. A width with no within-series variation scores 1.0 and reports
`width_is_constant_within_series: True`, rather than an undefined statistic.

**Rationale:** the panel is ordered series-major, so consecutive rows are usually two different
customers. A lag-1 over the raw row order measures how much one customer's envelope differs
from the next one's - a statement about the customer mix. That mistake reported an
autocorrelation near 0.16 for the selected envelope, whose width is in fact constant per
customer, and would have failed `is_stable` on a completely smooth band. Returning `None` for
zero variation is equally wrong in the other direction: downstream it reads as "no stability
evidence" when it is the strongest evidence available.

**Consequence:** `is_stable` is YES for the published envelope, which is true.

**Status:** Accepted


## D-118 - A conditional-reference calendar baseline lost to persistence, and the reason is recorded

**Context:** four configurations were fitted on `CAL_FIT` and selected on `CAL_CONF` by lowest
weighted interval score among configurations inside the coverage tolerance.

**Selected:** `persistence` at the flat `horizon` granularity. Selection is on the weighted
interval score rather than width, because width alone rewards a band that covers nothing, and
uncalibrated configurations rank last.

**Rationale:** the calendar variants covered 0.43-0.66. Instructively, `calendar|horizon` had
a band 4.4 times *wider* than the selected one and covered **less** (0.6579 against 0.9150): a
mis-centred band is worse than a well-centred narrow one, because conditioning on a calendar
slot while centring on a pooled median puts a wide band around the wrong number. That is a
reason to distrust naive slot pooling, not a reason to distrust calendar conditioning in
general - the finer granularities were also the thinnest cells (median ~11-23 observations),
which on this dataset is indistinguishable from being less well estimated. The run records
this rather than presenting the winner as the only sensible choice.

**Consequence:** the published envelope conditions on one value per customer. Extending it to
a properly conditioned calendar reference is the first thing worth trying with more
calibration data, and it needs its own measurement rather than an assumption.

**Status:** Accepted


## D-119 - The domain layer validates its own magnitudes without a numeric library

**Context:** `FlexibilityEnvelope` was validating per-row magnitudes with numpy, which the
project's `tests/test_package.py` forbids in the domain layer.

**Selected:** read the grids structurally with a pure-Python `_as_float_grid`, returning shape
and a row-major flat list, and keep the magnitude fields typed `Any` so callers may pass lists,
tuples or arrays.

**Rationale:** the rule exists so plain-Python consumers can use the domain objects, and a
domain object that needs numpy to check its own invariants cannot serve them. The test caught
the violation, which is the argument for having the rule at all.

**Consequence:** the domain layer imports no ML library again, the invariant test passes, and
validation messages are slightly more verbose because shapes are read rather than coerced.

**Status:** Accepted


## D-120 - Scenario assumptions are a separate entry point and are barred from real-data paths

**Context:** a control workflow needs demonstration values for assets this dataset lacks, and
those values are assumptions.

**Selected:** `ScenarioAssumptions` with an explicit `scenario_mode` tag;
`assert_real_data_mode(scenario)` raises `ScenarioViolation` if a scenario reaches a real-data
path; battery usable energy and rated power must be supplied together; anything derived from a
scenario is tagged `basis=ASSUMED`, which can carry advisory authority and nothing more.

**Rationale:** the risk is not the scenario, it is a scenario value quietly reaching a measured
result. A gate that every real-data entry point calls turns that from a review question into a
type error. Requiring energy and power together stops a demo inventing a battery with unlimited
power, which is the specific fabrication most likely to be reached for.

**Consequence:** no scenario value appears in any Phase 9 result, and a future phase can
demonstrate dispatch without being able to contaminate a measurement.

**Status:** Accepted


## D-121 - Phase 10 is named `feature_ablation`, not `ablation`, because `ml/ablation.py` already exists

**Context:** `src/energy_intelligence/ml/ablation.py` is Phase 5's TCN architecture ablation
and is imported by `phase5.run_phase5_experiment`. A new package cannot take the name
`ablation` without shadowing it.

**Decision:** the new package is `src/energy_intelligence/ml/feature_ablation/`. The existing
module is untouched.

**Rationale:** renaming a module other phases import is a wider blast radius than choosing a
distinct name, and `ml/ablation.py` means something different from what Phase 10 does.

**Consequence:** `from ...ml.feature_ablation import select` is unambiguous alongside
`ml.ablation.run_ablations`.

**Status:** Accepted


## D-122 - One fit per feature set, scored on every fold

**Context:** the obvious implementation fits inside the fold loop. That fits six times per
feature set.

**Decision:** fit once on the train split, then score all six folds from the same fitted
object. `run_feature_set` returns one `FoldResult` per fold.

**Rationale:** the model does not depend on the fold - only the scoring rows do - so fitting per
fold multiplies cost for no additional information. Measured on this dataset one
`classical_hist_gbm` fit is 254s against a ridge fit of 0.4s, making the difference between
~21 minutes and ~2 hours for the classical grid. `run_one` is kept for single-fold callers and
for tests.

**Consequence:** the official classical grid completes in about 105s, which is what made it
possible to ship the whole phase inside the time budget.

**Status:** Accepted


## D-123 - `value_at_origin` belongs to the lag set, not the calendar set

**Context:** the five feature sets are a nested ladder, and the brief describes set B as
"autoregressive lags". `value_at_origin` is a fourth candidate that had to be placed.

**Decision:** B = `value_at_origin`, `lag_1`, `lag_4`, `lag_96`, `lag_672`.

**Rationale:** `value_at_origin` reads the target series at or before the origin, exactly as the
lag terms do. Putting it in A would make `A -> B` a comparison between "calendar only" and
"history except the present", which is not the question the increment is meant to ask, and would
make A a mixed family.

**Consequence:** C is a clean union of two disjoint families, and the incremental arithmetic
A -> C, C -> D, D -> E, A vs B is well defined.

**Status:** Accepted


## D-124 - The tie-break compares feature COUNT, and this was a real bug

**Context:** the simplicity tie-break must prefer the set with fewer **features**. The first
implementation compared `len(set_id)`.

**Decision:** compare `len(members(set_id))`.

**Rationale:** the set ids are `"A"` through `"E"`, so `len("A") == len("B") == 1` and every set
compared as the same size. The tie-break could therefore never fire. It was caught by a test
that asserted the tie-break applies to a near-tie, not by reading the code - the loop ran,
exited on the first iteration and reported a decision that looked reasonable.

**Consequence:** the tie-break works, and it changed the selected feature set for
`customer_load` from B to A. That is a material difference: B's mean MAE was 0.28% better, which
is inside the frozen 0.5% tolerance, so the 4-feature calendar set is the correct choice under
the rule that was frozen before any result was visible.

**Status:** Accepted


## D-125 - F05-F06 are called confirmation, never unseen test data

**Context:** the brief asks that F05/F06 be evaluated as post-ablation confirmation and never
called unseen test data.

**Decision:** the confirmation record carries a `held_out_caveat` field stating they are
contiguous blocks inside the validation range - held out from selection, but not the final test
split and carrying no unseen-test claim. A test asserts the phrase is present.

**Rationale:** they genuinely are held out from the selection, which is a real property worth
reporting. They are also not the benchmark, and conflating the two would let a confirmation
number be quoted as a test result.

**Consequence:** `reports/tables/feature_ablation_confirmation.md` carries the caveat, and the
completion report repeats it.

**Status:** Accepted


## D-126 - WIND is excluded and reported, not attempted and not substituted

**Context:** the brief names LOAD, WIND and PV as targets. `target_spec("wind_generation")`
returns status UNSUPPORTED: SMART-DS v1.0 contains no wind assets, wind speed appears only as a
weather covariate in the solar file, and `metrics.csv` reports 0 wind capacity.

**Decision:** exclude `wind_generation`, record the reason in the protocol freeze, in
`config/ablation/phase_10.yaml` under `excluded_targets`, and in the completion report. The
script refuses an explicit `--target wind` with that reason and exits non-zero.

**Rationale:** there is no wind series to forecast. Producing a wind number would require
inventing one, which is the failure mode this project has refused in every earlier phase.

**Consequence:** two of three targets run. The exclusion is visible in three places rather than
being an absence a reader has to notice.

**Status:** Accepted


## D-127 - The MLP robustness arm is recorded NOT RUN rather than approximated

**Context:** the brief asks for a secondary `PYTORCH_MLP_V1` robustness check with a frozen
configuration, but also states that MLP training must not prevent the classical ablation,
documentation, tests and console from shipping.

**Decision:** run the classical grid to completion, ship everything, and record the MLP arm as
**NOT RUN** in the config, the script's stderr, and the completion report's limitations.

**Rationale:** a half-trained MLP ablation is worse than no MLP ablation, because it produces a
number that looks like a robustness result and is not one. With a fixed one-hour budget the
classical grid, the console, the tests and the documentation are worth more than a
misleading secondary arm.

**Consequence:** Phase 10 ships PARTIAL on one arm and COMPLETE on the rest, stated plainly.
The frozen MLP configuration checksum is already in the protocol freeze, so the arm can be run
later without renegotiating the protocol.

**Status:** Accepted


## D-128 - The console shows flexibility as three lines and never as one number

**Context:** the frontend consumes backend state, and a console that printed
`flexibility: 4 kW` would undo Phase 9's central result.

**Decision:** `console flexibility` prints PHYSICAL, STATISTICAL and ASSUMED as three separate
lines, shows UNKNOWN per basis when Phase 9 has not run, and states that the proxy is never
reported as dispatchable capacity. A test asserts that the word "dispatchable" appears only
inside a negation.

**Rationale:** the risk is not that someone reads the docs and misunderstands; it is that a
summary line makes the distinction invisible at the moment someone is scanning for a number.

**Consequence:** the three-base property is enforced by a test rather than by convention.

**Status:** Accepted


## D-129 - Research tables are never written without data, and never from smoke rows

**Context:** the brief lists six table paths and warns against fake rows.

**Decision:** `write_tables` returns an empty mapping when a target has no official rows, skips
any row tagged `NON_EVIDENCE_SMOKE`, and the report's status line is PARTIAL whenever the grid
or the confirmation is incomplete. Tests assert both.

**Rationale:** an empty table that looks complete is worse than no table, because it invites a
reader to treat an absence as a zero.

**Consequence:** the PV and load tables exist because both targets produced official rows; no
table exists for WIND, and its absence is explained in the completion report.

**Status:** Accepted


## D-130 - H24 is horizon step 96, and the mapping is recorded in the freeze

**Context:** the brief names H24. The native resolution is 15 minutes, so 24 hours is 96 steps.

**Decision:** `horizon_steps = 96`, with `horizon_label = "H24"` computed from
`horizon_steps * 15 / 60` rather than typed in, and both recorded in the protocol freeze.

**Rationale:** a hard-coded "H24" next to a horizon of 96 can drift apart silently. Computing
the label from the steps makes a mismatch impossible to express.

**Consequence:** the freeze carries both, and the config asserts both.

