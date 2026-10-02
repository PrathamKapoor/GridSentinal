# Phase 1 System Flow

How the Phase 1 foundation actually works. Every function and file named here
exists in the repository.

Phase 1 has no energy-domain logic. The diagram below is the complete runtime
surface: configuration in, validated configuration and structured logs out.

```text
                  CLI invocation / import
                            │
                            ▼
              ┌──────────────────────────┐
              │ cli.main(argv)           │  src/energy_intelligence/cli.py
              │  build_parser()          │
              └────────────┬─────────────┘
                           │  argparse → Namespace
                           ▼
              ┌──────────────────────────┐
              │ health.check_health()    │  src/energy_intelligence/health.py
              └────────────┬─────────────┘
                           │  runs six independent checks
      ┌──────────┬─────────┼──────────┬───────────┐
      ▼          ▼         ▼          ▼           ▼
 check_      check_     check_     check_     check_test_
 python_     package_   config()   directories() infrastructure()
 version()   import()
      │          │         │          │           │
      │          │         │          │           │
      ▼          ▼         ▼          ▼           ▼
  stdlib    importlib  ┌──────────────────┐   find_spec("pytest")
  sys       .import_   │ config.load_config│   metadata.version
  .version  module      │      ()           │   glob("test_*.py")
                     └────────┬─────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ 1. load_raw_toml(default.toml)│  config/loader.py
              │ 2. load_raw_toml(--config)    │    tomllib.loads
              │ 3. os.environ[ENERGY_INTEL_   │
              │       CONFIG]  (if no --config)│
              │ 4. merge CLI overrides        │
              └───────────────┬───────────────┘
                              │  dict
                              ▼
              ┌───────────────────────────────┐
              │ config.coerce_app_config(     │  config/schema.py
              │   raw, project_root=...)      │
              │  - reject unknown keys        │
              │  - require every CONFIG_FIELD │
              │  - collect ALL errors         │
              └───────────────┬───────────────┘
                              │  AppConfig (frozen, slots)
                              ▼
              ┌───────────────────────────────┐
              │ check_logging(config)         │  health.py
              │  └─ logging_setup.            │
              │       initialize_logging()    │  logging_setup.py
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ HealthReport                  │
              │  .ok / .exit_code / render_   │  health.py
              └───────────────────────────────┘
```

---

## 1. Command line entry — `src/energy_intelligence/cli.py`

`main(argv)` is the single implementation behind both the `energy-intel` console
script and `python -m energy_intelligence` (`__main__.py` delegates to it).

```text
main(argv)
  │
  ├─ build_parser()                    argparse; adds --config, --log-level,
  │                                    --experiment-name, --seed
  ├─ parse_args(argv)                  + subcommands:
  │                                    health | config {show,validate}
  │                                    init-dirs | paths
  └─ dispatch on args.command
       │
       ├─ "health"     → _command_health(args)   → check_health(...) → render_report()
       ├─ "config"     → _command_config(args)   → load_config(...)   → JSON or "valid"
       ├─ "init-dirs"  → _command_init_dirs()    → ProjectPaths.ensure()
       └─ "paths"      → _command_paths()        → JSON presence map
```

`ConfigError` is caught at the `main` boundary and printed to stderr with exit
code 1. The CLI never lets an exception escape as a traceback.

`_overrides(args)` converts CLI flags into the config override mapping, dropping
`None` so an unspecified flag cannot blank a real value (D-011).

---

## 2. Project root discovery — `config/loader.py`

```text
find_project_root(start=None)
  │
  ├─ origin = Path(__file__).resolve()          # src/energy_intelligence/config/loader.py
  ├─ walk origin and every ancestor
  │    └─ accept when BOTH pyproject.toml AND src/energy_intelligence/ exist
  └─ raise ConfigError if no ancestor qualifies
```

The upward walk means `find_project_root()` works from any working directory,
including inside `src/`. It raises rather than falling back to
`Path.cwd()`, because silently resolving paths against an arbitrary directory
would produce a config that is wrong in a way nothing would report.

Result is cached at module import as `PROJECT_ROOT`; `CONFIG_DIR` derives from
it.

---

## 3. Configuration loading — `config/loader.py`

```text
load_config(path=None, *, overrides=None)
  │
  ├─ merged = {}
  ├─ merged.update(load_raw_toml(configs/default.toml))     # layer 1
  ├─ chosen = path  or  os.environ["ENERGY_INTEL_CONFIG"]   # layer 2/3
  │    └─ relative → resolved against PROJECT_ROOT
  │    └─ load_raw_toml(chosen) → reject unknown keys here
  ├─ merged.update({k: v for k, v in overrides.items() if v is not None})  # layer 4
  └─ return coerce_app_config(merged, project_root=PROJECT_ROOT)

load_raw_toml(path)
  ├─ not is_file()                 → ConfigError("Config file not found")
  ├─ tomllib.TOMLDecodeError       → ConfigError("... is not valid TOML")
  ├─ OSError                       → ConfigError("Could not read ...")
  └─ return parsed dict
```

Resolution order and precedence are specified in D-011.

---

## 4. Configuration validation — `config/schema.py`

`coerce_app_config` is pure: same input always yields the same output, and it
never touches the filesystem or creates directories.

```text
coerce_app_config(raw, *, project_root, source)
  │
  ├─ raw is not a dict                          → ConfigValidationError
  ├─ errors = []
  ├─ for key in raw − CONFIG_FIELDS             → "unknown configuration key"
  ├─ for key in CONFIG_FIELDS − raw             → "missing required configuration key"
  ├─ _coerce_seed(raw["seed"])                  int, not bool, 0 ≤ seed ≤ 2**32−1
  ├─ _coerce_choice(raw["device"], …)           ∈ {cpu, cuda, mps}, lowercased
  ├─ _coerce_choice(raw["log_level"], …)        ∈ {DEBUG,…,CRITICAL}, uppercased
  ├─ _coerce_experiment_name(raw[…])            matches ^[a-z0-9][a-z0-9._-]{0,63}$
  ├─ for key in _PATH_FIELDS
  │    └─ _coerce_path(raw[key], …)             non-empty str → expanduser()
  │                                            → absolute → .resolve()
  ├─ if errors: raise ConfigValidationError(errors)   # ALL errors at once (D-008)
  └─ return AppConfig(...)                      frozen, slots
```

`validate_app_config(config)` re-validates an already-built instance. It exists
for configs constructed by hand or crossing a process boundary; the loader's own
output is already validated.

`AppConfig.fingerprint()` returns a 12-character SHA-256 prefix over the
resolved settings, excluding only the `project_root` key. Because every path has
already been resolved to an absolute path containing that root, a fingerprint
identifies one concrete checkout — not just a set of relative values (D-009).

`_PATH_FIELDS` = `model_path`, `dataset_path`, `output_dir`, `artifact_dir`,
`log_dir`. None of them must exist (D-009).

---

## 5. Structured logging — `src/energy_intelligence/logging_setup.py`

```text
initialize_logging(config, *, console=True, to_file=True, force=False) → Path | None
  │
  ├─ logger = logging.getLogger("energy_intelligence")
  ├─ existing = handlers tagged _energy_intelligence_managed
  ├─ if existing and not force: return _active_log_file()   # idempotent
  ├─ if existing: detach and close them                     # ours only (D-013)
  ├─ module state ← run_id = uuid4().hex[:8]
  │                 experiment = config.experiment_name
  │                 config_id  = config.fingerprint()
  ├─ logger.setLevel(config.log_level); logger.propagate = False
  ├─ logger.addFilter(RunContextFilter())
  ├─ if console: StreamHandler(stdout) + ConsoleFormatter + RunContextFilter
  ├─ if to_file: mkdir(log_dir)
  │              FileHandler(log_dir/<experiment_name>.jsonl, mode="a")
  │              + JsonLinesFormatter + RunContextFilter
  │              OSError → log_file = None, log a warning, keep running
  ├─ logger.info("logging initialised", extra={event, log_file})   # INFO: run marker
  └─ return log file path or None
```

Handler ownership: only handlers tagged `_energy_intelligence_managed` are
counted, reset or closed. A host handler (pytest `caplog`, a Jupyter kernel) can
neither disable setup nor be removed by `reset_logging()` (D-013).

`RunContextFilter` injects `run_id`, `experiment` and `config_id` onto every
record, so no call site can forget them.

`JsonLinesFormatter.format()` emits `timestamp` (UTC ISO-8601, millisecond
precision), `level`, `logger`, `message`, `experiment`, `config_id`, `run_id`,
plus every non-reserved attribute set via `extra=` as a top-level key. Values
that are not JSON primitives are `repr()`-stringified. `exc_info` becomes an
`exception` string.

`reset_logging()` detaches our handlers, clears the filters and nulls the run
identity. `shutdown_logging()` detaches and calls `logging.shutdown()`.

`get_logger(name)` always returns a logger under the `energy_intelligence.*`
namespace, so third-party and stdlib loggers are never captured by these
handlers.

---

## 6. Health check — `src/energy_intelligence/health.py`

```text
check_health(config_path=None, *, paths=None, create_missing_dirs=False) → HealthReport
  │
  ├─ resolved_paths = paths or ProjectPaths.from_root()
  ├─ if create_missing_dirs: resolved_paths.ensure()
  ├─ config_result, config = check_config(config_path)
  └─ run six checks, always all six (never short-circuit)
       │
       ├─ check_python_version()   sys.version_info ≥ (3, 12)
       ├─ check_package_import()   import energy_intelligence;
       │                           warn if resolved inside site-packages
       ├─ check_config()           load_config(); on failure returns (result, None)
       ├─ check_directories()      ProjectPaths.missing() → must be empty
       ├─ check_test_infrastructure()
       │                           find_spec("pytest"), metadata.version,
       │                           glob("tests/test_*.py") — must be non-empty
       └─ check_logging(config)    initialize_logging(force=True) → write a probe
                                   record → flush → assert the file grew.
                                   reset_logging() in a finally block.
```

```text
HealthReport
  .ok          → no check has status "fail"     (warnings do not fail it)
  .exit_code   → 0 if ok else 1                (CI gate)
  .failures()  → results with status "fail"
  .warnings()  → results with status "warn"

render_report(report) → str
  ├─ sorts by CheckResult.sort_key so "ok" first and problems at the bottom
  ├─ prints [PASS|WARN|FAIL] name, indented detail, and a fix line for problems
  └─ prints a "N passed, M warning(s), K failed - HEALTHY|UNHEALTHY" summary
        plus the config fingerprint
```

`check_config()` returns a `(CheckResult, AppConfig | None)` tuple so the config
is loaded once and reused by `check_logging()`. When configuration fails,
`check_logging(None)` returns a *warning* rather than an error: the missing
logging is a consequence of the config failure, not a second independent fault.

---

## 7. Filesystem conventions — `src/energy_intelligence/paths.py`

```text
ProjectPaths.from_root(root=None)   # default: detected PROJECT_ROOT
  ├─ root, configs, logs, models, experiments, artifacts,
  │  raw_data, processed_data, external_data          (frozen, slots)
  ├─ .tracked_directories   configs models experiments artifacts data/*
  │                         — the set git must keep via .gitkeep (D-016)
  ├─ .required_directories  tracked + logs             — must exist at runtime
  ├─ .missing()             required dirs that do not exist
  ├─ .ensure()              mkdir(parents=True) for each missing; returns created
  └─ .experiment_dir(name)  experiments/<name>
```

---

## 8. Tests — `tests/`

```text
conftest.py
  autouse  _isolate_logging     reset_logging() before AND after every test,
                                so process-wide logger state cannot leak
  fixtures  config_table        valid table + overrides (_Remove sentinel deletes)
            make_config         validated AppConfig rooted at tmp_path
            tmp_paths           ProjectPaths at tmp_path, dirs created
            write_toml          writes a TOML file into tmp_path

test_package.py     (  7) import, __all__, __main__, zero-dependency guard
test_config.py      ( 84) every validation rule, layering, env var, root discovery
test_logging.py     ( 21) JSONL structure, run context, idempotency, host handlers
test_health.py      ( 22) every check passing AND failing; report rendering
test_paths.py       ( 10) layout, missing/ensure, real-checkout assertions
test_cli.py         ( 16) exit codes, JSON output, overrides, error handling
                     ----
                     160
```

Every test asserts behaviour that exists. There are no placeholder or
"AI-behaviour" tests, because there is no AI implementation to test.

Two tests are regression guards on recorded decisions and would otherwise fail
silently if those decisions were quietly reversed:

* `test_runtime_dependencies_are_empty` and
  `test_no_ai_or_ml_dependencies_are_imported_at_import_time` — enforce D-004.
* `test_host_handlers_do_not_disable_initialisation` and
  `test_reset_logging_leaves_host_handlers_alone` — enforce D-013.

---

## Verification commands actually run

```powershell
uv venv --python 3.12
uv sync --extra dev
uv run pytest                                             # 160 passed
uv run pytest --cov --cov-report=term-missing             # 160 passed, 94%
uv run python -m energy_intelligence health                # 6 passed, HEALTHY, exit 0
uv run energy-intel health --json                          # "ok": true, exit 0
uv run python -m energy_intelligence config show
uv run python -m energy_intelligence config validate
```

---

# Phase 2 — Energy domain flow

Everything above is the Phase 1 foundation, unchanged. What follows is the
energy domain added in Phase 2, by real module and function name.

```text
                      configs/domain.toml
                              │
                              ▼
              config/domain.py :: load_domain_config()
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
             TimeConfig            EnergySystemConfig
                    └─────────┬──────────┘
                              ▼
                       DomainConfig
                              │
                              │  (fingerprint recorded in logs)
                              ▼
     ┌────────────────────────────────────────────────────────┐
     │  energy_intelligence.domain  — the formal contract      │
     ├────────────────────────────────────────────────────────┤
     │  errors.py      DomainError / DomainValidationError      │
     │  enums.py       the controlled vocabulary                │
     │  identifiers.py AssetId, NodeId, ConstraintId, ...        │
     │  quantities.py  Quantity(value, Unit)                    │
     │  provenance.py  SourceReference/ProvenanceEvent/          │
     │                 ProcessingStep/Provenance                  │
     │  quality.py     DataQuality(flags)                       │
     │  timebase.py    TimeBase.step_time() / step_index()       │
     │  uncertainty.py UncertaintyEstimate                      │
     │  observations.py ObservationRecord  ── role: OBS/DERIVED  │
     │  topology.py    Node / NetworkElement / NetworkTopology   │
     │  assets.py      Asset / Battery / SolarAsset / WindAsset  │
     │                 EVCharger / FlexibleLoad                  │
     │  state.py       NodePower / StorageState / DemandState /  │
     │                 RenewableState / EVState / GridState /    │
     │                 EnergyState                               │
     │  actions.py     Action  ── role: ACTION, authority-gated  │
     │  constraints.py Constraint  ── declared, never enforced  │
     │  objectives.py  Objective  ── never weighted             │
     │  forecasts.py   Forecast                                 │
     │  system.py      EnergySystem  ── referential integrity   │
     │  serialization.py encode() / decode()                     │
     └────────────────────────────────────────────────────────┘
                              │
                              ▼
         validation happens in every __post_init__
         (an invalid energy state cannot be constructed)
```

## 1. Configuration

```text
load_domain_config(path=None)
  │
  ├─ target = path or configs/domain.toml   (relative → PROJECT_ROOT)
  ├─ tomllib.loads(target.read_text())
  ├─ reject unknown top-level keys          → ConfigValidationError
  ├─ require [time] and [energy_system]     → ConfigValidationError
  ├─ reject unknown keys inside each section → ConfigValidationError
  ├─ TimeConfig(timestep_minutes ∈ {1,5,15,30,60}, horizon_steps>0, origin)
  ├─ EnergySystemConfig(system_id validated as SystemId, name, description,
  │                     topology_kind ∈ {aggregate, feeder})
  └─ return DomainConfig(time, energy_system, data_anchor)
```

`AppConfig` is deliberately **not** widened — see D-037.

## 2. Construction and validation

Every domain object validates itself in `__post_init__` and is
`frozen=True, slots=True`, so an invalid object cannot exist and cannot be
mutated into an invalid one. Errors accumulate into `DomainValidationError`.

```text
Quantity(value, unit)        finite, numeric, known unit
SourceReference(...)         source_id, dataset, locator, version all non-empty
Provenance(source, events)   source required; ≥1 event required  ← R4, D-032
DataQuality(flags)           ≥1 flag; ok cannot combine; flags may co-occur
TimeBase(step, horizon, origin)
                             step>0, horizon>0, origin timezone-aware
UncertaintyEstimate(...)     kind is UNKNOWN or exactly one numeric summary
Node / NetworkElement / NetworkTopology
                             ids valid; parent/endpoints resolve; a grid
                               connection node has no parent; ≥1 grid connection
Asset hierarchy              availability ∈ [0,1]; SOC window ordered; wind speeds
                             cut_in<rated<cut_out; unit-consistent power bounds
NodePower(...)               kW units; residual within POWER_BALANCE_TOLERANCE
StorageState                 0≤SOC≤1; not charging and discharging at once
DemandState                  category breakdown sums to the total
EVState                      deferrable ≤ charging
EnergyState                  origin required; provenance required; per-node
                               balance closes; state covers every node
Action                       role must be ACTION; issued_under must be a
                               controllable level AND in the sufficiency set for
                               the action type; setpoint unit and sign correct;
                               duration ≥1; provenance required
Constraint                   ≥1 asset or node; bounds ordered; unit matches
                               the category's natural unit
Objective                    measurement and why_it_matters required;
                               NO weight field                          ← D-031
Forecast                     issued_at < target_time; role is OBSERVATION;
                               uncertainty unit matches the value unit
```

## 3. Cross-entity validation — `system.py`

```text
EnergySystem.__post_init__()
  └─ _referential_errors()
       ├─ no duplicate asset_id
       ├─ every asset's node exists in the topology
       ├─ every constraint's asset_id / node_id resolves
       └─ no duplicate constraint_id / objective_id

system.validate_state(state)            → validate_state_against_system()
       ├─ state covers EXACTLY the system's nodes
       ├─ storage reports only assets the system owns as a battery
       └─ renewables report only assets the system owns as solar/wind

system.validate_actions(actions)        → validate_actions_against_system()
       ├─ target asset exists
       ├─ target asset is controllable (ADVISORY_ONLY → refused)
       ├─ issued authority is in sufficient_authorities_for(action_type)
       └─ all failures reported together, not one at a time

system.validate_observations(records)   → asset/node references resolve
system.validate_forecasts(forecasts)   → horizon within the configured grid
```

## 4. Observation versus action — the boundary

```text
ObservationRecord(variable, role, value, quality, provenance, timestamp, scope)
  └─ role ∈ {OBSERVATION, DERIVED}
     role == ACTION  → DomainValidationError

Action(action_id, action_type, target_asset_id, setpoint, role,
       issued_under, start_step, duration_steps, provenance)
  └─ role == ACTION
     role ∈ anything else → DomainValidationError

Forecast(...)
  └─ role == OBSERVATION
```

This is what makes battery SOC (observation) structurally incapable of becoming a
commanded variable. See D-034.

## 5. Serialization — `serialization.py`

```text
encode(obj) → to_dict(obj) → _encode_value() recursively
           → {"_schema_version": SCHEMA_VERSION, **payload}
           → json.dumps(sort_keys=True, separators=(",",":"))

decode(text) → parse JSON
             → require _schema_version, refuse a mismatched one
             → from_dict(body) dispatch by discriminating field
                 EnergySystem  ← system_id
                 Forecast      ← forecast_id
                 Action        ← action_id
                 Constraint    ← constraint_id
                 Objective     ← objective_id
                 EnergyState   ← node_power
                 Observation   ← variable + quality
                 NetworkElement← element_id
                 Asset         ← asset_type      (BEFORE the Node check:
                                                   an asset also carries node_id)
                 NetworkTopology ← nodes
                 Node          ← node_id
```

Round-trip and byte-determinism are enforced by tests over twelve object kinds.
See D-036.

## 6. Runtime inspection

```text
energy-intel domain validate          → validate configs/domain.toml
energy-intel domain show             → resolved domain config as JSON
energy-intel domain vocabulary       → the whole domain vocabulary as JSON,
                                       derived from the enums at runtime so it
                                       cannot drift from the code

energy-intel health                  → now includes a 7th check:
  check_domain_config()
    ├─ load_domain_config() succeeds
    ├─ the configured grid builds a domain TimeBase
    └─ the domain model round-trips (a contract that cannot be persisted
       cannot be handed between phases)
```

## 7. Test layout

```text
tests/conftest.py                              Phase 1 fixtures (unchanged)
tests/test_package.py            (  7)  import, __all__, __main__, deps guards
tests/test_config.py             ( 84)  validation rules, layering, env var, roots
tests/test_logging.py            ( 21)  JSONL structure, run context, host handlers
tests/test_health.py             ( 22)  every check passing AND failing; rendering
tests/test_paths.py              ( 10)  layout, missing/ensure, checkout assertions
tests/test_cli.py                ( 16)  exit codes, JSON output, overrides, errors
                                  -----
                                  160   Phase 1 (unchanged, 0 regressions)

tests/domain/conftest.py                reference environment fixtures
tests/domain/test_values.py             ( 94) identifiers, Quantity, provenance,
                                              quality, TimeBase, uncertainty
tests/domain/test_topology_assets_state.py ( 93) topology, assets, state, actions,
                                              constraints, objectives, observations
tests/domain/test_state_components.py   ( 40) remaining EnergyState branches
tests/domain/test_system_serialization.py ( 52) cross-entity validation, Forecast,
                                              encode/decode
tests/domain/test_domain_config.py      ( 39) domain configuration strictness
tests/domain/test_domain_cli.py         ( 14) domain CLI + health integration
                                         -----
                                         332   Phase 2
                                         =====
                                         492   total
```

