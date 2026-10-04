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


---

# Phase 3 — Data ingestion flow

Everything above is Phases 1-2. What follows is the ingestion pipeline, by real
module and function name.

```text
                       configs/data.toml
                              |
                              v
                config/data.py :: load_data_config()
                              |
                              v
                data/smartds/layout.py :: SmartDsLayout
                              |
                              v
    +-------------------------+--------------------------+
    |                         |                          |
    v                         v                          v
smartds/adapter.py       smartds/schema.py        smartds/mapping.py
  parse_dss()             discover_feeder_schema()  SMART_DS_MAPPINGS
  load_buscoords()        discover_csv_schema()
  _Parsed                 discover_profile_schema()
    |                         |                          |
    +------------+------------+                          |
                 |                                       |
                 v                                       |
      adapter.loads(indices)  adapter.topology()         |
      adapter.der_assets()    adapter.time_axis()        |
                 |                                       |
                 v                                       |
      smartds/domain_mapper.py :: map_to_domain()        |
                 |                                       |
                 v                                       |
      Phase 2 domain objects (EnergySystem)              |
                 |                                       |
                 v                                       |
      smartds/balance.py :: validate_feeder_balance() <--+
                 |
                 v
      BalanceReport  +  all reports written to artifacts/data/
```

## 1. Configuration

```text
load_data_config(path=None)
  |-- target = path or configs/data.toml   (relative -> PROJECT_ROOT)
  |-- tomllib.loads(...)
  |-- reject unknown keys                    -> ConfigValidationError
  |-- require dataset/version/year/region/subregion/scenario/substation/feeder
  |-- raw_root: relative -> PROJECT_ROOT, expanduser()
  |-- peak_only: bool, default True
  `-> DataConfig
```

Three configuration files now exist, each a distinct concern:

| File | Concern |
|---|---|
| `configs/default.toml` | **a run** - seed, device, paths, logging |
| `configs/domain.toml` | **the environment** - time base, system boundary |
| `configs/data.toml` | **the dataset** - what external data is ingested |

Keeping the third separate follows D-037: a machine can select a dataset without
inheriting settings it does not use.

## 2. Layout resolution — `data/smartds/layout.py`

`SmartDsLayout` is the only object that knows the dataset's directory
convention. It builds `profiles_dir`, `solar_data_dir`, `placements_dir`,
`feeder_dir` and friends, maps SMART-DS names to Phase 2 identifiers
(`sanitize_identifier`), and exposes `s3_prefix()` for provenance locators.

```text
sanitize_identifier("p1uhs0_1247--p1udt12703") -> "p1uhs0-1247-p1udt12703"
layout.feeder_node_id("p1udm971")            -> NodeId("node-smartds-p1udm971")
```

## 3. Parsing — `data/smartds/dss.py` and `buscoords.py`

`parse_dss()` is **dataset-agnostic**: it knows the OpenDSS grammar and nothing
about SMART-DS. It is strict — a malformed directive raises rather than being
skipped, because a silently dropped `New Storage.*` line would make a battery
vanish with no trace.

```text
split_directives(text)
  |-- strip // and ! comments
  |-- a "new" directive implicitly terminates the previous element
  `-- returns ordered (verb, remainder)

_parse_params(remainder)
  |-- head must be "Class.name"           else raise
  |-- scan for key=value boundaries via _KEY_VALUE_RE
  |     the key may be percent-prefixed (%Cutout, %EffCharge) - see D-049
  |     a value may contain '=' (mult=(file=../../x.csv)) and is not split on it
  `-- preserve source order in .order
```

`DssObject.get_float()` raises on a non-numeric value rather than defaulting: a
silently defaulted rating would corrupt every downstream energy total.

`parse_buscoords()` is **separate on purpose**. `Buscoords.dss` contains bare
`<bus> <lon> <lat>` lines with no directives, so `parse_dss` returns zero
elements and every node would silently appear to have no location.

## 4. Schema discovery — `data/smartds/schema.py`

Everything reported is derived by reading the files.

```text
discover_dss_schema(path)        -> classes, per-class parameter union, element count
discover_csv_schema(path)        -> columns, inferred type, min/max, missing, distinct
discover_profile_schema(path)    -> rows, range, is_normalised
discover_buscoords_schema(path)  -> count, fields, units, example
discover_feeder_schema(layout)   -> all of the above for one feeder,
                                    ABSENT files reported as absent, not assumed empty
```

## 5. Normalization — `data/smartds/normalize.py`

Three things, each recorded as provenance.

```text
TimeAxis(timestep_minutes, points, year, assumed_utc)
  |-- built from LoadShapes npts, NOT from a constant
  |-- timestamp(index) / index(timestamp) round-trip exactly
  |-- rejects an off-grid timestamp rather than rounding it
  `-- matches_domain_grid is True (15 min, multiple of 96)

P(t) = sum over customer members of (kW_rating * profile_pu(t))
```

`TIMEZONE_ASSUMPTION` is exported and embedded in every time-axis description
and provenance chain (D-045).

## 6. Adapter — `data/smartds/adapter.py`

The facade. Dataset-specific knowledge stops here.

```text
SmartDsAdapter(layout)
  |
  |-- parsed()        -> _Parsed(loads, shapes, lines, transformers, pv, storage, base_kv)
  |                     each file parsed once and cached
  |-- time_axis()     -> TimeAxis from LoadShapes npts
  |                     raises if no npts is declared: the vector length is unknown
  |-- loads(indices)  -> LoadCatalogue
  |     group by base name, stripping the _1/_2 centre-tap suffix
  |     SUM both members (D-043)
  |     series built only at the requested indices
  |     a missing profile -> recorded as a QualityFinding, load listed as unmapped
  |-- topology()      -> TopologyCatalogue
  |     edges from Lines bus1/bus2 with phase suffixes stripped
  |     every discovered element bus added, so every load, PV and storage
  |     point is a declared node (D-053)
  |     the true bus name carried alongside each sanitised NodeId
  |     Buscoords attached when present, absence reported
  |-- source_bus()    -> the Circuit element's bus1 from Master.dss, i.e. the
  |                     point of common coupling. Read, never inferred
  |-- der_assets()    -> DerAssetCatalogue
  |     solar  from PVSystems Pmpp
  |     battery from Storage kWRated / kWhRated / kWhStored / %EffCharge /
  |     %EffDischarge as a BatteryAssetSpec (round-trip = charge x discharge)
  |     EV sites counted from placement JSON
  |     battery dispatch, usable energy, SOC window and directional limits
  |     reported as a MISSING finding (D-046)
  `-- quality_report() -> QualityReport(findings, axis, mappings)
```

## 7. Domain mapping — `data/smartds/domain_mapper.py`

The only place that constructs Phase 2 objects from real data.

```text
map_to_domain(adapter, load_catalogue, topology_catalogue, der_catalogue,
              demand_kw, demand_quality, summary_customer_kw, event_timestamp)
  |
  |-- Nodes          from discovered buses; exactly one is_grid_connection.
  |                   The PCC is READ from Master.dss `Circuit.bus1`, never
  |                   inferred from the folder name (D-051). If it is missing or
  |                   absent from the graph, raise UnresolvedBoundaryError -
  |                   no bus is ever promoted to PCC by guesswork.
  |-- NetworkElements one per Line edge, from/to resolved
  |-- Assets         ONE Asset per customer (NOT_CONTROLLABLE), each on its own
  |                  connection bus, members and centre-tap status in metadata,
  |                  flexibility recorded as NOT STATED (D-052).
  |                  SolarAsset per PVSystem (CURTAILABLE).
  |                  Battery per Storage (DISPATCHABLE): nameplate_energy only;
  |                  usable_energy, min_soc, max_soc and both directional power
  |                  limits are None because the dataset states none (D-050).
  |                  NO EV asset: a charging profile cannot be invented.
  |-- Observations   total_demand_kw as DERIVED, and the dataset's own
  |                  published_customer_demand_kw as OBSERVATION, both at the
  |                  real event timestamp with mandatory provenance.
  |-- TimeBase       timestep and origin from the discovered axis
  `-- EnergySystem   with mandatory Provenance pointing at the real SMART-DS path
```

Every field the source does not state arrives as `None` and is named in
`MappingReport.unknown_fields`. `None` means *not known*; it never means zero.

`availability = 1.0` for every asset, with the basis recorded in
`MappingReport.availability_basis`: SMART-DS models everything as in service,
which is a modelling fact, not a measurement (G-08).

## 8. Balance validation — `data/smartds/balance.py`

```text
validate_feeder_balance(summary, reconstructed_*, node_count)
  |
  |-- identity 1: reconstruction fidelity, total customer demand
  |-- identity 2: reconstruction fidelity, commercial
  |-- identity 3: reconstruction fidelity, residential
  |-- identity 4: published internal, losses + customer = circuit
  |-- identity 5: published internal, commercial + residential = customer
  |
  `-- residual statistics: max / mean / median / p95, count above tolerance
```

Identities 1-3 test **our ingestion**. Identities 4-5 test **SMART-DS's own
consistency**. They are reported separately because conflating them would hide
which one is failing.

`TOLERANCE_KW = 0.5` is a module constant with no setter. Metrics that could not
be computed are `None`, never zero. Unavailable evaluations are listed in
`report.unavailable` with their reason.

## 9. Pipeline and artifacts — `data/smartds/pipeline.py`

```text
run_ingestion(layout, peak_only)
  |-- build_manifest()  -> SHA-256 per acquired file, content digest excluding time
  |-- adapter.time_axis() -> native grid
  |-- read Summary_data.csv -> published peak index
  |-- adapter.loads(indices=(peak_index,)) / topology() / der_assets()
  |-- map_to_domain()      -> EnergySystem
  |-- validate_feeder_balance()
  `-- artifacts written to artifacts/data/:
        manifest.json  schema_report.json  quality_report.json
        balance_report.json  reconstruction.json  mapping_report.json
        energy_system.json

`mapping_report.json` also carries the two ObservationRecords, so a reader can
tell a reconstruction from the dataset's own published figure without opening
a second file.
```

The provenance event timestamp is the **data's** timestamp, not wall-clock time,
so a re-run produces a byte-identical `energy_system.json`. Acquisition time
lives in the manifest, documented as the one nondeterministic field.

## 10. CLI

```text
energy-intel data config     print the resolved dataset selection
energy-intel data inspect    schema + assets + quality, no domain mapping
energy-intel data ingest     full pipeline, writes all artifacts
energy-intel data validate   balance report; exit 1 when it FAILS
energy-intel data mapping    the full field-mapping table as JSON
energy-intel data manifest   manifest summary with the content digest
```

A missing dataset exits 1 with a pointer to `docs/smart_ds_acquisition.md`
rather than crashing.

## 11. Tests — `tests/data/`

```text
conftest.py                     layout built from real extracted fixtures
fixtures/                       verbatim excerpts of real SMART-DS v1.0 files
test_parsing.py                  OpenDSS + Buscoords + CSV parsing, manifest, digests
test_adapter_and_balance.py      time axis, loads, topology, DER, mapping,
                                 what must NOT be invented, balance, plus
                                 integration against the full real dataset
```

Fixtures are produced by `scripts/extract_fixtures.py`, which records every file
and every truncation in `FIXTURE_PROVENANCE.json` (D-048).

---

# Phase 4 — ML pipeline

Everything above is Phases 1-3. What follows is the machine-learning pipeline, by
real module and function name.

```text
data/raw/smart_ds/v1.0/...                       (acquired in Phase 3)
        │
        ├── loads/*.csv  Loads.dss  →  adapter.loads(names=…)      DERIVED
        └── solar_data/*.csv                      →  solar_target()  OBSERVED
                            │
                            v
              ml/targets.py :: TargetSeries
              (values, series_ids, scale, value_origin, provenance)
                            │
                            v
              ml/dataset.py :: build_dataset(TargetSeries, DatasetBuildConfig)
                 ├── ml/features.py :: available_feature_names() → what is legal
                 ├── ml/features.py :: build_feature_matrix()      → X
                 ├── ml/splits.py   :: split_boundaries()          → 70/15/15
                 ├── ml/splits.py   :: per-split origin windows    → no crossing
                 ├── ml/features.py :: assert_no_future_access()   → leakage proof
                 └── ml/dataset.py  :: save_dataset()  → artifacts/ml/<label>/dataset
                            │
                            v
              ml/experiment.py :: run_experiment(dataset, model, seed, series)
                 ├── ml/scaling.py     FeatureScaler.fit(train only) → assert
                 ├── ml/baselines/naive.py      fit_naive / naive_forecast
                 ├── ml/baselines/classical.py  fit_classical
                 ├── ml/baselines/neural.py     fit_neural
                 └── ml/metrics.py    evaluate() → MetricSet
                            │
                            v
              ml/analysis.py :: regime_report(), error_analysis()
                            │
                            v
       artifacts/ml/<label>/reports/*.json + experiments/registry.jsonl
```

## 1. Target extraction — `ml/targets.py`

`TARGET_REGISTRY` declares every candidate the brief listed, each with source, unit,
resolution, coverage, missingness, granularity, feasibility, status and **evidence**.
Three targets are extracted; four are refused.

| Target | Extractor | Value origin | Scale (per-unit divisor) |
|---|---|---|---|
| `customer_load` | `load_target()` | `DERIVED` | the customer's summed `kW` rating |
| `feeder_load` | `feeder_load_target()` | `DERIVED` | total connected `kW` |
| `pv_generation` | `solar_target()` | `OBSERVED` | the 1000 kW array rating in the column name |

`select_customer_sample()` draws a stratified, seeded subset — stratified by class and
rated-power decile, because the first N names in sorted order are one neighbourhood
and would bias both class mix and magnitude.

`feeder_load_target()` accumulates **per profile** (341 files weighted by the summed
kW of every Load object using them) rather than per customer, which avoids holding
1,871 whole-year series in memory. A test asserts it equals Phase 3's per-customer sum
to 1e-6 kW at the published peak.

**Error behaviour:** an unknown target raises `KeyError` listing the declared ones; a
request for an `UNSUPPORTED` target raises with the missing quantity and its
evidence; a solar file with the wrong row count raises rather than being padded;
missing columns raise rather than being substituted; zero-rated-power series raise
rather than producing infinite per-unit values.

## 2. Features and the availability policy — `ml/features.py`

`FEATURE_CATALOGUE` declares 17 features with formula, reason, source, availability
(`at_or_before_origin` or `origin_only`), value kind and minimum lookback.

`available_feature_names(lookback, max_horizon, has_weather)` resolves what is
actually usable and returns **both** the selection and the withdrawals with reasons,
so the dataset manifest records why a feature is missing rather than a shorter list
appearing without explanation. Weather is withdrawn beyond one step, because the
dataset has no weather *forecast*.

`build_feature_matrix()` gathers lags, trailing windows and calendar values. The
trailing window is built by gather over `[t-96, t-1]`, so it provably cannot include
`t`.

`assert_no_future_access()` is the leakage guard. For a deterministic sample of
positions it rebuilds that one row twice — once from the real series, once from a copy
where everything **strictly after** that row's origin is `NaN` — and requires
byte-identical results. It is per sample on purpose: poisoning a whole block also
destroys later samples' legitimate inputs and would prove nothing.

**Error behaviour:** an unknown feature name, a wrong-length builder result, a
non-finite value, an origin earlier than the required history, and a `steps_per_day`
that is not a whole number of hours all raise rather than degrade.

## 3. Splits — `ml/splits.py`

`split_boundaries()` cuts the grid into three contiguous ranges. `sample_splits()`
assigns a sample only when its **entire** lookback and target windows fit inside one
range, and marks anything else `-1` (dropped). `build_dataset()` constructs origins
per split so nothing needs filtering afterwards, then re-derives the assignment with
`sample_splits()` and asserts the two agree — a belt-and-braces check on the single
most important property in the phase.

## 4. The dataset artifact — `ml/dataset.py`

One flat row per `(series, origin)`: `features`, `targets`, `origin_index`,
`series_index`, `split`, plus `series_scale` and the manifest. `dataset_version()` is
content-addressed over target, lookback, horizons, split fractions, stride, weather
flag and series identity, so any shape change produces a new version.

Saved as `<version>.npz` (arrays) beside `<version>.json` (manifest with feature
definitions, withdrawals, split boundaries, provenance, normalisation basis and an
array digest).

## 5. Scaling — `ml/scaling.py`

`FeatureScaler.fit(train_rows)` then `transform` on everything.
`assert_train_only_fit()` raises unless the fitted row count equals the training row
count exactly. Constant columns map to scale 1.0 and become exactly zero rather than
dividing by zero.

## 6. Baselines

| Module | Models | Notes |
|---|---|---|
| `baselines/naive.py` | `last_value`, `seasonal_day`, `seasonal_week`, `drift` | unfitted; seasonal offsets are exact `t + h − m`, asserted in tests |
| `baselines/classical.py` | `ridge`, `hist_gbm` | ridge exposes a coefficient table, which is the interpretability diagnostic |
| `baselines/neural.py` | `mlp`, `gru` | one conventional baseline each; fixed seeds; no search |

Naive models are given the **real series history** rather than approximated from
feature columns, because the seasonal offset depends on the horizon and an
approximation would misalign it by up to `h` steps. Asking for a naive model without
the series raises.

## 7. Evaluation — `ml/experiment.py`, `ml/metrics.py`

The runner fits on training rows only, predicts the test split, converts predictions
back to kW by multiplying by the row's own scale, and reports MAE, RMSE, sMAPE, R²,
peak-decile error, ramp error, zero-period error, daytime error, plus the per-unit MAE
and the sample count.

Ramp error is computed only where two **consecutive** horizons exist, and is `n/a`
otherwise. `_series_context()` adds per-series identity columns at fit time from
training rows only — never written into the dataset, because they depend on the split.

## 8. Analysis and artifacts — `ml/analysis.py`, `ml/runner.py`, `ml/registry.py`

`assign_regimes()` derives regime axes from the target (demand terciles, ramp terciles;
solar phase from the generation derivative). `regime_report()` reports per-regime
error and the worst-to-best ratio. `error_analysis()` reports worst rows with
timestamps, worst series, and the error profile by hour and month.

`run_phase4()` writes the dataset, the comparison table (Markdown and JSON), the
regime report, one error report per model, a summary, and one registry record per
experiment.

## 10. Phase 5: Energy Demand Dynamics Model

### 10.1 What it adds

Phase 4 established *whether* machine learning helps on this dataset and where. It
answered that a classical histogram gradient-boosting model beats both neural
baselines. Phase 5 asks a different question: **does the order of the history carry
information that a flat vector of lags discards?**

```text
Phase 4 input   y(t-96) ... y(t-24), y(t-4), y(t-1)   -> 13 features, order-free
Phase 5 input   [y(t), dy/dt, sin/cos hour, sin/cos day-of-year] x 168 tokens, ordered
```

### 10.2 Pipeline

```text
TargetSeries (kW per customer)
  -> normalise by each series' own rated kW        per-unit, scale known for the horizon
  -> build_sequence_index()                          ordered windows, per-series expansion,
                                                      chronological split, causality guard
  -> token stride 4, INPUTS ONLY                     672 native steps -> 168 tokens;
                                                      targets stay at native 15-minute steps
  -> model window x [B, C=6, T=168]
  -> TCN residual blocks, dilations 1..64, GroupNorm within the window
  -> per-series embedding added to the trunk
  -> 3 linear heads -> deltas at h = 1, 4, 96
  -> yhat(t+h) = y(t) + delta(t+h)                   D-074, the delta parameterisation
  -> per-unit -> kW by the row's own scale
  -> evaluate against Phase 4's artifact metrics
```

Training origins are strided by 4 for compute; validation and test origins keep stride
1, so the `179,520` evaluated test rows are **the same rows** Phase 4 scored
(D-073). Checkpoints are selected on validation per-unit MAE; the test split is
touched once, after selection (D-075).

### 10.3 Modules

| Module | Responsibility |
|---|---|
| `ml/sequence.py` | `SequenceConfig`, `build_sequence_index()`, the causality guard, the channel contract, `sequence_version()` |
| `ml/temporal.py` | `build_model()` for `tcn` and `transformer`; parameter counting; config recording |
| `ml/training.py` | `fit_temporal_model()`, checkpoint save/restore, early stopping, `evaluate_checkpoint()` |
| `ml/phase5.py` | the runner, `analyse_predictions()`, `replay_evaluation()`, the Phase 4 comparison |
| `ml/ablation.py` | the reduced-budget ablation suite and its Markdown table |
| `config/temporal.py` | strict config loading; every ablation must declare the question it answers |

### 10.4 Results on the shared test split

kW MAE, lower is better, identical `179,520` rows for every row of this table:

| Model | h=1 (15 min) | h=4 (1 h) | h=96 (24 h) |
|---|---|---|---|
| persistence | **0.4043** | 0.8897 | 2.4343 |
| Phase 4 classical GBM | 0.4242 | 0.8579 | **1.5648** |
| **Phase 5 TCN** | 0.4178 | **0.8307** | 1.6510 |
| Phase 5 vs GBM | **+1.5%** | **+3.2%** | **-5.5%** |

Recorded verdict: **mixed / partially improves** (D-079). Two of three horizons beat
gradient boosting; the longest does not. `h=96` is the open problem.

### 10.5 What the ablations measured

Every row trained at the same reduced budget (`2` epochs, training stride `32`), so
comparisons are valid **within** the table and not against the main run:

| Ablation | Question | Answer |
|---|---|---|
| `level_target` | change or level? | **change**, by a wide margin: level costs +253% / +52% / +20% |
| `no_cyclic` | do sin/cos channels help? | **no** - removing them is 3.6% / 28.7% / 18.6% *better* |
| `single_horizon` | does trunk sharing help? | **yes** - one head is 38.5% worse at h=1 |
| `lookback_96` | is one day enough? | **no** - costs +38.3% at h=1, +34.6% at h=96 |
| `lookback_192` | does the benefit saturate? | **not yet** - still +13.6% / +9.7% against a week |
| `transformer` | does attention beat convolution? | **not here** - +49.8% at h=1, -6.0% at h=4, +0.6% at h=96 |

Two consequences are carried forward rather than smoothed over. The cyclic channels
are **not earning their place** but stay in the shipped configuration, because dropping
them after seeing the test result would be tuning against the test split (D-077), and
because the ablation budget is too small to settle it. And the transformer result is
the evidence base for `docs/qwen_energy_requirements.md`.

### 10.6 Entry points

```text
energy-intel temporal config       resolved experiment configuration
energy-intel temporal run          train, evaluate, analyse, record
energy-intel temporal ablate       the ablation suite and its table
energy-intel temporal experiments  the recorded registry
```

## 11. Phase 6: Qwen3-1.7B-Base energy specialization

### 11.1 What it adds

Phase 6 asks whether a **pretrained** transformer representation transfers to energy
demand dynamics better than models trained only for this task. Phase 5's ablation had
already measured that a compact attention model lost to a 74k-parameter convolution on
this data, so the question is not "is attention good" but "does pretraining carry
information that survives the transfer".

```text
Phase 4 (flat lag features)        Phase 5 (ordered windows, trained from scratch)
  HistGBM, MLP, GRU                  dilated causal TCN, 74,099 params
        \                                  /
         \                                /
          ----  Phase 6  ----
           frozen Qwen3-1.7B trunk
           1,720,574,976 params, 0 trainable
           21 numeric tokens per window, 28 layers
           -> PCA(256, fitted on train only)
           -> head: features + per-series embedding -> 3 horizon deltas
```

### 11.2 The bridge, step by step

```text
raw kW per customer
  -> per-unit by that series' own rated kW         ml/sequence.py gather_sequences
  -> [demand, dy/dt, sin/cos hour, sin/cos doy]     6 channels, 168 sampled positions
  -> patches of 8 positions                         build_patches() -> 21 tokens
  -> LayerNorm + fixed linear projection            EnergyPatchProjector, 100,448 params
  -> inputs_embeds [B, 21, 2048]                   into Qwen's residual stream
  -> Qwen3Model trunk, frozen                       checkpoint.load_backbone(dtype=float32)
  -> last-token hidden state, layer 28              hidden_states[28][:, -1, :]
  -> PCA to 256 dims, fitted on TRAIN rows only     FeatureBasis.fit()
  -> concat per-series embedding (8 dims)           ProbeHead.series_embedding
  -> Linear -> 3 per-unit deltas                    ProbeHead.output
  -> yhat(t+h) = y(t) + delta(t+h)                  Phase 5's delta parameterisation
  -> x row's rated kW                              evaluate() in kW
```

The vocabulary embedding and the LM head are never used, so `inputs_embeds` replaces
`input_ids` entirely.

### 11.3 Compute reality, measured before anything was built

| Measurement | Value |
|---|---|
| Installed torch | `2.14.1+cpu`, `torch.cuda.is_available() == False` |
| GPU | RTX 4050 laptop, 6,141 MiB, ~2,400 MiB used by the desktop |
| Weights | 3.44 GB bf16 / 6.9 GB fp32 - no optimiser state fits in free VRAM |
| bf16 vs fp32 forward, 21 tokens | 2.568 vs 0.533 s/sample (**4.8x**) |
| fp32, 16 threads, batch 64 | **0.236 s/sample** |
| Real extraction, 25,000 rows | 0.248-0.278 s/sample |
| Full Phase 5 test population | ~11.8 h of forward passes per arm |

These numbers, not preference, chose a frozen backbone (D-084) and fp32 (D-085).

### 11.4 Modules

| Module | Responsibility |
|---|---|
| `ml/qwen/checkpoint.py` | Pinned identity, verification against the real files, architecture invariants (never used for surgery - see Phase 7) |
| `ml/qwen/representation.py` | The numeric bridge: patches, the fixed projector, per-layer feature extraction |
| `ml/qwen/features.py` | Deterministic per-series subsample, and the on-disk feature cache |
| `ml/qwen/probe.py` | `FeatureBasis` (train-only PCA), `ProbeHead`, `fit_probe`, `probe_predict` |
| `ml/qwen/baselines.py` | Persistence, refitted GBM, the real Phase 5 checkpoint, paired bootstrap |
| `ml/qwen/experiment.py` | The runner, regime/error analysis, representation diagnostics, the verdict |
| `config/qwen.py` | Strict config; refuses a redefined task |
| `scripts/phase6_extract.py` | Backbone extraction (the expensive step) |
| `scripts/phase6_probe.py` | Probe experiments over cached features (seconds) |

### 11.5 Results on identical rows

10,000 test rows, every model scored on the same rows. kW MAE:

| Model | h=1 | h=4 | h=96 |
|---|---|---|---|
| persistence | **0.3952** | 0.8925 | 2.4427 |
| Phase 5 TCN (real checkpoint) | 0.4081 | **0.8224** | 1.6522 |
| classical GBM (refitted) | 0.4946 | 0.9749 | **1.6168** |
| Qwen probe, 2-layer head | 0.5452 | 1.6060 | 2.7400 |

**Verdict: NO.** Worse at every horizon, worse than persistence at h=1, significant
under a paired bootstrap. Qwen wins no regime.

### 11.6 The two measurements that explain it

```text
pretrained vs random backbone, identical everything else:
  h=1    1.3324 vs 2.0154   pretrained better by 33.9%
  h=4    1.8264 vs 1.8375   tie
  h=96   2.9956 vs 3.8840   pretrained better by 22.9%

participation ratio of the frozen readout (effective dimensions out of 2048):
  layer        8     16    24    28
  pretrained  1.0    1.0   1.1   2.1
  random      7.1    8.4   9.6  10.0
```

Pretraining contributes real signal. It is simply not enough: the state the trunk hands
the head is nearly rank-2.

### 11.7 Entry points

```text
energy-intel qwen config       resolved experiment configuration
energy-intel qwen verify       checkpoint verification against the pinned facts
energy-intel qwen inspect      architecture report and the available surgery sites
energy-intel qwen probe        probe experiments over cached features
energy-intel qwen experiments  the recorded Phase 6 results
```

`verify` and `inspect` need only the checkpoint; `probe` needs the cached features
produced by `scripts/phase6_extract.py` and never downloads anything itself.

## 12. Phase 7: heterogeneous energy expert router

### 12.1 What it adds

Phase 7 asks whether a **learned routing mechanism** can pick or combine forecasting experts
according to the current demand state and beat the strongest single expert. The plan to do
this inside Qwen3-1.7B was dropped after Phase 6 measured that Qwen wins no regime and its
frozen readout has a participation ratio of 2.1 of 2,048 dimensions. See `decisions.md`
D-091 and `docs/expert_router_design.md`.

```text
Phase 4 (flat lag features)        Phase 5 (ordered windows)
  HistGBM, 950,400 training rows     dilated causal TCN, 74,099 params
        \                                  /
         \                                /
          ----  Phase 7  ----
   heterogeneous expert pool
     persistence          0 params
     classical_hist_gbm   36,600 tree nodes
     phase5_tcn           74,099 weights
        |
   router: 24 origin-observable features
     -> Linear(24, 32) + ReLU
     -> Linear(32, 3 experts x 3 horizons)
     -> softmax over experts, per horizon
     1,097 parameters
```

### 12.2 Two commands, because the costs differ by 30x

```text
energy-intel router experts   _script_path("phase7_experts.py") -> subprocess
                              build_panel()            experts.py
                              build_experts()          experts.py   ~5 min
                              pool.forecast_block()    experts.py
                              -> artifacts/phase7/experts.npz

energy-intel router run       _command_router_run()    cli.py
                              load_expert_cache()      cache.py
                              run_phase7()             offline/experiment.py
                              -> artifacts/phase7/<label>/result.json
                              -> experiments/registry.jsonl
```

`router experts` is separated because it is the expensive step - a HistGBM fit on 950,400
training rows plus a 74k-parameter temporal model over 359,040 rows - and because every
later question in the phase must read the *same* cached forecasts. Without that separation,
"the router beat the GBM" could mean "they were scored on different rows", and there would be
no way to tell from a table.

### 12.3 The panel, and why it is Phase 5's rows

```text
data/raw/smart_ds -> SmartDsAdapter -> load_target()        ml/targets.py
                  -> series.normalised()                    per unit of rated kW
   build_sequence_index(train_origin_stride=1)              ml/sequence.py
   950,400 train / 179,520 validation / 179,520 test
                  -> build_panel()                           experts.py
   concatenates validation then test, numbering them 0..N-1
                  -> cached with the forecasts              cache.py
```

`build_panel` is the single canonical ordering. Expert forecasts, router features, targets
and oracle labels all index those positions, so nothing downstream has to agree on how to
slice two different row arrays. The panel is ordered origin-major and series-minor, which is
what makes the lagged-error neighbour findable by binary search; `append_lagged_error_features`
raises rather than guessing if it is not.

### 12.4 Building the experts

```text
build_experts()                                   experts.py
  GBM, Phase 4's model reproduced exactly:
    build_feature_matrix(PHASE4_FEATURE_NAMES)   13 columns, Phase 4's order
    FeatureScaler.fit(train rows only)           ml/scaling.py
    + series_index, series_level, series_volatility
      fitted on training rows only               _phase4_series_context()
    fit_classical("classical_hist_gbm")          ml/baselines/classical.py
      one estimator per horizon, on PER-UNIT targets
  TCN, Phase 5's real checkpoint:
    torch.load(artifacts/phase5/tcn-main/checkpoint.pt)
    build_model("tcn", params=payload["config"])
    _window_scaler(values, index, index.rows(0)) ml/phase5.py
```

Two unit traps, both caught and both now enforced by an interface test. Phase 4 fits on
**per-unit** targets, so the GBM's estimate is per-unit and `GbmExpert` scales by rated kW.
Phase 5's `predict_series` returns **per-unit** levels, so `TcnExpert` scales by rated kW.
Every expert returns kW; `forecast_block` refuses a non-finite or mis-shaped forecast.

### 12.5 Router features, and the leakage guard

```text
build_router_features()                features.py
  level       y(t) per-unit and kW
  trajectory  trailing means over 4 and 24 steps, last ramp
  volatility  rolling std over 96 steps
  calendar    sin/cos hour-of-day and day-of-year, weekend
  scale       log rated kW
append_lagged_error_features()         features.py
  past_error  |f_e(t-h-1; h) - y(t-1)| / rated_kw   per expert per horizon
  past_seen   flag: was a predecessor reachable?
assert_router_features_are_causal()   features.py
  poison every value after a row's own origin in its series
  rebuild every row of that series up to that origin
  require bit-identical features
fill_missing(router_train_rows)        features.py
  NaN -> column mean over the router's own training rows only
```

The regime terciles are **not** inputs. They are computed from the demand being predicted,
so a router using them would be reading the target. They are used in
`offline/evaluation.py::router_by_regime` for the analysis and nowhere else.

### 12.6 Two training variants, chosen per horizon on validation

```text
fixed_ensemble, warm start          ensembles.py / offline/experiment.py
  fit_fixed_weights(router-train rows)   5,151-point simplex grid, chunked
fit_router(initial_weights=fixed)  training.py
  head bias = log(w), head weights = 0   so epoch 0 IS the fixed ensemble
  epoch 0 scored and eligible for selection
fit_router(initial_weights=None)   training.py     the cold variant
select per horizon on the selection rows           offline/experiment.py
```

### 12.7 The evaluation, in the order of the argument

```text
1  diversity        offline/diversity.py    correlations, agreement, per-regime splits
2  oracle           offline/oracle.py       per-row best expert with hindsight
3  router features  features.py             built, causality-checked, imputed
4  baselines        ensembles.py            uniform, best-single, fixed-weight
5  router           training.py             warm and cold variants
6  routing quality  offline/evaluation.py   regret, selection accuracy, capture ratio
                   offline/evaluation.py   stability, calibration, failure analysis
7  cost             offline/evaluation.py   parameters, seconds, per-row overhead
8  ablations        offline/ablations.py    four feature/head contracts
                   offline/crossfit.py      in-sample vs out-of-sample labels
9  parity           offline/experiment.py   Phase 7 experts vs Phase 4/5 published
10 verdict, trace, artifacts, registry
```

Steps 1 and 2 come **before** any router exists, because the honest question is whether the
premise holds. If the experts made the same mistakes, or the oracle barely beat the best
single expert, no router was going to help.

### 12.8 Modules

| Module | Responsibility |
|---|---|
| `ml/router/experts.py` | The panel, the three experts, the pool, Phase 4's feature contract |
| `ml/router/features.py` | The 24 origin-observable columns, the lagged-error lookup, the poisoning guard |
| `ml/router/model.py` | `EnergyRouterNet`, per-horizon simplex heads, the warm start |
| `ml/router/routing.py` | `mix`, `hard_selection`, `RoutingDecision`, the deployable `EnergyRouter` |
| `ml/router/training.py` | `fit_router`, `load_router`, epoch-0 eligibility, the checkpoint |
| `ml/router/ensembles.py` | Uniform, best-single and simplex-searched fixed weights |
| `ml/router/cache.py` | The single on-disk forecast block every number is read from |
| `ml/router/offline/oracle.py` | The upper bound. Not importable from the deployable side |
| `ml/router/offline/diversity.py` | Error correlations, agreement, per-regime expert preference |
| `ml/router/offline/evaluation.py` | Metrics, regret, stability, calibration, per-regime routing, cost |
| `ml/router/offline/crossfit.py` | How much in-sample expert accuracy would have distorted the labels |
| `ml/router/offline/ablations.py` | Four ablations and their stated questions |
| `ml/router/offline/experiment.py` | The runner, the verdict, the trace, the summary |
| `config/router.py` | Strict config; refuses a redefined task |
| `scripts/phase7_experts.py` | The expensive expert pass |

### 12.9 Entry points

```text
energy-intel router config        resolved experiment configuration
energy-intel router experts       refit the pool and cache forecasts (the slow step)
energy-intel router run           diversity, oracle, router, ablations, sealed evaluation
energy-intel router experiments   the recorded result
```

`experts` shells out to `scripts/phase7_experts.py`; `run` never downloads anything and never
reads the test split until its final evaluation.

### 12.10 Result

kW MAE on the sealed test split, 179,520 rows, every method scored on the same rows:

| method | h=1 | h=4 | h=96 |
|---|---|---|---|
| persistence | **0.4043** | 0.8897 | 2.4343 |
| classical GBM | 0.4242 | 0.8576 | 1.5737 |
| Phase 5 TCN | 0.4178 | 0.8307 | 1.6510 |
| uniform ensemble | 0.3955 | 0.7943 | 1.7615 |
| fixed ensemble | **0.3983** | **0.7917** | **1.5399** |
| router (soft) | 0.4069 | 0.7917 | 1.5600 |
| router (hard) | 0.4150 | 0.8307 | 1.5709 |
| oracle expert assignment | 0.2871 | 0.5217 | 0.9942 |

**Verdict: PARTIALLY, and the negative half is the important half.** Versus the best single
expert, 2 of 3 horizons - it *loses* at 15 minutes. Versus a fixed weighted ensemble,
**0 of 3**. The oracle shows 30-38% of achievable gain is present in the pool and the
experts disagree on 59-75% of rows, so the diversity premise holds; the router captured
-2% to 13% of it. Warm-started from the state-independent optimum it selected **epoch 0 at
every horizon**: gradient descent found no state-dependent weighting that beat a constant
one. It selected persistence 13 times in 179,520 rows and is 9.4x worse than it in the
low-ramp tercile.

## 13. Phase 8: calibrated predictive uncertainty

### 13.1 What it adds

Phase 7 left the point forecast alone and answered a negative question about routing. Phase 8
leaves the point forecast alone again and answers a different one: **how uncertain is it,
and does saying so identify the forecasts that will be wrong?**

The forecast is not refitted. Phase 7's fitted fixed-ensemble weights are read from its own
`result.json`, checked against `configs/uncertainty.toml`, and **the run stops if they
disagree** — because an uncertainty result around a different point forecast is not a Phase 8
result. The recovered ensemble reproduces Phase 7's published test MAE to **0.0000%**.

### 13.2 The partition

```text
TRAIN          Phase 4/5 split. 950,400 rows. The experts were fitted here.
                      |
VALIDATION     Phase 5/7 split. 179,520 rows. Expert predictions are out-of-sample.
    +-- CAL_FIT      first 50%, 89,760 rows. Scale functions, pinball models.
    +-- CAL_CONF     last  50%, 89,760 rows. Conformity scores, band cut points.
                      |
TEST           Phase 5/7 split. 179,520 rows. Read once, at the end.
```

Validation is halved rather than reusing Phase 7's own 70/30 router split, because the scale
model must not be fitted on rows the ensemble's weights were fitted on. The overlap that
remains is **measured** rather than assumed — `split_parity_check` reports the point
forecast's MAE on all three row sets, and on this data the two validation halves differ by
**+8.6% at h=1 and -11.0% at h=96**.

### 13.3 One idea, six methods

Every symmetric method is `point ± multiplier × scale(row)`. The families differ only in
where `multiplier` comes from; the **scale** decides whether the width may vary by row.

| Method | `multiplier` from | `scale` |
|---|---|---|
| `global_residual` | empirical quantile of \|residual\| on CAL_CONF | constant |
| `conformal_global` | `ceil((n+1)(1-α))`-th smallest \|residual\| | constant |
| `state_residual` | empirical quantile of \|residual\|/scale on CAL_CONF | learned |
| `conformal_state` | conformal quantile of \|residual\|/scale | learned |
| `conformal_dispersion` | conformal quantile of \|residual\|/spread | expert spread |
| `quantile_regression` | pinball-loss model of the residual quantiles | asymmetric |

`global_residual` is the baseline every other must beat, and the same method is applied to
each individual expert so the answer does not depend on which predictor is in use.

The learned scale fits `E[|residual| | x]` with an **L1** objective — the conditional
*median*, because the absolute residual's right tail would otherwise inflate every row's
width. Its inputs are Phase 7's 24 origin-observable columns, reused rather than rebuilt,
with the imputation reference changed from the whole panel to `CAL_FIT`. The origin-poisoning
guard runs on **every** Phase 8 experiment.

### 13.4 What is measured, per method

Four nominal levels (50/80/90/95%) × three horizons on the sealed split: coverage, signed
coverage error, a PASS/FAIL flag against `max(0.01, 3·SE)`, mean and median width, the
width/MAE sharpness ratio, the Winkler interval score and the weighted interval score,
Spearman correlation between width and realised error with the top-decile error multiple,
reliability by width bin, and the count of confident-and-severely-wrong rows.

The published procedure is chosen **per horizon** at the 90% band level from the measured
coverage — among calibrated methods, the lowest weighted interval score. On this data:
`conformal_state` at h=1 and h=4, `conformal_dispersion` at h=96.

### 13.5 Modules

| Module | Responsibility |
|---|---|
| `ml/uncertainty/intervals.py` | `PredictionInterval`, symmetric and asymmetric constructors, the zero clip |
| `ml/uncertainty/scales.py` | `GlobalScale`, `LearnedScale`, `DisagreementScale` |
| `ml/uncertainty/calibration.py` | `fit_conformal`, the finite-sample index, the conditional guarantee text |
| `ml/uncertainty/quantile.py` | `fit_quantile_model`, monotone rearrangement, crossing counts |
| `ml/uncertainty/split.py` | `CalibrationSplit`, the chronological halving, `split_parity_check` |
| `ml/uncertainty/features_set.py` | The scale's inputs, and the poisoning guard re-run per experiment |
| `ml/uncertainty/measures.py` | The five spread statistics and `spearman` |
| `ml/uncertainty/metrics.py` | Coverage, width, interval score, WIS, ECE, sharpness |
| `ml/uncertainty/evaluation.py` | Ranking, reliability, overconfidence, imminent-ramp detection |
| `ml/uncertainty/artifact.py` | `ProbabilisticForecast`, the batch, band cuts, save/load, the domain join |
| `ml/uncertainty/offline/analysis.py` | `evaluate_method`, `method_table`, `regime_breakdown` |
| `ml/uncertainty/offline/verdict.py` | The three components and the per-horizon selection |
| `ml/uncertainty/offline/experiment.py` | The runner, the selection, the trace, the summary |
| `config/uncertainty.py` | Strict config; refuses a redefined task or re-weighted forecast |

### 13.6 Entry points

```text
energy-intel uncertainty config        resolved experiment configuration
energy-intel uncertainty run          fit six interval methods, evaluate once on test
energy-intel uncertainty experiments  the recorded Phase 8 result
```

`run` reads Phase 7's cached forecasts and its recorded weights, downloads nothing, and never
reads the test split until its final evaluation.

### 13.7 Result

**Verdict: YES** on all three components, with the most useful finding being negative.

| horizon | published procedure | coverage @90% | error | ρ(width, error) | top decile |
|---|---|---|---|---|---|
| h=1 | `conformal_state` | 0.9020 | +0.0020 | +0.709 | 4.61x |
| h=4 | `conformal_state` | 0.8973 | -0.0027 | +0.673 | 4.21x |
| h=96 | `conformal_dispersion` | 0.8934 | -0.0066 | +0.658 | 4.80x |

**The default `±1σ` interval fails 11 of 12 coverage cells, in opposite directions at the two
ends of the range** — 0.9836 at h=1 and 0.9071 at h=96, against a nominal 0.95. The split-parity
table predicted both signs. A constant width calibrated on one half of validation does not
transfer to a later period on this data, and no care in choosing the quantile fixes it.

Expert disagreement is positive but weak (ρ = +0.41 / +0.25 / +0.18), decays with horizon,
and becomes **negative** at h=4 and h=96 once normalised by the forecast level. A learned
state-conditioned scale is the most informative uncertainty and the most miscalibrated at the
50% level. Pinball quantile regression degrades with horizon and is published nowhere.

Width tracks error in every regime at every horizon: across the load-level terciles the width
ratio runs 7.8x / 7.8x / 16.0x against error ratios of 7.5x / 7.5x / 11.2x.

**Not claimed:** no coverage guarantee (the exchangeability assumption is violated, and the
phase's own numbers show it); no aleatoric/epistemic decomposition; coverage is uniform on
average, not conditional on regime; no decision rule.

---

# Phase 9: Flexibility Estimation and Capability Audit

Phase 9 adds a flexibility representation and an honest audit of what the dataset can support.
It changes no forecasting behaviour and solves no optimisation problem. The production path is
untouched; what is new is a claim vocabulary that cannot be misused, and a measurement of
whether any physical flexibility exists here.

```text
                     energy-intel flexibility audit
                                (fits nothing, ~1s)
                                     │
                                     ▼
                  ┌──────────────────────────────────┐
                  │ capability audit                  │  ml/flexibility/capability.py
                  │ 8 SMART-DS dimensions              │
                  │ PHYSICAL / STATISTICAL_PROXY /     │
                  │ ASSUMED / UNKNOWN                  │
                  └────────────────┬─────────────────┘
                                   │  0 of 8 physical, 8 unknown
                                   ▼
                  ┌──────────────────────────────────┐
                  │ demand reconstruction             │  ml/flexibility/demand.py
                  │ values x rated kW                 │
                  │ verified against panel targets    │
                  └────────────────┬─────────────────┘
                                   │  0.0 kW gap / 359,040 rows
                                   ▼
     Phase 7 experts.npz ──►┌───────────────────────────┐   Phase 8 artifact ──┐
     (panel, rated kW,      │ CAL_FIT 89,760 rows       │   (published widths) │
      origins, forecasts)   │  fit baseline x granularity│◄──── row alignment ──┘
                            │  fit envelope per horizon  │        proved to 1e-9 kW
                            └─────────────┬─────────────┘
                                          ▼
                            ┌───────────────────────────┐
                            │ CAL_CONF 89,760 rows      │
                            │  select by interval score │
                            │  solve conformal scale    │
                            └─────────────┬─────────────┘
                                          ▼
                            ┌───────────────────────────┐
                            │ TEST 179,520 rows         │
                            │  read ONCE                │
                            │  coverage / sharpness /   │
                            │  direction / stability    │
                            │  regime breakdown         │
                            │  aggregation + naive sum  │
                            └─────────────┬─────────────┘
                                          ▼
                            ┌───────────────────────────┐
                            │ verdict: weakest component│  offline/verdict.py
                            │ every figure STATISTICAL_ │
                            │ PROXY / NOT_CONTROLLABLE   │
                            └───────────────────────────┘
```

### The claim vocabulary, enforced in the constructor

`domain/flexibility.py` is where a claim cannot be smuggled in. Three rules, all in
`__post_init__`:

| rule | what it stops |
|---|---|
| `STATISTICAL_PROXY` may not carry a controllable authority | "demand moved this much before" becoming "this much can be commanded" |
| `UNKNOWN` may not carry a magnitude | "not known" being published as "none available" |
| every estimate is `role=DERIVED` with provenance | a derived figure being written as a measurement |

`FlexibilityEnvelope` serialises `is_dispatchable` explicitly, so a deserialised envelope
cannot recover an authority it was never allowed to hold.

Direction semantics: `DOWNWARD` reduces net demand; `FLEXIBLE_LOAD_SHIFT` is a signed load
setpoint so downward gives `as_signed_setpoint() == -5.0` for a 5 kW shed, while
`NodePower`'s load-is-negative convention gives `as_node_power_delta() == +5.0`.

### Results

**Verdict: NO** - the weakest component, with no physical flexibility supported anywhere.

| horizon | coverage @90% | error | mean width kW | width/MAE | WIS kW | up cov | down cov |
|---|---|---|---|---|---|---|---|
| 1 | 0.9140 | +0.0140 | 2.7258 | 6.742 | 3.9853 | 0.9079 | 0.9048 |
| 4 | 0.9110 | +0.0110 | 5.5658 | 6.256 | 7.7263 | 0.9103 | 0.9101 |
| 96 | 0.8925 | -0.0075 | 14.1165 | 5.799 | 18.0102 | 0.8969 | 0.8819 |

Selected on the conformity split: `persistence` at the flat `horizon` granularity, with
conformal scales 0.9184 / 0.9063 / 0.9572 applied unchanged to test.

The band is roughly **six times wider than the mean absolute deviation it bounds**. That is what
90% two-sided coverage costs on near-symmetric heavy-tailed deviations, and it makes the envelope
a containment statement rather than a planning resolution.

Aggregation shows 99% diversification against a naive sum ~100x larger (pairwise correlation
-0.0000). That is a statement about independence, not headroom, and the pooled band covers
*worse* than the individual ones - the same fact that makes pooling defensible for containment
makes commanding all members at once impossible.

### The uncertainty coupling, measured and withheld

`reliance = clip(trailing_median(normalised_width) / normalised_width, 0.25, 1)`, strictly
trailing. It is **supported** on this data (Spearman +0.393 / +0.246 / +0.312) and still
**withheld**: the factor was judged post hoc on the sealed split, so applying it would reuse
test information to build the quantity under evaluation, and it is worth 12-31% of band width -
a change in meaning, not a refinement.

### Not claimed

No physical flexibility, no dispatchable resource, no guaranteed demand response, no
aggregation benefit, no uncertainty discount, no optimisation. Scenario assumptions exist for
demonstrating a control workflow but are barred from every real-data path by
`assert_real_data_mode`.


---

# Phase 10: Controlled Feature Ablation

Which predefined feature families actually contribute predictive information when the model
configuration is held fixed? The feature set is the independent variable; everything else is
identical by construction.

```text
                    scripts/run_feature_ablation.py --smoke | --all
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │ PROTOCOL FREEZE (before any measurement)     │  freeze.py
              │  feature sets + checksums, model per target, │
              │  F01-F06, metrics, seed, tie-break 0.5%,     │
              │  final-test access: five flags, all NO       │
              └──────────────────────┬───────────────────────┘
                                     │  hash recorded into every artifact
                                     ▼
              ┌──────────────────────────────────────────────┐
              │ dataset + feature matrix, built ONCE         │  experiment.py
              │  the full 13-feature matrix is computed and  │
              │  each set indexes columns, so a shared feature│
              │  cannot get a different value per set        │
              └──────────────────────┬───────────────────────┘
                                     ▼
              ┌──────────────────────────────────────────────┐
              │ ONE fit per feature set on the TRAIN split   │  runner.py
              │  scored on all six folds                     │  run_feature_set
              │  timestamp rows retained per fold            │
              └──────────────────────┬───────────────────────┘
                                     ▼
              ┌──────────────────────────────────────────────┐
              │ F01-F04 SELECTION, mean MAE                  │  selection.py
              │  absolute MAE first, then the frozen 0.5%    │
              │  simplicity tie-break over feature COUNT     │
              │  refuses F05/F06 as an argument              │
              └──────────────────────┬───────────────────────┘
                                     ▼
              ┌──────────────────────────────────────────────┐
              │ SELECTION FREEZE written                     │
              └──────────────────────┬───────────────────────┘
                                     ▼
              ┌──────────────────────────────────────────────┐
              │ F05-F06 CONFIRMATION                         │
              │  held out from selection, NOT unseen test    │
              └──────────────────────┬───────────────────────┘
                                     ▼
              ┌──────────────────────────────────────────────┐
              │ tables + reports, from recorded rows only    │  reports/phase10.py
              │  no table without data, no smoke rows        │
              └──────────────────────────────────────────────┘

  final test split ──► unreachable: assert_final_test_denied() raises FinalTestAccessError
                       and no CLI flag exists to request it
```

### The feature ladder

Every name is an existing `FeatureSpec`. No feature was invented; the four weather-derived
features are excluded from every set.

```text
A calendar                4   hour_of_day, day_of_week, day_of_year, is_weekend
B autoregressive lags     5   value_at_origin, lag_1, lag_4, lag_96, lag_672
C = A + B                 9
D = C + rolling          11   + roll_mean_96, roll_std_96
E = D + ramp             13   + ramp_1, roll_mean_same_hour_7d   (= all non-weather)
```

### Measured result

```text
customer_load  classical_hist_gbm   selected A (4 features)   tie-break applied
pv_generation  classical_ridge       selected B (5 features)   outright
```

For `customer_load` all five sets fall within 0.9% of each other and the 0.5% simplicity
tie-break selects the 4-feature calendar set over B, whose mean MAE was 0.28% better. For
`pv_generation` the lags beat calendar by 27%, which is far outside any tolerance, so B wins
outright and the full 13-feature set E is **worse** than B alone.

On confirmation, load's selected set was **not** the best on F05-F06. That is kept.

### The console

```text
energy-intel console health      system, config, data, models, integrity, phase
energy-intel console config      active config, config dir, feature sets
energy-intel console data        SMART-DS availability, per target, unknowns
energy-intel console status      every phase with its real verdict
energy-intel console flexibility PHYSICAL / STATISTICAL / ASSUMED, three lines, never merged
energy-intel ablation smoke      non-evidence pipeline check
energy-intel ablation run        the official grid, then selection, then confirmation
energy-intel ablation report     the recorded result, and it regenerates the tables
```

The console is read-only over artifacts earlier phases wrote. It never recomputes a result,
and it renders a missing measurement as UNKNOWN rather than as zero.


## 9. Entry points

```text
energy-intel ml config       resolved experiment configuration
energy-intel ml targets      every target with its support status and evidence
energy-intel ml features     the feature catalogue and the availability policy
energy-intel ml dataset      build and persist the dataset, no training
energy-intel ml run          dataset + baselines + analysis + registry
energy-intel ml experiments  the recorded registry
energy-intel router config       Phase 7 configuration
energy-intel router experts      the expensive expert pass
energy-intel router run          routing, ablations, sealed evaluation
energy-intel uncertainty config  Phase 8 configuration
energy-intel uncertainty run     six interval methods, one sealed evaluation
energy-intel flexibility audit      Phase 9 capability audit, fits nothing
energy-intel flexibility config     Phase 9 configuration
energy-intel flexibility run        capability audit, envelope, sealed evaluation
energy-intel flexibility experiments  the recorded summary
energy-intel console health     Phase 1-10 system health
energy-intel console status     every phase and its real verdict
energy-intel console data       SMART-DS availability and unsupported targets
energy-intel console config     active config and the Phase 10 feature sets
energy-intel console flexibility  physical / statistical / assumed, kept separate
energy-intel console integrity  final-test state and the protocol freeze
energy-intel ablation smoke     Phase 10 non-evidence pipeline check
energy-intel ablation run       Phase 10 official grid, selection, confirmation
energy-intel ablation report    the recorded Phase 10 result
```

All are invoked as `energy-intel <group> --<group>-config <path> <subcommand>`, because the
config selects the subcommand's experiment before the subcommand runs.
