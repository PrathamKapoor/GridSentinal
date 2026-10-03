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

## 9. Entry points

```text
energy-intel ml config       resolved experiment configuration
energy-intel ml targets      every target with its support status and evidence
energy-intel ml features     the feature catalogue and the availability policy
energy-intel ml dataset      build and persist the dataset, no training
energy-intel ml run          dataset + baselines + analysis + registry
energy-intel ml experiments  the recorded registry
```

All are invoked as `energy-intel ml --ml-config <path> <subcommand>`, because the
config selects the subcommand's experiment before the subcommand runs.
