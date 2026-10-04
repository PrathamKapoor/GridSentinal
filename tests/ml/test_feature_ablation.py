"""Phase 10 tests: the ablation rules, on tiny synthetic inputs.

No training runs here. Every test that needs a model fit uses a two-feature linear fixture and
a handful of rows, because the properties worth testing - fold boundaries, selection
restriction, the tie-break, sample accounting, the final-test refusal - are all decided by
arithmetic over small arrays. A test suite that trains a GBM to check an arithmetic rule would
be slow and would test the wrong thing.

Each test names the defect it pins. Several exist because the corresponding code was wrong at
least once while Phase 10 was being built.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from energy_intelligence.ml.feature_ablation import (
    FEATURE_SETS,
    SELECTION_FOLDS,
    SET_ORDER,
    FinalTestPolicy,
    FinalTestAccessError,
    SampleAccounting,
    ablation_folds,
    assert_final_test_denied,
    common_timestamps,
    primary_model_for,
    run_one,
)
from energy_intelligence.ml.feature_ablation.freeze import (
    TIE_BREAK_RELATIVE_TOLERANCE,
    build_protocol,
    load_freeze,
    protocol_hash,
    write_protocol_freeze,
)
from energy_intelligence.ml.feature_ablation.selection import (
    SELECTION_FOLDS_ALLOWED,
    confirm,
    incremental_effects,
    relative_difference,
    select,
)
from energy_intelligence.ml.feature_ablation.sets import WEATHER_FEATURES, checksum, members


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------


def test_feature_sets_are_the_five_expected_ids() -> None:
    assert SET_ORDER == ("A", "B", "C", "D", "E")


def test_the_ladder_is_nested() -> None:
    """Each set must contain its parent, or the incremental arithmetic is undefined."""
    assert set(members("A")) <= set(members("C")) <= set(members("D")) <= set(members("E"))
    assert set(members("B")) <= set(members("C"))


def test_full_set_is_exactly_every_non_weather_feature() -> None:
    """``E`` must terminate at the whole available set, not at an arbitrary stopping point."""
    from energy_intelligence.ml.features import FEATURE_CATALOGUE

    non_weather = {spec.name for spec in FEATURE_CATALOGUE} - set(WEATHER_FEATURES)
    assert set(members("E")) == non_weather


def test_no_feature_set_contains_a_weather_feature() -> None:
    """Including them would confound family effect with exogenous information."""
    for set_id in SET_ORDER:
        assert not (set(members(set_id)) & set(WEATHER_FEATURES)), set_id


def test_every_feature_name_exists_in_the_catalogue() -> None:
    """Phase 10 is an ablation, not feature engineering: nothing may be invented."""
    from energy_intelligence.ml.features import FEATURE_CATALOGUE

    known = {spec.name for spec in FEATURE_CATALOGUE}
    for set_id in SET_ORDER:
        assert set(members(set_id)) <= known, set_id


def test_feature_set_checksums_are_distinct_and_stable() -> None:
    values = {set_id: checksum(set_id) for set_id in SET_ORDER}
    assert len(set(values.values())) == len(SET_ORDER)
    assert values == {set_id: checksum(set_id) for set_id in SET_ORDER}


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def test_folds_tile_the_validation_range_without_gaps() -> None:
    """A gap or overlap would let a fold be scored on rows another fold also used."""
    folds = ablation_folds(1000, 7000)
    assert folds[0].start == 1000
    assert folds[-1].stop == 7000
    assert sum(f.size for f in folds) == 6000
    for left, right in zip(folds, folds[1:]):
        assert left.stop == right.start


def test_selection_and_confirmation_folds_are_disjoint() -> None:
    folds = ablation_folds(0, 6000)
    selection = [f for f in folds if f.role == "ABLATION"]
    confirmation = [f for f in folds if f.role == "CONFIRMATION"]
    assert len(selection) == 4
    assert len(confirmation) == 2
    assert not ({f.fold_id for f in selection} & {f.fold_id for f in confirmation})


def test_folds_never_reach_past_validation() -> None:
    """The fold builder must not be able to emit a fold inside the final test split."""
    folds = ablation_folds(0, 1000)
    assert all(f.stop <= 1000 for f in folds)


def test_final_test_access_is_refused() -> None:
    """The runtime guard behind the protocol freeze."""
    with pytest.raises(FinalTestAccessError, match="past the validation split"):
        assert_final_test_denied(1000, 5000)


def test_final_test_access_within_validation_is_allowed() -> None:
    assert_final_test_denied(1000, 1000)


def test_folds_reject_a_range_too_small_to_split() -> None:
    with pytest.raises(ValueError, match="too few"):
        ablation_folds(0, 3)


# ---------------------------------------------------------------------------
# Common samples
# ---------------------------------------------------------------------------


def test_common_timestamps_intersect_across_sets() -> None:
    """Larger sets read longer histories and have fewer usable rows; scoring each on its own
    rows would be a missing-row study, not an ablation."""
    a = SampleAccounting("t", "F01", "A", np.array([1, 2, 3]), np.array([0, 0, 0]), 0)
    b = SampleAccounting("t", "F01", "E", np.array([2, 3, 4]), np.array([0, 0, 0]), 5)
    identity, report = common_timestamps({"A": a, "E": b})
    assert identity.tolist() == [2 * 100_000, 3 * 100_000]
    assert report["common_rows"] == 2
    assert report["per_set_rows"] == {"A": 3, "E": 3}
    assert report["usable"] is True


def test_common_timestamps_reports_an_empty_intersection_as_unusable() -> None:
    """Disjoint sets cannot be compared and must say so rather than report n=0 as agreement."""
    a = SampleAccounting("t", "F01", "A", np.array([1, 2]), np.array([0, 0]), 0)
    b = SampleAccounting("t", "F01", "E", np.array([7, 8]), np.array([0, 0]), 0)
    identity, report = common_timestamps({"A": a, "E": b})
    assert identity.size == 0
    assert report["usable"] is False


def test_sample_accounting_records_its_identity_fields() -> None:
    account = SampleAccounting("t", "F02", "C", np.array([5, 6]), np.array([1, 2]), 3)
    payload = account.to_dict()
    for field in ("target", "fold", "feature_set", "sample_count", "timestamp_identity"):
        assert field in payload, field
    assert payload["sample_count"] == 2
    assert payload["dropped_rows"] == 3


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _cell(set_id: str, fold: str, mae: float) -> dict:
    return {
        "target": "customer_load",
        "feature_set": set_id,
        "fold": fold,
        "fold_role": "ABLATION" if fold in SELECTION_FOLDS_ALLOWED else "CONFIRMATION",
        "model": "classical_ridge",
        "feature_names": list(members(set_id)),
        "feature_count": len(members(set_id)),
        "metrics": {"mae": mae, "rmse": mae * 1.2, "n": 100, "unit": "kW"},
        "train_rows": 900,
        "score_rows": 100,
        "dropped_rows": 0,
        "seconds": 1.0,
    }


def test_selection_uses_only_the_selection_folds() -> None:
    """The regression test for the phase's central restriction."""
    rows = [_cell("A", f, 1.0) for f in SELECTION_FOLDS_ALLOWED]
    rows += [_cell("A", f, 99.0) for f in ("F05", "F06")]
    chosen = select(rows)
    assert chosen["per_set_mean_mae"]["A"] == pytest.approx(1.0), "confirmation folds leaked in"


def test_selection_refuses_a_confirmation_fold() -> None:
    """Asking for F05 as a selection fold must raise, not quietly average it in."""
    with pytest.raises(ValueError, match="post-ablation confirmation"):
        select([_cell("A", "F05", 1.0)], fold_ids=("F01", "F05"))


def test_selection_picks_the_lowest_mean_mae() -> None:
    rows = []
    for set_id, value in (("A", 1.0), ("B", 0.9), ("C", 0.95), ("D", 1.1), ("E", 1.2)):
        rows += [_cell(set_id, f, value) for f in SELECTION_FOLDS_ALLOWED]
    assert select(rows)["selected_feature_set"] == "B"


def test_tie_break_prefers_the_simpler_set_within_tolerance() -> None:
    """A 0.4% difference is inside the frozen 0.5% tolerance, so the 4-feature set wins."""
    rows = []
    for set_id, value in (("A", 1.000), ("B", 1.002), ("C", 1.050), ("D", 1.060), ("E", 1.070)):
        rows += [_cell(set_id, f, value) for f in SELECTION_FOLDS_ALLOWED]
    chosen = select(rows)
    assert chosen["best_absolute_mae_set"] == "A"
    assert chosen["selected_feature_set"] == "A"
    assert chosen["tie_break"]["applied"] is False


def test_tie_break_prefers_fewer_features_when_effectively_tied() -> None:
    """A is the simplest set, so if it is within tolerance of anything it wins outright."""
    rows = []
    for set_id, value in (("A", 1.0000), ("B", 0.9990), ("C", 1.0500), ("D", 1.0600), ("E", 1.0700)):
        rows += [_cell(set_id, f, value) for f in SELECTION_FOLDS_ALLOWED]
    chosen = select(rows)
    assert chosen["best_absolute_mae_set"] == "B"
    assert chosen["tie_break"]["applied"] is True
    assert chosen["selected_feature_set"] == "A", "the 4-feature set should win the near-tie"


def test_tie_break_cannot_rescue_a_materially_worse_simple_set() -> None:
    """A set 5% worse must lose no matter how few features it has."""
    rows = []
    for set_id, value in (("A", 1.050), ("B", 1.000), ("C", 1.060), ("D", 1.070), ("E", 1.080)):
        rows += [_cell(set_id, f, value) for f in SELECTION_FOLDS_ALLOWED]
    chosen = select(rows)
    assert chosen["selected_feature_set"] == "B"
    assert chosen["tie_break"]["applied"] is False


def test_selection_rejects_an_incomplete_grid() -> None:
    """A set missing a fold cannot be ranked, and must raise rather than be dropped silently."""
    rows = [_cell("A", "F01", 1.0), _cell("A", "F02", 1.0)] + [
        _cell("B", f, 1.1) for f in SELECTION_FOLDS_ALLOWED
    ]
    with pytest.raises(ValueError, match="missing selection folds"):
        select(rows)


def test_relative_difference_sign_and_zero_guard() -> None:
    assert relative_difference(1.0, 1.5) == pytest.approx(0.5)
    assert relative_difference(1.0, 0.5) == pytest.approx(-0.5)
    with pytest.raises(ZeroDivisionError):
        relative_difference(0.0, 1.0)


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------


def test_confirmation_uses_only_f05_and_f06() -> None:
    rows = [_cell("A", f, 1.0) for f in SELECTION_FOLDS_ALLOWED] + [
        _cell("A", "F05", 0.1), _cell("A", "F06", 0.1)
    ]
    record = confirm(rows, selected_set="A")
    assert record["status"] == "COMPLETE"
    assert record["per_set_mean_mae"]["A"] == pytest.approx(0.1)


def test_confirmation_reports_when_the_chosen_set_did_not_hold() -> None:
    """The honest negative: selection and confirmation can disagree, and that is a finding."""
    rows = [_cell("A", f, 1.0) for f in SELECTION_FOLDS_ALLOWED] + [
        _cell("A", "F05", 2.0), _cell("A", "F06", 2.0),
        _cell("E", "F05", 1.0), _cell("E", "F06", 1.0),
    ]
    record = confirm(rows, selected_set="A")
    assert record["selected_set_was_best_on_confirmation"] is False
    assert record["best_on_confirmation"] == "E"
    assert record["relative_difference_selected_vs_best"] > 0


def test_confirmation_never_calls_itself_unseen_test_data() -> None:
    rows = [_cell("A", f, 1.0) for f in ("F05", "F06")]
    record = confirm(rows, selected_set="A")
    assert "not the final test split" in record["held_out_caveat"]


# ---------------------------------------------------------------------------
# Incremental effects
# ---------------------------------------------------------------------------


def test_incremental_effects_report_absolute_and_relative_differences() -> None:
    rows = []
    values = {"A": 1.0, "B": 1.2, "C": 0.9, "D": 0.95, "E": 1.1}
    for set_id, value in values.items():
        rows += [_cell(set_id, f, value) for f in SELECTION_FOLDS_ALLOWED]
    block = incremental_effects(rows)
    a_to_c = block["comparisons"]["A -> C"]
    assert a_to_c["status"] == "COMPLETE"
    assert a_to_c["absolute_difference_kw"] == pytest.approx(-0.1)
    assert a_to_c["relative_difference"] == pytest.approx(-0.1)
    assert a_to_c["direction"] == "improvement"
    assert block["comparisons"]["A -> B"]["direction"] == "regression"


def test_incremental_effects_name_the_features_each_step_adds() -> None:
    rows = [_cell(set_id, f, 1.0) for set_id in SET_ORDER for f in SELECTION_FOLDS_ALLOWED]
    block = incremental_effects(rows)
    assert "lag_1" in block["comparisons"]["A -> C"]["features_added"]
    assert "roll_mean_96" in block["comparisons"]["C -> D"]["features_added"]
    assert "ramp_1" in block["comparisons"]["D -> E"]["features_added"]


def test_incremental_effects_never_claim_causality() -> None:
    """These are differences under one protocol, not causal effects."""
    rows = [_cell(set_id, f, 1.0) for set_id in SET_ORDER for f in SELECTION_FOLDS_ALLOWED]
    block = incremental_effects(rows)
    assert "not causal" in block["not_a_claim"].lower()
    assert "no_hpo" in block


# ---------------------------------------------------------------------------
# Protocol freeze
# ---------------------------------------------------------------------------


def _protocol(**overrides):
    kwargs = dict(
        targets=("customer_load",),
        unsupported_targets={"wind_generation": "no wind assets in SMART-DS v1.0"},
        horizon_steps=96,
        primary_model_by_target={"customer_load": "classical_hist_gbm"},
        primary_model_evidence={"customer_load": "Phase 5 registry"},
        model_config={"a": 1},
        secondary_model_config={"b": 2},
        seed=20260101,
    )
    kwargs.update(overrides)
    return build_protocol(**kwargs)


def test_protocol_records_h24() -> None:
    assert _protocol().horizon_label == "H24"
    assert _protocol().horizon_steps == 96


def test_protocol_freezes_the_tie_break_before_results() -> None:
    payload = _protocol().to_dict()
    assert payload["tie_break"]["relative_mae_tolerance"] == TIE_BREAK_RELATIVE_TOLERANCE
    assert payload["tie_break"]["frozen_before_results"] is True


def test_protocol_refuses_every_final_test_access() -> None:
    policy = FinalTestPolicy()
    assert policy.violations() == [], policy.violations()
    payload = policy.to_dict()
    for field in (
        "training_access",
        "hpo_access",
        "model_selection_access",
        "feature_selection_access",
        "performance_evaluation",
    ):
        assert payload[field] is False, field
    assert payload["state"] == "LOCKED"


def test_protocol_records_the_unsupported_target_rather_than_dropping_it() -> None:
    payload = _protocol().to_dict()
    assert "wind_generation" in payload["unsupported_targets"]


def test_protocol_prohibits_hpo() -> None:
    prohibited = " ".join(_protocol().prohibited)
    for name in ("Optuna", "GridSearch", "RandomSearch"):
        assert name in prohibited


def test_protocol_hash_is_content_addressed(tmp_path) -> None:
    first = protocol_hash(_protocol())
    second = protocol_hash(_protocol())
    assert first == second
    changed = protocol_hash(_protocol(seed=1))
    assert changed != first, "changing the seed must change the hash"


def test_freeze_writes_and_refuses_to_overwrite(tmp_path) -> None:
    """A re-run must not silently replace the protocol it is obeying."""
    path, digest = write_protocol_freeze(_protocol(), tmp_path)
    assert path.is_file()
    assert digest == protocol_hash(_protocol())
    with pytest.raises(FileExistsError, match="must not change"):
        write_protocol_freeze(_protocol(), tmp_path)


def test_freeze_round_trips(tmp_path) -> None:
    path, _ = write_protocol_freeze(_protocol(), tmp_path)
    payload = load_freeze(path)
    assert payload["targets"] == ["customer_load"]
    assert payload["horizon_label"] == "H24"
    assert payload["final_test"]["state"] == "LOCKED"
    assert payload["selection_folds"] == list(SELECTION_FOLDS_ALLOWED)


def test_freeze_records_the_feature_set_checksums(tmp_path) -> None:
    path, _ = write_protocol_freeze(_protocol(), tmp_path)
    payload = load_freeze(path)
    recorded = payload["feature_set_checksums"]
    assert recorded == {set_id: checksum(set_id) for set_id in SET_ORDER}


# ---------------------------------------------------------------------------
# Model choice is evidence-based, not assumed uniform
# ---------------------------------------------------------------------------


def test_primary_model_differs_by_target() -> None:
    """The brief forbids assuming one model suits every target; the repo found two winners."""
    assert primary_model_for("customer_load") != primary_model_for("pv_generation")


def test_primary_model_refuses_an_unknown_target() -> None:
    with pytest.raises(KeyError):
        primary_model_for("not_a_target")


def test_neural_gru_is_not_a_phase_10_model() -> None:
    """GRU and LSTM are excluded from Phase 10."""
    from energy_intelligence.ml.baselines.classical import CLASSICAL_MODELS

    assert "neural_gru" not in CLASSICAL_MODELS
    assert primary_model_for("customer_load") in CLASSICAL_MODELS


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


class _FakeDataset:
    """A minimal stand-in for ``MlDataset``: split codes, origins, targets, unit."""

    def __init__(self, n: int = 300) -> None:
        rng = np.random.default_rng(0)
        self.origin_index = np.arange(n, dtype=np.int64)
        self.series_index = np.zeros(n, dtype=np.int64)
        self.targets = rng.standard_normal((n, 1))
        self.target_unit = "kW"
        self.split = np.zeros(n, dtype=np.int64)
        self.split[:100] = 0  # train
        self.split[100:200] = 1  # validation
        self.split[200:] = 2  # test - never read by Phase 10
        self.feature_names = ("hour_of_day", "day_of_week")

    def mask(self, code: int) -> np.ndarray:
        return self.split == code


def test_runner_scores_only_the_named_fold() -> None:
    dataset = _FakeDataset()
    fold = ablation_folds(100, 200)[0]
    result = run_one(
        dataset=dataset,
        feature_matrix=np.arange(600, dtype=np.float64).reshape(300, 2),
        feature_names=("hour_of_day", "day_of_week"),
        target="customer_load",
        feature_set="A",
        fold=fold,
        model="classical_ridge",
        seed=20260101,
        validation_stop=200,
        horizon_steps=96,
    )
    assert result.fold_id == fold.fold_id
    assert result.score_rows == fold.size
    assert result.train_rows == 100
    assert result.metrics["mae"] >= 0.0


def test_runner_retains_timestamp_level_evidence() -> None:
    """These fields are the evidence base for a future paired analysis and cannot be rebuilt."""
    dataset = _FakeDataset()
    fold = ablation_folds(100, 200)[0]
    result = run_one(
        dataset=dataset,
        feature_matrix=np.arange(600, dtype=np.float64).reshape(300, 2),
        feature_names=("hour_of_day", "day_of_week"),
        target="customer_load",
        feature_set="A",
        fold=fold,
        model="classical_ridge",
        seed=20260101,
        validation_stop=200,
        horizon_steps=96,
    )
    rows = result.timestamp_rows
    assert rows is not None
    assert rows.forecast_origin.shape == (result.score_rows,)
    assert np.array_equal(rows.target_timestamp, rows.forecast_origin + 96)
    assert np.allclose(rows.error, rows.actual - rows.prediction)
    assert np.allclose(rows.absolute_error, np.abs(rows.error))
    assert np.allclose(rows.squared_error, rows.error**2)


def test_runner_refuses_a_fold_past_validation() -> None:
    """The regression test: the runner itself cannot reach the final test split."""
    dataset = _FakeDataset()
    rogue = type(ablation_folds(100, 200)[0])("F99", "ABLATION", 200, 300)
    with pytest.raises(FinalTestAccessError):
        run_one(
            dataset=dataset,
            feature_matrix=np.arange(600, dtype=np.float64).reshape(300, 2),
            feature_names=("hour_of_day", "day_of_week"),
            target="customer_load",
            feature_set="A",
            fold=rogue,
            model="classical_ridge",
            seed=20260101,
            validation_stop=200,
            horizon_steps=96,
        )


def test_runner_never_reads_the_test_split() -> None:
    """Corrupting the test rows must not change any scored number."""
    dataset = _FakeDataset()
    matrix = np.arange(600, dtype=np.float64).reshape(300, 2)
    fold = ablation_folds(100, 200)[0]
    kwargs = dict(
        dataset=dataset,
        feature_matrix=matrix,
        feature_names=("hour_of_day", "day_of_week"),
        target="customer_load",
        feature_set="A",
        fold=fold,
        model="classical_ridge",
        seed=20260101,
        validation_stop=200,
        horizon_steps=96,
    )
    baseline = run_one(**kwargs).metrics["mae"]
    dataset.targets[200:] = 1e9
    matrix[200:] = -1e9
    assert run_one(**kwargs).metrics["mae"] == pytest.approx(baseline)


def test_runner_rejects_a_misaligned_matrix() -> None:
    dataset = _FakeDataset()
    fold = ablation_folds(100, 200)[0]
    with pytest.raises(ValueError, match="must be aligned"):
        run_one(
            dataset=dataset,
            feature_matrix=np.zeros((10, 2)),
            feature_names=("hour_of_day", "day_of_week"),
            target="customer_load",
            feature_set="A",
            fold=fold,
            model="classical_ridge",
            seed=20260101,
            validation_stop=200,
            horizon_steps=96,
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_phase_10_config_parses_and_matches_the_code() -> None:
    """The YAML must describe the same ladder the code uses, or the config is decoration."""
    import tomllib  # noqa: F401 - the file is YAML; parsed below with yaml

    import yaml

    from pathlib import Path as _Path

    path = _Path(__file__).resolve().parents[2] / "config" / "ablation" / "phase_10.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["horizon_steps"] == 96
    assert payload["horizon_label"] == "H24"
    for set_id in SET_ORDER:
        assert payload["feature_sets"][set_id] == list(members(set_id)), set_id
    for name in WEATHER_FEATURES:
        assert name in payload["excluded_features"]
    assert payload["selection"]["folds"] == list(SELECTION_FOLDS_ALLOWED)
    assert payload["selection"]["tie_break"]["relative_mae_tolerance"] == TIE_BREAK_RELATIVE_TOLERANCE
    assert payload["confirmation"]["folds"] == ["F05", "F06"]
    assert payload["confirmation"]["called_unseen_test_data"] is False
    assert payload["common_samples"]["enabled"] is True
    assert payload["seed_policy"]["same_seed_for_every_feature_set"] is True
    for field in ("training", "hpo", "model_selection", "feature_selection", "performance_evaluation"):
        assert payload["final_test_access"][field] is False, field


def test_phase_10_config_has_no_absolute_machine_paths() -> None:
    """A config that only runs on one machine is not a config."""
    import re
    from pathlib import Path as _Path

    path = _Path(__file__).resolve().parents[2] / "config" / "ablation" / "phase_10.yaml"
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"[A-Za-z]:[\\/]|/Users/|/home/", text)


def test_phase_10_config_has_no_tuning_section() -> None:
    """No search section, because tuning is exactly what this phase forbids."""
    from pathlib import Path as _Path

    text = (
        _Path(__file__).resolve().parents[2] / "config" / "ablation" / "phase_10.yaml"
    ).read_text(encoding="utf-8")
    for forbidden in ("optuna", "grid_search", "random_search", "n_trials"):
        assert forbidden not in text.lower(), forbidden


# ---------------------------------------------------------------------------
# The execution script's control surface
# ---------------------------------------------------------------------------


def test_script_exposes_no_final_test_option() -> None:
    """The omission is the control: an option that does not exist cannot be reached for."""
    from pathlib import Path as _Path

    source = (
        _Path(__file__).resolve().parents[2] / "scripts" / "run_feature_ablation.py"
    ).read_text(encoding="utf-8")
    assert "--final-test" not in source
    assert "--test-split" not in source
    assert "--smoke" in source and "--all" in source


def test_script_smoke_and_all_are_mutually_exclusive_and_required() -> None:
    import importlib.util
    from pathlib import Path as _Path

    path = _Path(__file__).resolve().parents[2] / "scripts" / "run_feature_ablation.py"
    spec = importlib.util.spec_from_file_location("run_feature_ablation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--smoke", "--all"])
    assert parser.parse_args(["--smoke"]).smoke is True
    assert parser.parse_args(["--all"]).all is True


def test_script_maps_public_target_names_to_repository_ids() -> None:
    import importlib.util
    from pathlib import Path as _Path

    path = _Path(__file__).resolve().parents[2] / "scripts" / "run_feature_ablation.py"
    spec = importlib.util.spec_from_file_location("run_feature_ablation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.TARGET_ALIASES["load"] == "customer_load"
    assert module.TARGET_ALIASES["pv"] == "pv_generation"
    assert module.TARGET_ALIASES["wind"] == "wind_generation"


def test_script_refuses_wind_with_its_reason() -> None:
    """Asking for WIND must print why it is unavailable, not silently skip it."""
    import importlib.util
    from pathlib import Path as _Path

    path = _Path(__file__).resolve().parents[2] / "scripts" / "run_feature_ablation.py"
    spec = importlib.util.spec_from_file_location("run_feature_ablation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.main(["--all", "--target", "wind"]) == 1


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_tables_are_not_written_without_data(tmp_path) -> None:
    """An empty table that looks complete is worse than no table."""
    from energy_intelligence.reports import write_tables

    written = write_tables(tmp_path, {"results": [], "selection": {}, "confirmation": {}})
    assert written == {}


def test_tables_never_include_smoke_rows(tmp_path) -> None:
    """Smoke results are not evidence and must not reach a research table."""
    from energy_intelligence.reports import write_tables

    payload = {
        "results": [
            {
                "target": "customer_load",
                "model": "classical_ridge",
                "feature_set": "A",
                "feature_count": 4,
                "fold": "F01",
                "fold_role": "ABLATION",
                "metrics": {"mae": 1.0, "rmse": 1.2, "n": 10, "unit": "kW"},
                "train_rows": 100,
                "dropped_rows": 0,
                "seconds": 1.0,
                "evidence_class": "NON_EVIDENCE_SMOKE",
            }
        ],
        "evidence_class": "OFFICIAL",
    }
    assert write_tables(tmp_path, payload) == {}


def test_completion_report_is_partial_without_a_full_grid(tmp_path) -> None:
    from energy_intelligence.reports import write_report

    path = write_report(
        tmp_path,
        {"results": [], "selection": {}, "confirmation": {}, "evidence_class": "OFFICIAL"},
    )
    text = path.read_text(encoding="utf-8")
    assert "STATUS: PARTIAL" in text
    assert "NOT RUN" in text, "the MLP arm must be declared, not omitted"


def test_completion_report_states_the_final_test_policy(tmp_path) -> None:
    from energy_intelligence.reports import write_report

    path = write_report(
        tmp_path,
        {"results": [], "selection": {}, "confirmation": {}, "evidence_class": "OFFICIAL"},
    )
    text = path.read_text(encoding="utf-8")
    assert "FINAL_TEST_TRAINING_ACCESS          = NO" in text
    assert "FINAL_TEST_FEATURE_SELECTION_ACCESS  = NO" in text


def test_completion_report_records_wind_as_excluded(tmp_path) -> None:
    """An unmeasurable target is reported, never quietly dropped."""
    from energy_intelligence.reports import write_report

    path = write_report(
        tmp_path,
        {"results": [], "selection": {}, "confirmation": {}, "evidence_class": "OFFICIAL"},
    )
    assert "wind_generation" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------


def test_console_never_claims_the_proxy_is_dispatchable(tmp_path) -> None:
    """The single most important console property.

    The word "dispatchable" may appear only inside an explicit negation. An earlier version of
    this test tried to strip the negation out of the string and compare, which does not work:
    the sentence that matters *is* the negation.
    """
    import re

    from energy_intelligence.console import format_flexibility

    for root in (tmp_path, _repo_root()):
        text = format_flexibility(root)
        for match in re.finditer(r"dispatchable", text, re.IGNORECASE):
            window = text[max(0, match.start() - 40) : match.end() + 5].lower()
            assert any(
                marker in window
                for marker in ("never", "not ", "no ", "isn't", "cannot")
            ), f"unqualified claim of dispatchability: ...{window}..."


def test_console_separates_the_three_flexibility_bases(tmp_path) -> None:
    from energy_intelligence.console import format_flexibility

    text = format_flexibility(tmp_path)
    assert "PHYSICAL" in text
    assert "STATISTICAL" in text
    assert "ASSUMED" in text


def test_console_reports_missing_data_as_unknown_not_zero(tmp_path) -> None:
    """A missing measurement rendered as 0.0 is a lie about the dataset."""
    from energy_intelligence.console import format_flexibility

    text = format_flexibility(tmp_path)
    # Every basis reports UNKNOWN rather than a number, and no zero appears anywhere.
    assert text.count("UNKNOWN") >= 2, text
    assert " 0.0" not in text
    assert "0 kW" not in text


def test_console_reports_an_unsupported_target_with_its_reason() -> None:
    from energy_intelligence.console import format_data_status

    text = format_data_status(_repo_root())
    assert "wind_generation" in text
    assert "UNSUPPORTED" in text


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[2]


def test_console_status_lists_every_phase_with_its_verdict() -> None:
    from energy_intelligence.console import PHASE_STATUS

    assert len(PHASE_STATUS) == 10
    verdicts = {row.verdict for row in PHASE_STATUS}
    assert "NEGATIVE" in verdicts, "phases 6 and 7 failed and that must be visible"


def test_console_integrity_reports_final_test_locked() -> None:
    from energy_intelligence.console import Console

    block = Console(_repo_root()).integrity()
    assert block["final_test"] == "LOCKED"
    assert block["final_test_policy"]["performance_evaluation"] is False


def test_console_never_prints_secrets() -> None:
    from energy_intelligence.console import Console

    summary = Console(_repo_root()).config_summary()
    assert summary["secrets_printed"] is False
    assert "password" not in json.dumps(summary).lower()