"""Phase 7, part 1: the expert interface and the panel it is defined over.

The interface contract is where a heterogeneous pool quietly breaks. Three models written
by three phases, refitted at different times, storing per-unit values in different places.
These tests pin the four properties everything downstream assumes:

* every expert returns ``[rows, horizons]`` in **kW**, never per-unit;
* every expert agrees on which horizon each column is;
* the pool refuses an expert that disagrees, rather than silently broadcasting;
* the panel is ordered so the lagged-error lookup is a search rather than a guess.
"""

from __future__ import annotations

import numpy as np
import pytest

from energy_intelligence.ml.router.cache import ExpertCache, load_expert_cache, save_expert_cache
from energy_intelligence.ml.router.experts import (
    PHASE4_FEATURE_NAMES,
    PHASE7_EXPERTS,
    EvaluationPanel,
    ExpertForecast,
    ExpertPool,
    ForecastExpert,
    GbmExpert,
    PersistenceExpert,
    _phase4_series_context,
    _tree_size,
)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


def test_panel_positions_are_numbered_in_origin_major_series_minor_order(
    panel: EvaluationPanel,
) -> None:
    """The order the lagged-error lookup's ``searchsorted`` depends on.

    The invariant is the composite key ``origin * n_series + series`` being strictly
    increasing, which is what makes the lagged-error neighbour findable by binary search.
    Checking origins and series separately would not catch an interleaving that keeps both
    sorted locally.
    """
    series_count = int(panel.series.max()) + 1
    keys = panel.origins * series_count + panel.series
    assert np.all(np.diff(keys) > 0)
    assert np.all(np.diff(panel.origins) >= 0)


def test_panel_splits_partition_every_row_exactly_once(panel: EvaluationPanel) -> None:
    covered = np.concatenate(
        [np.arange(s.start, s.stop) for s in panel.split_slices.values()]
    )
    assert np.array_equal(np.sort(covered), np.arange(panel.n_rows))
    assert panel.split_mask("test").sum() == (
        panel.split_slices["test"].stop - panel.split_slices["test"].start
    )


def test_panel_describe_reports_both_splits(panel: EvaluationPanel) -> None:
    described = panel.describe()
    assert described["n_rows"] == panel.n_rows
    assert set(described["splits"]) == {"validation", "test"}
    assert described["horizons"] == [1, 4, 96]


# ---------------------------------------------------------------------------
# Expert interface
# ---------------------------------------------------------------------------


def test_persistence_expert_returns_the_persistence_baseline(panel: EvaluationPanel) -> None:
    forecast = PersistenceExpert(panel).predict(np.arange(10))
    assert np.array_equal(forecast.forecast_kw, panel.persistence_kw[:10])
    assert forecast.horizons == panel.horizons
    assert forecast.provenance["fitted"] is False


def test_persistence_is_phase4s_definition_not_a_lookalike(panel: EvaluationPanel) -> None:
    """``yhat(t+h) = y(t)`` means one column repeated across every horizon."""
    forecast = PersistenceExpert(panel).predict(np.arange(5, 9)).forecast_kw
    assert np.allclose(forecast[:, 0], forecast[:, 2])
    assert np.allclose(forecast, panel.persistence_kw[5:9])


def test_every_expert_returns_rows_by_horizons_in_kw(
    stub_pool: ExpertPool, panel: EvaluationPanel
) -> None:
    positions = np.arange(0, panel.n_rows, 7)
    block, timings = stub_pool.forecast_block(positions)
    assert block.shape == (stub_pool.n_experts, positions.size, 3)
    assert set(timings) == set(stub_pool.names)
    # kW, not per-unit: the stubs are built in the tens, per-unit would be under one.
    assert float(block.min()) > 1.0


def test_expert_forecast_rejects_a_wrong_column_count() -> None:
    with pytest.raises(ValueError, match="columns for"):
        ExpertForecast(name="bad", forecast_kw=np.zeros((3, 2)), horizons=(1, 4, 96))


def test_expert_forecast_rejects_a_non_matrix() -> None:
    with pytest.raises(ValueError, match=r"\[rows, horizons\]"):
        ExpertForecast(name="bad", forecast_kw=np.zeros(3), horizons=(1,))


def test_pool_rejects_duplicate_expert_names(panel: EvaluationPanel) -> None:
    with pytest.raises(ValueError, match="duplicate expert names"):
        ExpertPool([PersistenceExpert(panel), PersistenceExpert(panel)], panel.horizons)


def test_pool_rejects_an_empty_expert_list() -> None:
    with pytest.raises(ValueError, match="at least one expert"):
        ExpertPool([], (1, 4, 96))


def test_pool_rejects_an_expert_built_for_other_horizons(
    panel: EvaluationPanel, constant_expert
) -> None:
    """A horizon mismatch must fail loudly, not broadcast into the wrong column."""

    class WrongHorizons:
        name = "wrong_horizons"

        def predict(self, positions):
            return ExpertForecast(
                name=self.name,
                forecast_kw=np.zeros((np.asarray(positions).size, 2)),
                horizons=(1, 4),
            )

    pool = ExpertPool(
        [PersistenceExpert(panel), WrongHorizons(), constant_expert("third", 0.0)],
        panel.horizons,
    )
    with pytest.raises(ValueError, match="was built for horizons"):
        pool.forecast_block(np.arange(4))


def test_pool_rejects_an_expert_returning_the_wrong_row_count(
    panel: EvaluationPanel, constant_expert
) -> None:
    class TooFewRows:
        name = "too_few"

        def predict(self, positions):
            return ExpertForecast(
                name=self.name,
                forecast_kw=np.zeros((3, len(panel.horizons))),
                horizons=panel.horizons,
            )

    pool = ExpertPool(
        [PersistenceExpert(panel), TooFewRows(), constant_expert("third", 0.0)],
        panel.horizons,
    )
    with pytest.raises(ValueError, match="returned 3 rows for 7 requested"):
        pool.forecast_block(np.arange(7))


def test_pool_rejects_non_finite_expert_output(
    panel: EvaluationPanel, constant_expert
) -> None:
    class NotFinite:
        name = "not_finite"

        def predict(self, positions):
            out = np.full((np.asarray(positions).size, len(panel.horizons)), 1.0)
            out[0, 0] = np.nan
            return ExpertForecast(name=self.name, forecast_kw=out, horizons=panel.horizons)

    pool = ExpertPool(
        [PersistenceExpert(panel), NotFinite(), constant_expert("third", 0.0)],
        panel.horizons,
    )
    with pytest.raises(ValueError, match="non-finite forecasts"):
        pool.forecast_block(np.arange(5))


def test_expert_names_are_stable_and_in_reporting_order(stub_pool: ExpertPool) -> None:
    assert stub_pool.names == ("persistence", "classical_hist_gbm", "phase5_tcn")
    assert stub_pool.names == PHASE7_EXPERTS


def test_forecast_expert_protocol_accepts_a_plain_object(
    panel: EvaluationPanel, stub_expert
) -> None:
    assert isinstance(PersistenceExpert(panel), ForecastExpert)
    assert isinstance(stub_expert("x", 0.0), ForecastExpert)
    assert not isinstance(object(), ForecastExpert)


# ---------------------------------------------------------------------------
# The GBM expert's own guards
# ---------------------------------------------------------------------------


def test_gbm_expert_refuses_an_estimate_for_a_missing_horizon() -> None:
    with pytest.raises(ValueError, match=r"no estimator for horizon\(s\) \[4\]"):
        GbmExpert({1: object()}, np.zeros((4, 2)), (1, 4), np.ones(4))


def test_gbm_expert_refuses_positions_beyond_its_feature_matrix() -> None:
    class Estimator:
        def predict(self, matrix):
            return np.zeros(matrix.shape[0])

    expert = GbmExpert(
        {1: Estimator()}, np.zeros((4, 2)), (1,), np.ones(4)
    )
    with pytest.raises(IndexError, match="panel positions reach 9"):
        expert.predict(np.asarray([9]))


def test_gbm_expert_converts_per_unit_estimates_to_kw() -> None:
    """Phase 4 fitted on per-unit targets, so the estimate is per-unit and must be scaled."""

    class Estimator:
        def predict(self, matrix):
            return np.full(matrix.shape[0], 0.25)

    scale = np.asarray([4.0, 8.0])
    expert = GbmExpert({1: Estimator()}, np.zeros((2, 2)), (1,), scale)
    assert np.allclose(expert.predict(np.arange(2)).forecast_kw[:, 0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# Phase 4 feature contract, reproduced verbatim
# ---------------------------------------------------------------------------


def test_phase4_feature_names_are_phase4s_thirteen_in_order() -> None:
    assert PHASE4_FEATURE_NAMES == (
        "hour_of_day",
        "day_of_week",
        "day_of_year",
        "is_weekend",
        "value_at_origin",
        "lag_1",
        "lag_4",
        "lag_96",
        "lag_672",
        "roll_mean_96",
        "roll_std_96",
        "ramp_1",
        "roll_mean_same_hour_7d",
    )


def test_series_context_columns_are_identity_plus_two_fitted_statistics() -> None:
    series = np.asarray([0, 0, 1, 1, 2, 2])
    target = np.asarray([1.0, 3.0, 10.0, 12.0, 0.5, 0.5])
    context, names = _phase4_series_context(series, target, series_count=3)
    assert names == ("series_index", "series_level", "series_volatility")
    assert np.array_equal(context[:, 0], series)
    assert context[0, 1] == pytest.approx(2.0)
    assert context[0, 2] == pytest.approx(1.0)
    assert context[2, 1] == pytest.approx(11.0)
    assert context[4, 1] == pytest.approx(0.5)


def test_series_context_is_a_pure_function_of_the_rows_it_is_given() -> None:
    """The fitted statistics come only from the rows passed in, and nothing is cached.

    Phase 4 created these at fit time precisely so a dataset artifact could not embed a
    split-dependent column (D-066). A helper that carried state between calls would
    reintroduce exactly that leak, so the same rows must give the same numbers however
    many series the caller declared.
    """
    series = np.asarray([0, 0, 0])
    target = np.asarray([1.0, 3.0, 5.0])
    narrow, _ = _phase4_series_context(series, target, series_count=1)
    wide, _ = _phase4_series_context(series, target, series_count=9)
    assert np.array_equal(narrow, wide)
    assert wide[0, 1] == pytest.approx(3.0)
    assert wide[0, 2] == pytest.approx(np.std([1.0, 3.0, 5.0]))


def test_tree_size_counts_nodes_of_a_fitted_boosted_model() -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 4))
    y = x[:, 0] + 0.5 * x[:, 1]
    model = HistGradientBoostingRegressor(
        max_iter=5, max_leaf_nodes=4, early_stopping=False, random_state=0
    ).fit(x, y)
    assert _tree_size(model) > 5


def test_tree_size_is_the_sum_across_boosting_iterations() -> None:
    """Five iterations of a four-leaf tree is five times one tree, not one."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(4)
    x = rng.normal(size=(300, 3))
    y = x[:, 0]
    kwargs = dict(max_leaf_nodes=4, early_stopping=False, random_state=0)
    one = HistGradientBoostingRegressor(max_iter=1, **kwargs).fit(x, y)
    five = HistGradientBoostingRegressor(max_iter=5, **kwargs).fit(x, y)
    assert _tree_size(five) == 5 * _tree_size(one)


def test_tree_size_of_something_without_predictors_is_zero() -> None:
    assert _tree_size(object()) == 0


# ---------------------------------------------------------------------------
# Cache round trip
# ---------------------------------------------------------------------------


def test_expert_cache_round_trips_exactly(tmp_path, panel: EvaluationPanel, forecasts) -> None:
    cache = ExpertCache(
        forecasts=forecasts,
        actual_kw=panel.actual_kw,
        persistence_kw=panel.persistence_kw,
        row_scale=panel.row_scale,
        origins=panel.origins,
        series=panel.series,
        index_rows=panel.index_rows,
        horizons=panel.horizons,
        expert_names=("persistence", "classical_hist_gbm", "phase5_tcn"),
        split_names=("validation", "test"),
        split_bounds=np.asarray(
            [
                [0, panel.split_slices["validation"].stop],
                [panel.split_slices["test"].start, panel.n_rows],
            ],
            dtype=np.int64,
        ),
        metadata={"index_version": "sq-test", "expert_timings_seconds": {"persistence": 0.0}},
    )
    path = save_expert_cache(cache, tmp_path / "experts.npz")
    loaded = load_expert_cache(path)

    assert np.array_equal(loaded.forecasts, cache.forecasts)
    assert loaded.horizons == panel.horizons
    assert loaded.expert_names == cache.expert_names
    assert loaded.metadata["index_version"] == "sq-test"
    assert loaded.split_positions("test").size == cache.split_bounds[1][1] - cache.split_bounds[1][0]

    rebuilt = loaded.panel()
    assert np.array_equal(rebuilt.actual_kw, panel.actual_kw)
    assert rebuilt.horizons == panel.horizons
    assert set(rebuilt.split_slices) == {"validation", "test"}


def test_loading_a_non_cache_file_is_refused(tmp_path) -> None:
    path = tmp_path / "wrong.npz"
    np.savez(path, something_else=np.zeros(3))
    with pytest.raises(ValueError, match="is not an expert cache"):
        load_expert_cache(path)


def test_loading_a_missing_cache_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="expert cache not found"):
        load_expert_cache(tmp_path / "absent.npz")


def test_cache_rejects_an_unknown_split_name(tmp_path, panel: EvaluationPanel, forecasts) -> None:
    cache = ExpertCache(
        forecasts=forecasts,
        actual_kw=panel.actual_kw,
        persistence_kw=panel.persistence_kw,
        row_scale=panel.row_scale,
        origins=panel.origins,
        series=panel.series,
        index_rows=panel.index_rows,
        horizons=panel.horizons,
        expert_names=("a", "b", "c"),
        split_names=("validation", "test"),
        split_bounds=np.asarray([[0, 1], [1, panel.n_rows]], dtype=np.int64),
        metadata={},
    )
    with pytest.raises(KeyError, match="unknown split"):
        cache.split_positions("holdout")