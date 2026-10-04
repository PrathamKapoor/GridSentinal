"""Phase 7, part 3: leakage, and the isolation of the oracle.

Three things have to be true, and each can be checked mechanically rather than promised:

1. **No router feature depends on anything after the forecast origin.** Proven by
   poisoning the future of a whole series and requiring every earlier row's features to
   come back bit-identical.
2. **The oracle cannot be reached from a deployable path.** Enforced by reading the
   source of every deployable module, not by reading it once and hoping.
3. **The lagged expert-error feature really does lag by ``h + 1``**, which is the one place
   where an off-by-one leaks a future target.

The third deserves a note. For horizon ``h``, the newest expert outcome known at origin
``t`` is the one issued at ``t - h - 1``: it targets ``t - 1``. Using the forecast issued at
``t - 1`` instead would read a target at ``t + h - 1``, which has not happened yet. That is
a real leak, it is subtle, and it is the sort of thing that produces excellent offline
numbers and a system that fails in the field.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from energy_intelligence.ml.router.experts import EvaluationPanel
from energy_intelligence.ml.router.features import (
    FEATURE_GROUPS,
    RouterFeatureSpec,
    append_lagged_error_features,
    assert_router_features_are_causal,
    build_router_features,
    router_feature_version,
)

ROUTER_DIR = Path(__file__).resolve().parents[2] / "src" / "energy_intelligence" / "ml" / "router"

#: Modules a live system is allowed to import. Everything else under ``router/`` is either
#: offline analysis or a test-only helper, and naming them here makes the boundary an
#: assertion rather than a convention.
DEPLOYABLE_MODULES = (
    "cache.py",
    "ensembles.py",
    "experts.py",
    "features.py",
    "model.py",
    "routing.py",
    "training.py",
)


# ---------------------------------------------------------------------------
# 1. Causality of the router's inputs
# ---------------------------------------------------------------------------


def _full_features(panel: EvaluationPanel, values: np.ndarray, forecasts: np.ndarray):
    base = build_router_features(values=values, panel=panel)
    return append_lagged_error_features(
        features=base,
        forecasts=forecasts,
        actual_kw=panel.actual_kw,
        panel=panel,
        horizons=panel.horizons,
    )


def test_router_features_survive_poisoning_the_future(
    panel: EvaluationPanel, values: np.ndarray, forecasts: np.ndarray
) -> None:
    """Poison everything after one origin; every row up to it must be unchanged.

    The check covers whole series histories rather than single rows, so it is a much
    stronger statement than "one row did not move", and the poisoned value is a large
    negative sentinel so a leak cannot be mistaken for numerical noise.
    """
    features = _full_features(panel, values, forecasts)
    record = assert_router_features_are_causal(
        values=values,
        panel=panel,
        features=features,
        horizons=panel.horizons,
        forecasts=forecasts,
        sample=6,
        seed=1,
    )
    assert record["passed"]
    assert record["lagged_error_group_checked"] is True
    assert record["checked_rows"] == 6
    assert record["checked_rows_verified"] > record["checked_rows"]
    assert "strictly after" in record["boundary"]


def test_causality_check_reports_when_only_the_base_group_was_checked(
    panel: EvaluationPanel, values: np.ndarray
) -> None:
    base = build_router_features(values=values, panel=panel)
    record = assert_router_features_are_causal(
        values=values, panel=panel, features=base, sample=3, seed=0
    )
    assert record["lagged_error_group_checked"] is False


def test_causality_check_requires_forecasts_and_horizons_together(
    panel: EvaluationPanel, values: np.ndarray, forecasts: np.ndarray
) -> None:
    base = build_router_features(values=values, panel=panel)
    with pytest.raises(ValueError, match="must be supplied together"):
        assert_router_features_are_causal(
            values=values, panel=panel, features=base, forecasts=forecasts, sample=2
        )


def test_causality_check_actually_catches_a_leaking_builder(
    panel: EvaluationPanel, values: np.ndarray, forecasts: np.ndarray
) -> None:
    """The guard must fail on a deliberately broken feature, or it proves nothing."""
    import energy_intelligence.ml.router.features as module

    original = module.build_router_features

    def leaky(*, values, panel, **kwargs):
        built = original(values=values, panel=panel, **kwargs)
        origins = np.asarray(panel.origins)
        series = np.asarray(panel.series)
        # Deliberately read one step *after* the origin.
        column = values[series, np.minimum(origins + 1, values.shape[1] - 1)].astype(
            np.float32
        )
        matrix = np.column_stack([built.matrix, column])
        return module.RouterFeatures(
            matrix=matrix,
            names=built.names + ("leaky",),
            spec=built.spec,
        )

    features = leaky(values=values, panel=panel)
    module.build_router_features = leaky
    try:
        with pytest.raises(AssertionError, match="leaky"):
            assert_router_features_are_causal(
                values=values,
                panel=panel,
                features=features,
                sample=4,
                seed=2,
            )
    finally:
        module.build_router_features = original


def test_a_feature_group_ablation_never_widens_the_input(
    panel: EvaluationPanel, values: np.ndarray, forecasts: np.ndarray
) -> None:
    features = _full_features(panel, values, forecasts)
    full = set(features.names)
    for group in FEATURE_GROUPS:
        reduced = features.select(
            RouterFeatureSpec(
                groups=tuple(g for g in FEATURE_GROUPS if g != group)
            )
        )
        assert set(reduced.names).issubset(full)
        assert reduced.n_features < features.n_features
        assert np.array_equal(
            reduced.matrix,
            features.matrix[:, [features.names.index(n) for n in reduced.names]],
            equal_nan=True,
        )


def test_unknown_feature_name_has_no_declared_group(
    panel: EvaluationPanel, values: np.ndarray
) -> None:
    from energy_intelligence.ml.router.features import RouterFeatures

    features = RouterFeatures(
        matrix=np.zeros((2, 1), dtype=np.float32),
        names=("mystery_column",),
        spec=RouterFeatureSpec(),
    )
    with pytest.raises(KeyError, match="no declared group"):
        features.select(RouterFeatureSpec(groups=("level",)))


def test_an_origin_too_early_for_the_volatility_window_is_refused() -> None:
    """Reading before the start of the record would silently index from the end."""
    from energy_intelligence.ml.router.features import build_router_features

    panel = EvaluationPanel(
        index_rows=np.asarray([0]),
        split=np.asarray(["validation"], dtype=object),
        split_slices={"validation": slice(0, 1), "test": slice(1, 1)},
        origins=np.asarray([10]),
        series=np.asarray([0]),
        row_scale=np.asarray([1.0]),
        actual_kw=np.ones((1, 1)),
        persistence_kw=np.ones((1, 1)),
        horizons=(1,),
    )
    with pytest.raises(ValueError, match="earlier than the 96-step volatility window"):
        build_router_features(values=np.ones((1, 500), dtype=np.float32), panel=panel)


def test_feature_version_changes_with_the_contract(panel: EvaluationPanel) -> None:
    full = RouterFeatureSpec()
    reduced = RouterFeatureSpec(groups=("level", "calendar"))
    assert router_feature_version(full) != router_feature_version(reduced)
    assert router_feature_version(full) == router_feature_version(RouterFeatureSpec())
    assert router_feature_version(full).startswith("rf-")


def test_router_feature_spec_round_trips() -> None:
    original = RouterFeatureSpec(
        groups=("level", "scale"), include_level_kw=False, include_past_error=False
    )
    assert RouterFeatureSpec.from_dict(original.to_dict()) == original


# ---------------------------------------------------------------------------
# 2. The lagged expert error lags by exactly h + 1
# ---------------------------------------------------------------------------


def test_lagged_error_reads_the_forecast_issued_h_plus_1_steps_earlier(
    panel: EvaluationPanel, values: np.ndarray, forecasts: np.ndarray
) -> None:
    """Hand-check the arithmetic for horizon 1 on one row.

    With ``h = 1`` the neighbour must be the origin ``t - 2``, whose target is ``t - 1``.
    Reaching one step further would read the realised target at ``t``, which is the answer.
    """
    features = _full_features(panel, values, forecasts)
    names = features.names
    h1 = [index for index, name in enumerate(names) if name.startswith("past_1_e")]
    seen = names.index("past_seen_1")
    h96 = [index for index, name in enumerate(names) if name.startswith("past_96_e")]

    # A row far enough into the panel that even the h=96 neighbour (97 origins back) has a
    # predecessor to find. The panel begins at its own first origin, so early rows simply
    # have no long-horizon history - which is exactly what the availability flag records.
    row = panel.n_rows - 6
    origin = int(panel.origins[row])
    series = int(panel.series[row])
    neighbour_origin = origin - 2
    keys = panel.origins * (int(panel.series.max()) + 1) + panel.series
    neighbour = int(np.flatnonzero(keys == neighbour_origin * (int(panel.series.max()) + 1) + series)[0])

    scale = float(panel.row_scale[neighbour])
    for position, index in enumerate(h1):
        expected = abs(forecasts[position, neighbour, 0] - panel.actual_kw[neighbour, 0]) / scale
        assert features.matrix[row, index] == pytest.approx(expected, rel=1e-5)

    # h=96 needs an origin 97 steps back, so its value must come from a *different*
    # neighbour than h=1 does. If they were the same, the delay would not be h-dependent.
    h96_origin = origin - 97
    h96_neighbour = int(
        np.flatnonzero(
            keys == h96_origin * (int(panel.series.max()) + 1) + series
        )[0]
    )
    assert h96_neighbour != neighbour
    assert features.matrix[row, h96[0]] == pytest.approx(
        abs(forecasts[0, h96_neighbour, 2] - panel.actual_kw[h96_neighbour, 2])
        / float(panel.row_scale[h96_neighbour]),
        rel=1e-5,
    )
    assert features.matrix[row, seen] in (0.0, 1.0)


def test_lagged_error_is_nan_exactly_where_no_predecessor_exists(
    panel: EvaluationPanel, values: np.ndarray, forecasts: np.ndarray
) -> None:
    features = _full_features(panel, values, forecasts)
    names = features.names
    seen = np.asarray([names.index(f"past_seen_{h}") for h in panel.horizons])
    flags = features.matrix[:, seen]
    values_at = np.asarray(
        [[names.index(f"past_{h}_e0")] for h in panel.horizons]
    )
    for row, horizon_column in zip(range(panel.n_rows), range(3)):
        for column in range(3):
            has_it = flags[row, horizon_column] == 1.0
            has_value = not np.isnan(features.matrix[row, values_at[column, 0]])
            assert has_it == has_value
    # The availability flag is monotonic in horizon: the longer the delay, the earlier the
    # neighbour, so h=96 is available for fewer rows than h=1.
    assert np.count_nonzero(flags[:, 0]) >= np.count_nonzero(flags[:, 1])
    assert np.count_nonzero(flags[:, 1]) >= np.count_nonzero(flags[:, 2])


def test_lagged_error_is_normalised_per_unit_so_a_big_customer_is_not_a_big_feature(
    panel: EvaluationPanel, values: np.ndarray, forecasts: np.ndarray
) -> None:
    """A raw kW error would make one 388 kW shop dominate a 1.4 kW household's routing."""
    features = _full_features(panel, values, forecasts)
    name = "past_1_e0"
    if name not in features.names:
        pytest.skip("no lagged-error column in this configuration")
    column = features.matrix[:, features.names.index(name)]
    observed = column[~np.isnan(column)]
    assert observed.max() < 1.0


def test_fill_missing_uses_only_the_reference_rows(
    panel: EvaluationPanel, values: np.ndarray, forecasts: np.ndarray
) -> None:
    features = _full_features(panel, values, forecasts)
    name = "past_96_e0"
    if name not in features.names:
        pytest.skip("no lagged-error column in this configuration")
    index = features.names.index(name)
    assert features.missing_counts.get(name, 0) > 0

    # The reference set has to include rows deep enough into the panel for the long
    # horizon to have any predecessor at all - which is why the real experiment uses the
    # router's own training rows rather than the panel's first rows.
    reference = np.arange(panel.n_rows - 30, panel.n_rows)
    filled_a = features.fill_missing(reference)
    corrupted = features.matrix.copy()
    corrupted[: panel.n_rows - 30, index] = np.nan
    variant = type(features)(
        matrix=corrupted, names=features.names, spec=features.spec
    )
    filled_b = variant.fill_missing(reference)
    # Rows inside the reference set keep their own values, and the fill constant is
    # unchanged because it is computed from the same rows. An implementation that
    # imputed from the whole panel would see the corrupted second half and differ here.
    assert np.array_equal(
        filled_a.matrix[reference, index], filled_b.matrix[reference, index]
    )
    assert filled_a.missing_counts == {}


def test_fill_missing_refuses_a_column_with_no_observation(
    panel: EvaluationPanel, values: np.ndarray
) -> None:
    from energy_intelligence.ml.router.features import RouterFeatures

    features = RouterFeatures(
        matrix=np.full((4, 1), np.nan, dtype=np.float32),
        names=("all_missing",),
        spec=RouterFeatureSpec(),
    )
    with pytest.raises(ValueError, match="missing everywhere"):
        features.fill_missing(np.arange(4))
    with pytest.raises(ValueError, match="zero reference rows"):
        features.fill_missing(np.asarray([], dtype=np.int64))


def test_lagged_error_rejects_a_panel_that_is_not_in_origin_order(
    panel: EvaluationPanel, values: np.ndarray, forecasts: np.ndarray
) -> None:
    """The binary search is only valid on a sorted composite key, and says so."""
    shuffled = EvaluationPanel(
        index_rows=panel.index_rows,
        split=panel.split,
        split_slices=panel.split_slices,
        origins=panel.origins[::-1].copy(),
        series=panel.series[::-1].copy(),
        row_scale=panel.row_scale,
        actual_kw=panel.actual_kw,
        persistence_kw=panel.persistence_kw,
        horizons=panel.horizons,
    )
    base = build_router_features(values=values, panel=panel)
    with pytest.raises(ValueError, match="not in \\(origin, series\\) order"):
        append_lagged_error_features(
            features=base,
            forecasts=forecasts,
            actual_kw=panel.actual_kw,
            panel=shuffled,
            horizons=panel.horizons,
        )


def test_lagged_error_rejects_shapes_that_disagree(
    panel: EvaluationPanel, values: np.ndarray, forecasts: np.ndarray
) -> None:
    base = build_router_features(values=values, panel=panel)
    with pytest.raises(ValueError, match="forecasts have"):
        append_lagged_error_features(
            features=base,
            forecasts=forecasts[:, :10, :],
            actual_kw=panel.actual_kw,
            panel=panel,
            horizons=panel.horizons,
        )
    with pytest.raises(ValueError, match="horizon axes disagree"):
        append_lagged_error_features(
            features=base,
            forecasts=forecasts,
            actual_kw=panel.actual_kw,
            panel=panel,
            horizons=(1, 4),
        )


# ---------------------------------------------------------------------------
# 3. The oracle cannot be reached from a deployable path
# ---------------------------------------------------------------------------


def _imported_modules(path: Path) -> set[str]:
    """Every module name a Python file imports, resolved to their last component."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("module_name", DEPLOYABLE_MODULES)
def test_no_deployable_module_imports_the_offline_analysis(module_name: str) -> None:
    path = ROUTER_DIR / module_name
    assert path.is_file(), f"{module_name} is named as deployable but does not exist"
    imported = _imported_modules(path)
    offenders = {
        name
        for name in imported
        if "offline" in name or name.endswith("oracle") or name.endswith("evaluation")
    }
    assert not offenders, (
        f"{module_name} imports offline analysis: {sorted(offenders)}. The oracle reads "
        f"the realised target and must never be reachable from a live routing path."
    )


def _referenced_identifiers(path: Path) -> set[str]:
    """Every identifier the module actually uses: names, attributes, args, aliases.

    Docstrings and comments are ``Constant`` nodes and so are excluded, which is the
    point. A module is allowed to *explain* that the oracle is off limits; it is not
    allowed to name one in code.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.arg):
            found.add(node.arg)
        elif isinstance(node, ast.alias):
            found.add(node.asname or node.name)
        elif isinstance(node, ast.keyword):
            found.add(node.arg or "")
    return found


@pytest.mark.parametrize("module_name", DEPLOYABLE_MODULES)
def test_no_deployable_module_names_the_oracle_in_code(module_name: str) -> None:
    """Naming the oracle in code - not in prose - is the thing being forbidden."""
    offenders = {
        name
        for name in _referenced_identifiers(ROUTER_DIR / module_name)
        if "oracle" in name.lower()
    }
    assert not offenders, (
        f"{module_name} names the oracle in code: {sorted(offenders)}. Prose may explain "
        f"why it is unreachable, but no deployable path may touch it."
    )


def test_the_offline_subpackage_exists_and_declares_the_boundary() -> None:
    package = ROUTER_DIR / "offline" / "__init__.py"
    assert package.is_file()
    text = package.read_text(encoding="utf-8")
    assert "router.offline" in text
    assert (ROUTER_DIR / "offline" / "oracle.py").is_file()


def test_the_oracle_module_declares_itself_undeployable() -> None:
    from energy_intelligence.ml.router.offline.oracle import ORACLE_NAME, oracle_report

    report = oracle_report(
        forecasts=np.stack([np.ones((4, 1)), np.zeros((4, 1)), np.full((4, 1), 2.0)]),
        actual_kw=np.full((4, 1), 0.9),
        horizons=(1,),
        expert_names=("a", "b", "c"),
    )
    assert report["is_deployable"] is False
    assert report["name"] == ORACLE_NAME
    assert "upper bound" in report["definition"]


def test_the_oracle_is_correct_and_cannot_be_better_than_the_best_expert() -> None:
    from energy_intelligence.ml.router.offline.oracle import (
        oracle_assignment,
        oracle_forecast,
        oracle_report,
    )

    truth = np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    forecasts = np.stack(
        [
            np.asarray([[5.0, 0.0], [1.0, 30.0], [9.0, 12.0], [2.0, 8.0]]),
            np.asarray([[0.5, 20.0], [4.0, 2.0], [1.0, 11.0], [3.0, 14.0]]),
            np.asarray([[7.0, 1.0], [3.0, 25.0], [4.0, 30.0], [1.5, 9.0]]),
        ]
    )
    picks = oracle_assignment(forecasts, truth)
    assert picks.shape == (4, 2)
    # Row 0: expert b is nearest at h=1 (0.5 off) and expert c is nearest at h=4 (9 off).
    assert np.array_equal(picks[0], [1, 2])
    # Row 1: expert a is nearest at h=1 (1 off) and expert b at h=4 (8 off versus c's 15).
    assert np.array_equal(picks[1], [0, 1])
    chosen = oracle_forecast(forecasts, truth)
    assert chosen.shape == truth.shape
    # A per-row oracle can never be worse than the best fixed expert.
    report = oracle_report(
        forecasts=forecasts,
        actual_kw=truth,
        horizons=(1, 4),
        expert_names=("a", "b", "c"),
    )
    for key in ("1", "4"):
        entry = report["by_horizon"][key]
        assert entry["oracle_mae_kw"] <= entry["best_single_mae_kw"] + 1e-12
        assert entry["oracle_gain_kw"] >= -1e-12
        assert entry["expert_share"]["b"] > 0.0


def test_oracle_mixture_is_not_an_upper_bound_on_a_state_dependent_router() -> None:
    """The documented caveat, as an executable claim.

    ``oracle_mixture`` is the best *constant* weighting chosen in hindsight. A mixture
    whose weights depend on the row can beat it, and Phase 7 measured that it does at
    h=1. Asserting the relationship here keeps the docstring honest if the code changes.
    """
    from energy_intelligence.ml.router.offline.oracle import oracle_mixture

    # Three experts that each predict a constant: 0, 5 and 10 kW. Any fixed convex
    # combination is itself a constant, so on a series alternating between 0 and 10 it
    # cannot beat 5 kW of mean error whatever the weights are. Choosing the low expert on
    # low rows and the high expert on high rows is state-dependent, needs no knowledge of
    # the target beyond the state, and is exact.
    truth = np.asarray([[0.0], [10.0], [0.0], [10.0], [0.0], [10.0]])
    forecasts = np.stack(
        [
            np.zeros_like(truth),
            np.full_like(truth, 5.0),
            np.full_like(truth, 10.0),
        ]
    )
    fixed = oracle_mixture(forecasts, truth)
    state_dependent = np.where(truth < 5.0, forecasts[0], forecasts[2])
    assert float(np.abs(fixed - truth).mean()) == pytest.approx(5.0)
    assert float(np.abs(state_dependent - truth).mean()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The router's calendar convention is Phase 4's
# ---------------------------------------------------------------------------


def test_the_router_weekend_flag_matches_phase4s_convention(
    panel: EvaluationPanel, values: np.ndarray
) -> None:
    """Phase 4's ``is_weekend`` is ``(absolute_day % 7) >= 5`` with no day offset.

    The router's own calendar columns must agree with the model it routes over. An offset
    here shifts the weekend by however many days, which degrades the routing slightly and
    shows up nowhere - it is only visible by comparing against Phase 4's own arithmetic.
    """
    built = build_router_features(values=values, panel=panel)
    column = built.names.index("weekend")
    expected = ((panel.origins // 96) % 7 >= 5).astype(np.float32)
    assert np.array_equal(built.matrix[:, column], expected)

    from energy_intelligence.ml.features import build_feature_matrix

    phase4 = build_feature_matrix(
        values,
        panel.origins,
        lookback=672,
        feature_names=("hour_of_day", "day_of_week", "day_of_year", "is_weekend"),
        series_index=panel.series,
    )
    assert np.allclose(built.matrix[:, column], phase4[:, 3])


def test_the_router_hour_channel_matches_phase4s(
    panel: EvaluationPanel, values: np.ndarray
) -> None:
    built = build_router_features(values=values, panel=panel)
    sin = built.matrix[:, built.names.index("hour_sin")]
    cos = built.matrix[:, built.names.index("hour_cos")]
    hour = (panel.origins % 96) * 15 / 60.0
    assert np.allclose(sin, np.sin(2.0 * np.pi * hour / 24.0), atol=1e-6)
    assert np.allclose(cos, np.cos(2.0 * np.pi * hour / 24.0), atol=1e-6)
