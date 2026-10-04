"""Router inputs. Everything here must be knowable at the forecast origin.

The router's whole scientific value rests on one property: it must decide using only
what is available *before* the answer exists. The regime terciles that Phases 5 and 6
used for analysis are computed from the demand being predicted, so they are excellent
diagnostics and **useless as router inputs**. That distinction is enforced here by
construction and then by a poisoning test.

Feature groups, all functions of the value at or before the origin:

``level``       the current demand, per-unit and in kW
``trajectory``  trailing means over 4 and 24 steps, and the last ramp
``volatility``  rolling standard deviation over 96 steps
``calendar``    sin/cos of hour-of-day and day-of-year, weekend flag
``scale``       the series' rated kW, which tells the router how much a mistake costs
``past_error``  each expert's **lagged realised** error for the same horizon

``past_error`` deserves its own paragraph because it is the only group that reads the
experts rather than the data, and it is where a leakage mistake would be easiest to
make. At origin ``t`` the router may legitimately know how each expert performed
*already*. For horizon ``h`` the most recent forecast whose outcome is known at ``t``
is the one issued at origin ``t - h - 1``: it targets ``t - 1``. So

.. code-block:: text

    past_error[e] = | f_e(t - h - 1; h) - y(t - 1) |      (per unit)

Using the error of the forecast issued at ``t - 1`` instead would leak, because that
forecast targets ``t + h - 1``, which has not happened yet. The delay is ``h + 1`` for
that reason and no other.

The horizon is not a feature here: it is handled by giving the router a separate output
head per horizon, which is a stronger statement than feeding the horizon in as a number
and hoping the network uses it.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "RouterFeatureSpec",
    "RouterFeatures",
    "FEATURE_GROUPS",
    "router_feature_version",
    "build_router_features",
    "append_lagged_error_features",
    "assert_router_features_are_causal",
]

#: Feature groups, so an ablation can switch one off by name.
FEATURE_GROUPS: tuple[str, ...] = (
    "level",
    "trajectory",
    "volatility",
    "calendar",
    "scale",
    "past_error",
)


@dataclass(frozen=True, slots=True)
class RouterFeatureSpec:
    """Which features the router is allowed to see.

    Attributes:
        groups: Enabled feature groups. An ablation removes one.
        include_level_kw: Whether the raw kW level is included alongside the per-unit
            level. It is origin-observable - it is a measurement, not a target - and it
            tells the router how expensive an error would be in physical units.
        include_past_error: Whether the lagged realised expert errors are included. They
            require the expert forecasts, so they are built by
            :func:`append_lagged_error_features` rather than
            :func:`build_router_features`.
    """

    groups: tuple[str, ...] = FEATURE_GROUPS
    include_level_kw: bool = True
    include_past_error: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": list(self.groups),
            "include_level_kw": self.include_level_kw,
            "include_past_error": self.include_past_error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouterFeatureSpec":
        return cls(
            groups=tuple(payload.get("groups", FEATURE_GROUPS)),
            include_level_kw=bool(payload.get("include_level_kw", True)),
            include_past_error=bool(payload.get("include_past_error", True)),
        )


@dataclass(frozen=True, slots=True)
class RouterFeatures:
    """A feature matrix plus the names that describe its columns.

    ``NaN`` is allowed and means *this row had no usable observation*, never "the
    feature does not exist". Only the lagged-error group can produce it: the earliest
    rows of a split cannot reach back ``h + 1`` steps into a panel that starts there.
    :meth:`fill_missing` resolves it from a stated reference set, and the count is
    recorded rather than silently dropped.

    Attributes:
        matrix: ``[N, n_features]`` float32, origin-observable only.
        names: Column names, same length.
        spec: The specification that produced it.
    """

    matrix: np.ndarray
    names: tuple[str, ...]
    spec: RouterFeatureSpec

    @property
    def n_features(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def missing_counts(self) -> dict[str, int]:
        """Per-column count of rows with no usable observation."""
        if self.matrix.size == 0:
            return {}
        missing = np.isnan(self.matrix)
        if not missing.any():
            return {}
        return {
            name: int(np.count_nonzero(missing[:, index]))
            for index, name in enumerate(self.names)
            if missing[:, index].any()
        }

    def select(self, spec: RouterFeatureSpec) -> "RouterFeatures":
        """Restrict to ``spec``'s groups, for an ablation."""
        if (
            spec.groups == self.spec.groups
            and spec.include_level_kw == self.spec.include_level_kw
            and spec.include_past_error == self.spec.include_past_error
        ):
            return self
        keep = [
            index
            for index, name in enumerate(self.names)
            if _name_in_groups(name, spec)
        ]
        return RouterFeatures(
            matrix=np.ascontiguousarray(self.matrix[:, keep]),
            names=tuple(self.names[i] for i in keep),
            spec=spec,
        )

    def fill_missing(self, reference_rows: np.ndarray) -> "RouterFeatures":
        """Replace ``NaN`` with the column mean over ``reference_rows``.

        The reference rows must come from the **router's own training split**. Using the
        whole panel would let a test row's error rate choose the value a test row is
        scored against, which is test-set information even though it is only a mean.

        Args:
            reference_rows: Row positions whose observed values define the fill.

        Returns:
            A copy with no ``NaN``.

        Raises:
            ValueError: If a column has no observed value at all in ``reference_rows``,
                or if any ``NaN`` remains afterwards.
        """
        reference_rows = np.asarray(reference_rows, dtype=np.int64)
        matrix = self.matrix.copy()
        missing = np.isnan(matrix)
        if not missing.any():
            return RouterFeatures(matrix=matrix, names=self.names, spec=self.spec)
        if reference_rows.size == 0:
            raise ValueError("cannot fill missing router features from zero reference rows")
        for index in np.unique(np.where(missing)[1]):
            column = matrix[:, index]
            observed = column[reference_rows]
            observed = observed[~np.isnan(observed)]
            if observed.size == 0:
                raise ValueError(
                    f"router feature {self.names[index]!r} is missing everywhere in the "
                    f"reference rows; it cannot be imputed"
                )
            column[np.isnan(column)] = float(observed.mean())
            matrix[:, index] = column
        if np.isnan(matrix).any():
            raise AssertionError("router features still contain NaN after filling")
        return RouterFeatures(matrix=matrix, names=self.names, spec=self.spec)

    def describe(self) -> dict[str, Any]:
        finite = np.where(np.isnan(self.matrix), np.nan, self.matrix)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(finite, axis=0) if self.matrix.size else np.zeros(self.n_features)
            std = np.nanstd(finite, axis=0) if self.matrix.size else np.zeros(self.n_features)
        return {
            "n_features": self.n_features,
            "names": list(self.names),
            "spec": self.spec.to_dict(),
            "mean": [None if v != v else float(v) for v in mean],
            "std": [None if v != v else float(v) for v in std],
            "missing": self.missing_counts,
        }


_GROUP_OF_PREFIX = {
    "level": "level",
    "traj": "trajectory",
    "ramp": "trajectory",
    "vol": "volatility",
    "hour": "calendar",
    "doy": "calendar",
    "weekend": "calendar",
    "rated": "scale",
    "past": "past_error",
}


def _name_in_groups(name: str, spec: RouterFeatureSpec) -> bool:
    if name == "level_kw":
        return spec.include_level_kw and "scale" in spec.groups
    prefix = name.split("_")[0]
    group = _GROUP_OF_PREFIX.get(prefix)
    if group is None:
        raise KeyError(f"router feature {name!r} has no declared group")
    if group == "past_error" and not spec.include_past_error:
        return False
    return group in spec.groups


def router_feature_version(spec: RouterFeatureSpec) -> str:
    """A content hash of the feature contract, recorded in every artifact."""
    payload = json.dumps(spec.to_dict(), sort_keys=True)
    return "rf-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def build_router_features(
    *,
    values: np.ndarray,
    panel: Any,
    timestep_minutes: int = 15,
    steps_per_day: int = 96,
) -> RouterFeatures:
    """Build the router's feature matrix for every panel position.

    Args:
        values: ``[n_series, n_steps]`` per-unit values.
        panel: The evaluation panel, supplying origins, series and rated kW.
        timestep_minutes: Native step size, for the calendar channels.
        steps_per_day: Steps per day.

    Returns:
        The features.

    Raises:
        ValueError: If any requested origin is too early for a trailing window, which
            would mean reading before the start of the record.
    """
    origins = np.asarray(panel.origins, dtype=np.int64)
    rows = np.asarray(panel.series, dtype=np.int64)
    if origins.size == 0:
        return RouterFeatures(
            matrix=np.zeros((0, 0), dtype=np.float32), names=(), spec=RouterFeatureSpec()
        )

    earliest = int(origins.min())
    if earliest - 96 < 0:
        raise ValueError(
            f"origin {earliest} is earlier than the 96-step volatility window needs"
        )

    at_origin = values[rows, origins].astype(np.float32)
    trailing_4 = np.stack(
        [values[rows, origins - k] for k in range(4)], axis=1
    ).mean(axis=1).astype(np.float32)
    trailing_24 = np.stack(
        [values[rows, origins - k] for k in range(24)], axis=1
    ).mean(axis=1).astype(np.float32)
    volatility = np.stack(
        [values[rows, origins - k] for k in range(96)], axis=1
    ).std(axis=1).astype(np.float32)
    last_ramp = (values[rows, origins - 1] - values[rows, origins - 2]).astype(np.float32)
    hour = (origins % steps_per_day) * timestep_minutes / 60.0
    day_of_year = origins // steps_per_day
    # Phase 4's convention, exactly: ``day_of_week = absolute_day % 7`` with **no**
    # offset, so 5 and 6 are Saturday and Sunday. An offset here would silently shift the
    # weekend by however many days, and the trace would show ``is_weekend`` set on a
    # Thursday with nothing to indicate why.
    weekend = ((day_of_year % 7) >= 5).astype(np.float32)

    columns: list[np.ndarray] = [
        at_origin,
        trailing_4,
        trailing_24,
        last_ramp,
        volatility,
        np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32),
        np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32),
        np.sin(2.0 * np.pi * day_of_year / 365.25).astype(np.float32),
        np.cos(2.0 * np.pi * day_of_year / 365.25).astype(np.float32),
        weekend,
        np.log1p(np.asarray(panel.row_scale, dtype=np.float32)),
        (np.asarray(panel.row_scale, dtype=np.float32) * at_origin).astype(np.float32),
    ]
    names = [
        "level_pu",
        "traj_mean_4",
        "traj_mean_24",
        "ramp_1",
        "vol_std_96",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "weekend",
        "rated_kw_log",
        "level_kw",
    ]
    matrix = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("router features contain non-finite values")
    return RouterFeatures(
        matrix=matrix, names=tuple(names), spec=RouterFeatureSpec()
    )


def append_lagged_error_features(
    *,
    features: RouterFeatures,
    forecasts: np.ndarray,
    actual_kw: np.ndarray,
    panel: Any,
    horizons: tuple[int, ...],
    spec: RouterFeatureSpec | None = None,
) -> RouterFeatures:
    """Append each expert's lagged realised error for each horizon.

    For panel row ``(s, t)`` and horizon column ``h``:

    .. code-block:: text

        past_err[e, h] = | f_e(t - h - 1; h) - y(t - 1) | / rated_kw(s)

    which is the newest expert outcome that has actually happened by ``t``. Entries
    with no such predecessor in the panel - the first ``h + 1`` origins of each split -
    are ``NaN``, and the accompanying ``past_err_seen_h{h}`` flag says so. There are
    about 2.2% of such rows at ``h=96`` and none at all in the test split.

    The normalisation by rated kW is not cosmetic. A router that weights experts by a
    raw kW error would learn one policy for a 1.4 kW household and another for a 388 kW
    shop, from 40 series; the weights are dimensionless and the per-unit error is the
    dimensionless quantity.

    Args:
        features: Base features from :func:`build_router_features`.
        forecasts: ``[n_experts, N, n_horizons]`` in kW, in panel order.
        actual_kw: ``[N, n_horizons]`` realised targets in kW.
        panel: The evaluation panel.
        horizons: Horizon steps, matching the last axis of ``forecasts``.
        spec: Specification for the result; defaults to the base one with
            ``include_past_error=True``.

    Returns:
        The extended features.

    Raises:
        ValueError: If the shapes disagree or the panel is not ordered as the lookup
            requires.
    """
    spec = spec or RouterFeatureSpec(
        groups=features.spec.groups, include_level_kw=features.spec.include_level_kw
    )
    forecasts = np.asarray(forecasts, dtype=np.float64)
    actual_kw = np.asarray(actual_kw, dtype=np.float64)
    horizons = tuple(horizons)
    n_experts, n_rows, n_horizons = forecasts.shape
    if n_rows != features.matrix.shape[0]:
        raise ValueError(
            f"forecasts have {n_rows} rows but the feature matrix has "
            f"{features.matrix.shape[0]}"
        )
    if n_horizons != len(horizons) or actual_kw.shape != (n_rows, n_horizons):
        raise ValueError(
            f"horizon axes disagree: forecasts {forecasts.shape}, "
            f"actual_kw {actual_kw.shape}, horizons {horizons}"
        )

    origins = np.asarray(panel.origins, dtype=np.int64)
    series = np.asarray(panel.series, dtype=np.int64)
    n_series = int(series.max()) + 1 if series.size else 1
    # ``build_sequence_index`` emits origins ascending and cycles the series inside each
    # origin, so this composite key is strictly increasing and searchable.
    keys = origins * n_series + series
    if keys.size > 1 and np.any(np.diff(keys) <= 0):
        raise ValueError(
            "the panel is not in (origin, series) order, so the lagged-error lookup "
            "would be wrong; rebuild the panel rather than guessing"
        )

    scale = np.asarray(panel.row_scale, dtype=np.float64)
    if np.any(scale <= 0):
        raise ValueError("rated kW must be positive to normalise a lagged error")

    columns: list[np.ndarray] = []
    names: list[str] = []
    for horizon_index, horizon in enumerate(horizons):
        neighbour_origin = origins - horizon - 1
        seen = np.zeros(n_rows, dtype=np.float32)
        values = np.full((n_experts, n_rows), np.nan, dtype=np.float32)
        reachable = neighbour_origin >= 0
        if np.any(reachable):
            wanted = neighbour_origin[reachable] * n_series + series[reachable]
            found = np.searchsorted(keys, wanted)
            inside = found < keys.size
            matched = np.zeros(wanted.shape, dtype=bool)
            matched[inside] = keys[found[inside]] == wanted[inside]
            target_positions = np.flatnonzero(reachable)
            hit_positions = target_positions[matched]
            source_positions = found[matched]
            if hit_positions.size:
                lagged = np.abs(
                    forecasts[:, source_positions, horizon_index]
                    - actual_kw[source_positions, horizon_index][None, :]
                )
                values[:, hit_positions] = (lagged / scale[source_positions][None, :]).astype(
                    np.float32
                )
                seen[hit_positions] = 1.0
        for expert in range(n_experts):
            columns.append(values[expert])
            names.append(f"past_{horizon}_e{expert}")
        columns.append(seen)
        names.append(f"past_seen_{horizon}")

    extra = np.column_stack(columns).astype(np.float32)
    matrix = np.concatenate([features.matrix, extra], axis=1)
    return RouterFeatures(
        matrix=matrix,
        names=features.names + tuple(names),
        spec=spec,
    )




#: Written over the future in the causality check. A large negative sentinel rather
#: than zero or NaN: a real demand value is never negative, so a feature that starts
#: depending on the future changes by a wide margin and cannot be confused with noise.
_POISON = np.float32(-999.0)


def assert_router_features_are_causal(
    *,
    values: np.ndarray,
    panel: Any,
    features: RouterFeatures,
    horizons: tuple[int, ...] | None = None,
    forecasts: np.ndarray | None = None,
    sample: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Prove the features cannot depend on anything after the origin.

    The method is Phase 4's: poison the future and require the features to come back
    bit-identical. Three details matter and all three are easy to get wrong.

    *The boundary* is ``origin + 1`` for **each row's own origin**. The value *at* the
    origin is legitimately an input, so poisoning it would fail - correctly. Poisoning
    from the earliest origin in a series instead would corrupt *earlier* rows' windows
    and is not the property being tested.

    *The rows checked are every row of that series up to and including the chosen
    origin*, not just the chosen row. Checking one row proves very little; poisoning a
    series from ``origin + 1`` onwards must leave its entire history untouched, and
    that is a much sharper statement.

    *The realised targets are recomputed from the poisoned values*, not reused. The
    lagged-error group reads ``y`` through a neighbour row, and handing it the
    uncorrupted target array would make a look-up at ``t + h - 1`` pass this test
    silently. Recomputing ``y`` from the poisoned series closes that door: the target
    at ``origin + horizon`` really does change, so a look-up that reaches past the
    origin is caught.

    Args:
        values: ``[n_series, n_steps]`` per-unit values.
        panel: The evaluation panel the features were built for.
        features: The features to check.
        horizons: Required when ``forecasts`` is given, so the realised targets can be
            recomputed. Must match the panel's horizons.
        forecasts: ``[n_experts, N, n_horizons]`` in kW, when the lagged-error group is
            present. Omit to check the origin-observable columns only.
        sample: How many series/origin pairs to poison.
        seed: Seed for the deterministic sample.

    Returns:
        A record of the check.

    Raises:
        AssertionError: If any feature changes when the row's future is poisoned.
        ValueError: If ``forecasts`` is given without ``horizons``, or vice versa.
"""
    rows = np.asarray(panel.series, dtype=np.int64)
    origins = np.asarray(panel.origins, dtype=np.int64)
    if rows.size == 0:
        return {"checked_rows": 0, "checked_rows_verified": 0, "passed": True}
    if (forecasts is None) != (horizons is None):
        raise ValueError(
            "forecasts and horizons must be supplied together: the realised targets "
            "are recomputed from the poisoned values, which needs the horizons"
        )

    has_lagged = forecasts is not None
    checked_rows = 0
    count = min(int(sample), rows.size)
    picks = np.sort(
        np.random.default_rng(seed).choice(rows.size, size=count, replace=False)
    )
    for position in picks:
        position = int(position)
        series_id = int(rows[position])
        origin = int(origins[position])
        history = np.flatnonzero((rows == series_id) & (origins <= origin))
        poisoned = values.copy()
        poisoned[series_id, origin + 1 :] = _POISON
        view = _PanelView(panel, history)
        candidate = build_router_features(values=poisoned, panel=view)
        if has_lagged:
            horizons_t = tuple(int(h) for h in horizons)  # type: ignore[arg-type]
            actual = (
                poisoned[series_id, view.origins[:, None] + np.asarray(horizons_t)[None, :]]
                * view.row_scale[:, None]
            )
            candidate = append_lagged_error_features(
                features=candidate,
                forecasts=np.asarray(forecasts)[:, history, :],
                actual_kw=actual,
                panel=view,
                horizons=horizons_t,
)
        expected = features.matrix[history]
        if not np.array_equal(expected, candidate.matrix, equal_nan=True):
            if expected.shape[1] != candidate.matrix.shape[1]:
                # Reported before the per-column diff, because indexing the two arrays in
                # lockstep would raise IndexError instead - and an IndexError from a
                # leakage guard reads as a bug in the guard rather than as the mismatch it
                # was meant to report.
                raise AssertionError(
                    f"the rebuilt feature matrix has {candidate.matrix.shape[1]} columns "
                    f"but the supplied one has {expected.shape[1]}, for series "
                    f"{series_id} at or before origin {origin}. The supplied features are "
                    f"not the ones these builders produce, so the poisoning comparison "
                    f"cannot be made and no causality claim can rest on it"
                )
            differing = [
                name
                for index, name in enumerate(features.names)
                if not np.array_equal(expected[:, index], candidate.matrix[:, index], equal_nan=True)
            ]
            raise AssertionError(
                f"router features changed for series {series_id} at or before origin "
                f"{origin} when values after that origin were poisoned: {differing}"
            )
        checked_rows += int(history.size)
    return {
        "checked_rows": count,
        "checked_rows_verified": checked_rows,
        "checked_of": int(rows.size),
        "checked_series": int(np.unique(rows[picks]).size),
        "boundary": "values strictly after each poisoned row's own origin",
        "lagged_error_group_checked": has_lagged,
        "features": list(features.names),
        "passed": True,
    }


class _PanelView:
    """A panel restricted to specific positions, for the per-row poisoning check."""

    def __init__(self, panel: Any, positions: np.ndarray) -> None:
        positions = np.asarray(positions, dtype=np.int64)
        self.origins = np.asarray(panel.origins)[positions]
        self.series = np.asarray(panel.series)[positions]
        self.row_scale = np.asarray(panel.row_scale)[positions]
        self.split = np.asarray(panel.split)[positions]
        self.split_slices: dict[str, slice] = {}
        for name in ("validation", "test"):
            self.split_slices[name] = slice(0, 0)
        self.actual_kw = np.asarray(panel.actual_kw)[positions]
        self.persistence_kw = np.asarray(panel.persistence_kw)[positions]
        self.index_rows = np.asarray(panel.index_rows)[positions]
        self.horizons = tuple(panel.horizons)

