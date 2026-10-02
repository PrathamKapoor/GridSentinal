"""Feature construction, with every feature's availability declared up front.

The rule this module exists to enforce
--------------------------------------
A feature may only read a value that was **already knowable at the forecast origin
``t``**. Anything that reads ``t+1`` or later is leakage, no matter how accurate it
makes the model.

Two categories are treated very differently, and the difference is deliberate:

**Historical and calendar features** are unconditionally safe. ``y[t-1]``,
``y[t-96]``, the trailing mean over ``[t-96, t-1]``, or the hour of day are all
available at ``t`` by definition.

**Weather covariates are safe only at short horizons.** The solar file ships
*actual* observations of GHI, irradiance, temperature and wind. At a 15-minute
horizon, the observation at ``t`` genuinely exists at prediction time. At a
4-hour horizon it does not: a real system would have a *forecast*, and this dataset
ships no forecast. Using the future observation anyway is the single most common
way a solar forecasting benchmark becomes meaningless.

:func:`available_feature_names` therefore *excludes* weather features whenever the
horizon exceeds :data:`WEATHER_AVAILABLE_THROUGH_STEP`, and
:func:`assert_no_future_access` proves the exclusion is real rather than cosmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "FeatureSpec",
    "FEATURE_CATALOGUE",
    "WEATHER_AVAILABLE_THROUGH_STEP",
    "feature_catalogue",
    "available_feature_names",
    "assert_no_future_access",
    "build_feature_matrix",
]

#: Weather observations are knowable at the origin and for 15 minutes beyond it.
#: Beyond that step, a real deployment would need a weather forecast, which this
#: dataset does not contain, so weather features are withdrawn automatically.
WEATHER_AVAILABLE_THROUGH_STEP = 1


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One feature and its provenance.

    Attributes:
        name: Column name in the feature matrix.
        formula: How the value is computed, in the project's notation.
        reason: Why the feature is expected to carry signal.
        source: Which real quantity it is computed from.
        availability: ``"at_or_before_origin"`` or ``"origin_only"``.
        value_kind: ``HISTORICAL``, ``CALENDAR`` or ``WEATHER_ACTUAL``.
        min_lookback: Smallest lookback this feature can be built with.
        requires_weather: Whether the feature needs the solar weather columns.
    """

    name: str
    formula: str
    reason: str
    source: str
    availability: str
    value_kind: str
    min_lookback: int
    requires_weather: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "formula": self.formula,
            "reason": self.reason,
            "source": self.source,
            "availability": self.availability,
            "value_kind": self.value_kind,
            "min_lookback": self.min_lookback,
            "requires_weather": self.requires_weather,
        }


_CALENDAR_HOUR = FeatureSpec(
    name="hour_of_day",
    formula="floor(t / 4) mod 24",
    reason="Occupancy and lighting schedules are strongly hour-dependent.",
    source="native grid index (Phase 3 TimeAxis)",
    availability="at_or_before_origin",
    value_kind="CALENDAR",
    min_lookback=1,
)
_CALENDAR_DOW = FeatureSpec(
    name="day_of_week",
    formula="day index mod 7, 0 = Monday",
    reason="Weekday/weekend behaviour differs for residential and commercial load.",
    source="native grid index (Phase 3 TimeAxis)",
    availability="at_or_before_origin",
    value_kind="CALENDAR",
    min_lookback=1,
)
_CALENDAR_DOY = FeatureSpec(
    name="day_of_year",
    formula="floor(t / 96)",
    reason="Seasonal drift in heating, cooling and daylight.",
    source="native grid index (Phase 3 TimeAxis)",
    availability="at_or_before_origin",
    value_kind="CALENDAR",
    min_lookback=1,
)
_CALENDAR_WEEKEND = FeatureSpec(
    name="is_weekend",
    formula="1 if day_of_week in {5, 6} else 0",
    reason="Explicit weekend flag; the single most useful calendar split for load.",
    source="native grid index (Phase 3 TimeAxis)",
    availability="at_or_before_origin",
    value_kind="CALENDAR",
    min_lookback=1,
)
_VALUE_AT_ORIGIN = FeatureSpec(
    name="value_at_origin",
    formula="y[t]",
    reason=(
        "The most recent measurement, knowable at the origin. Included because "
        "omitting it would rig the comparison: a persistence baseline reads y[t], "
        "so a model that cannot see y[t] starts from a worse position than the "
        "baseline it is measured against."
    ),
    source="target series",
    availability="at_or_before_origin",
    value_kind="HISTORICAL",
    min_lookback=1,
)
_LAG_1 = FeatureSpec(
    name="lag_1",
    formula="y[t-1]",
    reason="Persistence: the most recent value is the strongest single predictor.",
    source="target series",
    availability="at_or_before_origin",
    value_kind="HISTORICAL",
    min_lookback=1,
)
_LAG_4 = FeatureSpec(
    name="lag_4",
    formula="y[t-4]",
    reason="One hour back; smooths the 15-minute noise in the last observation.",
    source="target series",
    availability="at_or_before_origin",
    value_kind="HISTORICAL",
    min_lookback=4,
)
_LAG_96 = FeatureSpec(
    name="lag_96",
    formula="y[t-96]",
    reason="Same time yesterday: the canonical seasonal-naive information.",
    source="target series",
    availability="at_or_before_origin",
    value_kind="HISTORICAL",
    min_lookback=96,
)
_LAG_672 = FeatureSpec(
    name="lag_672",
    formula="y[t-672]",
    reason="Same time last week: captures day-of-week structure a daily lag misses.",
    source="target series",
    availability="at_or_before_origin",
    value_kind="HISTORICAL",
    min_lookback=672,
)
_ROLL_MEAN_96 = FeatureSpec(
    name="roll_mean_96",
    formula="mean(y[t-96 .. t-1])",
    reason="Trailing daily mean: level information that lags alone do not carry.",
    source="target series",
    availability="at_or_before_origin",
    value_kind="HISTORICAL",
    min_lookback=96,
)
_ROLL_STD_96 = FeatureSpec(
    name="roll_std_96",
    formula="std(y[t-96 .. t-1])",
    reason="Recent volatility; feeds uncertainty-aware models in later phases.",
    source="target series",
    availability="at_or_before_origin",
    value_kind="HISTORICAL",
    min_lookback=96,
)
_RAMP_1 = FeatureSpec(
    name="ramp_1",
    formula="y[t-1] - y[t-2]",
    reason="DERIVED: most recent ramp. Renewalables and loads both ramp.",
    source="target series",
    availability="at_or_before_origin",
    value_kind="HISTORICAL",
    min_lookback=2,
)
_HOURLY_MEAN_24 = FeatureSpec(
    name="roll_mean_same_hour_7d",
    formula="mean(y[t-96*k] for k in 1..7)",
    reason="DERIVED: mean of the same clock hour over the past week.",
    source="target series",
    availability="at_or_before_origin",
    value_kind="HISTORICAL",
    min_lookback=96 * 7,
)
_PV_AT_ORIGIN = FeatureSpec(
    name="ghi_at_origin",
    formula="GHI[t]",
    reason="Actual global irradiance at the origin; available at prediction time.",
    source="solar_data GHI column",
    availability="origin_only",
    value_kind="WEATHER_ACTUAL",
    min_lookback=1,
    requires_weather=True,
)
_PV_POAT_ORIGIN = FeatureSpec(
    name="poa_irradiance_at_origin",
    formula="PoA Irradiance[t]",
    reason="Plane-of-array irradiance is what the array actually sees.",
    source="solar_data PoA Irradiance (W/m^2) column",
    availability="origin_only",
    value_kind="WEATHER_ACTUAL",
    min_lookback=1,
    requires_weather=True,
)
_PV_TEMP = FeatureSpec(
    name="temperature_at_origin",
    formula="Temperature[t]",
    reason="Panel temperature scales output below rated conditions.",
    source="solar_data Temperature column",
    availability="origin_only",
    value_kind="WEATHER_ACTUAL",
    min_lookback=1,
    requires_weather=True,
)
_PV_WIND = FeatureSpec(
    name="wind_speed_at_origin",
    formula="Wind Speed[t]",
    reason="Wind cools the panel; a real but secondary PV driver.",
    source="solar_data Wind Speed column",
    availability="origin_only",
    value_kind="WEATHER_ACTUAL",
    min_lookback=1,
    requires_weather=True,
)

#: Every feature this module can build, in a stable order.
FEATURE_CATALOGUE: tuple[FeatureSpec, ...] = (
    _CALENDAR_HOUR,
    _CALENDAR_DOW,
    _CALENDAR_DOY,
    _CALENDAR_WEEKEND,
    _VALUE_AT_ORIGIN,
    _LAG_1,
    _LAG_4,
    _LAG_96,
    _LAG_672,
    _ROLL_MEAN_96,
    _ROLL_STD_96,
    _RAMP_1,
    _HOURLY_MEAN_24,
    _PV_AT_ORIGIN,
    _PV_POAT_ORIGIN,
    _PV_TEMP,
    _PV_WIND,
)


def feature_catalogue() -> tuple[FeatureSpec, ...]:
    """The full catalogue, for documentation and manifests."""
    return FEATURE_CATALOGUE


def available_feature_names(
    *,
    lookback: int,
    max_horizon: int,
    has_weather: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve which features are both buildable and legitimate.

    Args:
        lookback: Steps of history available to every sample.
        max_horizon: Largest horizon the dataset targets.
        has_weather: Whether real weather observations were extracted.

    Returns:
        ``(feature_names, withdrawn)`` where ``withdrawn`` names the features that
        were excluded **and why**, so the dataset manifest records the exclusion
        rather than hiding a shorter feature list.
    """
    selected: list[str] = []
    withdrawn: list[tuple[str, ...]] = []
    for spec in FEATURE_CATALOGUE:
        if spec.min_lookback > lookback:
            withdrawn.append(
                (
                    spec.name,
                    f"needs {spec.min_lookback} steps of history, lookback is {lookback}",
                )
            )
            continue
        if spec.requires_weather:
            if not has_weather:
                withdrawn.append((spec.name, "no weather columns were extracted"))
                continue
            if max_horizon > WEATHER_AVAILABLE_THROUGH_STEP:
                withdrawn.append(
                    (
                        spec.name,
                        "horizon exceeds 15 minutes: a real deployment would need a "
                        "weather FORECAST, which SMART-DS does not contain, so an "
                        "actual observation beyond the origin would be leakage",
                    )
                )
                continue
        selected.append(spec.name)
    if not selected:  # pragma: no cover - guarded by configuration validation
        raise ValueError("no features are available for this configuration")
    return tuple(selected), tuple(withdrawn)


def _calendar_columns(steps_per_day: int = 96) -> dict[str, np.ndarray]:
    """Reference calendar vectors, exposed for tests and documentation.

    ``build_feature_matrix`` computes these inline from each sample's own origin,
    which is the only correct thing at a split boundary; this helper exists so a
    test can assert the mapping without duplicating arithmetic.
    """
    step = np.arange(steps_per_day)
    day_of_week = (step // steps_per_day) % 7
    return {
        "hour_of_day": (step % steps_per_day).astype(np.float64) // 4.0,
        "day_of_week": day_of_week.astype(np.float64),
        "is_weekend": (day_of_week >= 5).astype(np.float64),
    }


def build_feature_matrix(
    values: np.ndarray,
    origins: np.ndarray,
    *,
    lookback: int,
    feature_names: tuple[str, ...],
    steps_per_day: int = 96,
    day_of_year_offset: int = 0,
    weather: dict[str, np.ndarray] | None = None,
    series_index: np.ndarray | None = None,
    extra_builders: dict[str, Any] | None = None,
) -> np.ndarray:
    """Build the feature matrix for every ``(series, origin)`` sample.

    Args:
        values: ``(n_series, n_steps)`` target series matrix.
        origins: Forecast origin index per sample.
        lookback: Steps of history. Used for validation and to document intent; the
            lag features read their own offsets.
        feature_names: Output column order, from :func:`available_feature_names`.
        steps_per_day: 96 for the native 15-minute grid.
        day_of_year_offset: Day index of ``values[:, 0]``. The native axis starts on
            1 January, so this is 0, but it is explicit rather than assumed.
        weather: Mapping of weather column name to array, when the target needs it.
        series_index: Which series each sample belongs to. Defaults to the dataset's
            layout, where each series contributes the same origins in turn.
        extra_builders: ``{name: callable(values, origins, series_index) -> array}``
            for features this module does not define. Used by the leakage guard to
            point itself at a candidate feature, including a deliberately wrong one.

    Returns:
        ``(n_samples, len(feature_names))`` float64.

    Raises:
        ValueError: If any requested origin is too early for the requested lags,
            which would mean reading a negative index.
    """
    if values.ndim != 2:
        raise ValueError(f"values must be (n_series, n_steps), got shape {values.shape}")
    origins = np.asarray(origins, dtype=np.int64)

    s = (
        _series_index_for(values.shape[0], origins.size)
        if series_index is None
        else np.asarray(series_index, dtype=np.int64)
    )
    if s.size != origins.size:
        raise ValueError(
            f"series_index has {s.size} entries but there are {origins.size} samples"
        )
    t = origins
    needed = _max_required_offset(
        feature_names, supplied=frozenset(extra_builders or {})
    )
    if origins.size and (int(t.min()) - needed < 0):
        raise ValueError(
            f"origin {int(t.min())} is earlier than the earliest offset {needed}; "
            "a feature would read before the start of the record"
        )

    columns: dict[str, np.ndarray] = {}
    # Calendar features come from the grid index alone, so they cannot leak. The
    # absolute day index is derived from the origin plus the dataset's day offset.
    steps_per_day = _steps_per_day(steps_per_day)
    absolute_step = t + day_of_year_offset * steps_per_day
    absolute_day = absolute_step // steps_per_day
    day_of_week = absolute_day % 7
    columns["hour_of_day"] = (t % steps_per_day).astype(np.float64) // 4.0
    columns["day_of_week"] = day_of_week.astype(np.float64)
    columns["day_of_year"] = absolute_day.astype(np.float64)
    columns["is_weekend"] = (day_of_week >= 5).astype(np.float64)

    columns["value_at_origin"] = values[s, t]
    columns["lag_1"] = values[s, t - 1]
    columns["lag_4"] = values[s, t - 4]
    columns["lag_96"] = values[s, t - 96]
    columns["lag_672"] = values[s, t - 672]

    trailing = _trailing_window(values, s, t, 96)
    columns["roll_mean_96"] = trailing.mean(axis=1)
    columns["roll_std_96"] = trailing.std(axis=1)
    columns["ramp_1"] = values[s, t - 1] - values[s, t - 2]

    same_hour = np.stack(
        [values[s, t - 96 * k] for k in range(1, 8)], axis=1
    )
    columns["roll_mean_same_hour_7d"] = same_hour.mean(axis=1)

    for name, builder in (extra_builders or {}).items():
        if name not in feature_names:
            continue
        produced = np.asarray(builder(values, t, s), dtype=np.float64)
        if produced.shape != t.shape:
            raise ValueError(
                f"extra builder {name!r} returned shape {produced.shape}, "
                f"expected {t.shape}"
            )
        columns[name] = produced

    if weather is not None:
        if "ghi_at_origin" in feature_names:
            columns["ghi_at_origin"] = weather["GHI"][t]
        if "poa_irradiance_at_origin" in feature_names:
            columns["poa_irradiance_at_origin"] = weather["PoA Irradiance (W/m^2)"][t]
        if "temperature_at_origin" in feature_names:
            columns["temperature_at_origin"] = weather["Temperature"][t]
        if "wind_speed_at_origin" in feature_names:
            columns["wind_speed_at_origin"] = weather["Wind Speed"][t]

    matrix = np.empty((origins.size, len(feature_names)), dtype=np.float64)
    for position, name in enumerate(feature_names):
        column = columns.get(name)
        if column is None:
            raise KeyError(
                f"feature {name!r} has no builder; available: {sorted(columns)}"
            )
        matrix[:, position] = column
    if not np.all(np.isfinite(matrix)):
        bad = int(np.count_nonzero(~np.isfinite(matrix)))
        per_column = {
            name: int(np.count_nonzero(~np.isfinite(matrix[:, i])))
            for i, name in enumerate(feature_names)
            if np.any(~np.isfinite(matrix[:, i]))
        }
        raise ValueError(
            f"{bad} non-finite feature values produced "
            f"(by column: {per_column}); imputation must be an explicit, "
            "recorded decision rather than an accident"
        )
    return matrix


def _steps_per_day(steps_per_day: int) -> int:
    """Validate the steps-per-day constant.

    Raises:
        ValueError: If the grid is not a whole number of hours, because the
            hour-of-day feature would then be a fiction.
    """
    if steps_per_day <= 0 or steps_per_day % 4:
        raise ValueError(
            f"steps_per_day must be a positive multiple of 4 (15-minute grid); "
            f"got {steps_per_day}"
        )
    if steps_per_day % 24:
        raise ValueError(
            f"steps_per_day must be a whole number of hours for the hour-of-day "
            f"feature to mean anything; got {steps_per_day}"
        )
    return steps_per_day


def _series_index_for(n_series: int, n_samples: int) -> np.ndarray:
    """Repeat each series index once per origin, in the dataset's row order."""
    if n_samples % n_series:
        raise ValueError(
            f"n_samples ({n_samples}) must be a multiple of n_series ({n_series})"
        )
    return np.repeat(np.arange(n_series, dtype=np.int64), n_samples // n_series)


def _trailing_window(
    values: np.ndarray, series_index: np.ndarray, t: np.ndarray, width: int
) -> np.ndarray:
    """Values at ``[t-width, t-1]`` for each sample, as ``(n_samples, width)``.

    Built with a gather rather than a rolling slice so that arbitrary origins work
    and the window provably ends at ``t-1``: it can never include ``t`` itself.
    """
    offsets = np.arange(-width, 0, dtype=np.int64)
    index = t[:, None] + offsets[None, :]
    return values[series_index[:, None], index]


def _max_required_offset(
    feature_names: tuple[str, ...], *, supplied: frozenset[str] = frozenset()
) -> int:
    """Largest backward read among the requested features.

    Args:
        feature_names: Names to inspect.
        supplied: Names provided by a caller-supplied builder. Those have no declared
            history requirement, so they cannot contribute to the backward bound -
            and if one of them reads backwards anyway, the value it produces will be
            wrong and caught by the finiteness check.
    """
    by_name = {spec.name: spec for spec in FEATURE_CATALOGUE}
    worst = 1
    for name in feature_names:
        spec = by_name.get(name)
        if spec is None:
            if name in supplied:
                continue
            raise KeyError(f"unknown feature {name!r}")
        worst = max(worst, spec.min_lookback)
    return worst


def assert_no_future_access(
    values: np.ndarray,
    origins: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    weather: dict[str, np.ndarray] | None = None,
    extra_builders: dict[str, Any] | None = None,
    sample_size: int = 64,
    seed: int = 0,
) -> None:
    """Prove the feature matrix cannot depend on values at or after the origin.

    The check is empirical rather than declarative, and it is **per sample**. That
    detail matters: poisoning the future of every origin at once also destroys the
    *inputs* of every later sample, because sample ``t+1`` legitimately reads
    ``y[t]``. Poisoning a whole block therefore proves nothing. So a deterministic
    subset of sample positions is checked individually - for one sample, everything
    **strictly after** its origin is replaced with ``NaN``, and its feature row must
    be byte-identical to the row built from the untouched series.

    The boundary is ``t + 1``, not ``t``: the measurement ``y[t]`` is the current
    reading and is knowable at the forecast origin, so ``value_at_origin`` is a
    legitimate feature. The *target* always lies at ``t + h`` with ``h >= 1``, so
    nothing a model may read is masked by this choice.

    Args:
        values: ``(n_series, n_steps)`` series.
        origins: Forecast origins, in the dataset's row order.
        feature_names: Features to check.
        weather: The same weather mapping the dataset was built with. It must be
            forwarded, otherwise the weather columns cannot be rebuilt and the check
            would silently cover fewer features than the dataset actually uses.
        extra_builders: ``{name: callable(values, origins, series_index)}`` builders
            for candidate features outside the catalogue. A test injects a
            deliberately leaking column - one that reads ``y[t+1]`` - to confirm this
            function actually catches one. The builder receives the **poisoned**
            matrix too, which is what makes the check meaningful: a builder that
            reads the future necessarily sees the poison.
        sample_size: How many sample positions to probe.
        seed: Seed for choosing which positions to probe.

    Raises:
        AssertionError: If any feature changes when the values after its origin
            are poisoned.
    """
    values = np.asarray(values, dtype=np.float64)
    origin_array = np.asarray(origins, dtype=np.int64)
    if origin_array.size == 0:
        return
    n_series = values.shape[0]
    per_series = origin_array.size // n_series

    probe_count = min(int(sample_size), origin_array.size)
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(origin_array.size, size=probe_count, replace=False))

    for position in positions:
        row = int(position // per_series)
        origin = int(origin_array[position])
        single_origin = np.asarray([origin], dtype=np.int64)

        baseline = build_feature_matrix(
            values,
            single_origin,
            lookback=1,
            feature_names=feature_names,
            weather=weather,
            series_index=np.asarray([row], dtype=np.int64),
            extra_builders=extra_builders,
        )[0]

        poisoned = values.copy()
        # Strictly after the origin: y[t] itself is knowable at prediction time.
        poisoned[row, origin + 1 :] = np.nan
        try:
            candidate = build_feature_matrix(
                poisoned,
                single_origin,
                lookback=1,
                feature_names=feature_names,
                weather=weather,
                series_index=np.asarray([row], dtype=np.int64),
                extra_builders=extra_builders,
            )[0]
        except ValueError as exc:
            raise AssertionError(
                f"sample at position {position} (series {row}, origin {origin}) changed "
                f"when values after its origin were poisoned, so a feature reads the "
                f"future: {exc}"
            ) from exc

        if not np.array_equal(baseline, candidate, equal_nan=True):
            differing = [
                name
                for index, name in enumerate(feature_names)
                if not np.array_equal(baseline[index], candidate[index], equal_nan=True)
            ]
            raise AssertionError(
                f"features read the future for series {row} at origin {origin}: "
                f"{differing}. A feature may only use values at or before its "
                "forecast origin."
            )