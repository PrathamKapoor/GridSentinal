"""Naive baselines: the minimum any model must beat.

These are not throwaway. A forecasting result without a naive reference is
uninterpretable, because "MAE 2.1 kW" means nothing until you know whether simply
repeating the last value would have given 2.0.

Three, chosen for what this data actually offers:

``last_value``
    ``y[t]`` repeated forward. The persistence floor for any series.
``seasonal_naive_day``
    ``y[t-96]`` - the same time yesterday. The canonical seasonal reference at
    15-minute resolution.
``seasonal_naive_week``
    ``y[t-672]`` - the same time last week. Included because a weekday load
    profile is not the same as a weekend one, so a daily lag carries systematic
    error on two days out of seven.
``drift``
    The last observation plus the mean change over the last hour. Included to test
    whether a trivial trend term helps at all.

None of them is fitted, so all four are deterministic: the same input always gives
the same forecast, and no seed is involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["NaiveForecaster", "NAIVE_MODELS", "fit_naive", "naive_forecast"]


@dataclass(frozen=True, slots=True)
class NaiveForecaster:
    """A fitted-free forecasting rule.

    Attributes:
        name: Model name, as recorded in the registry.
        offsets: How far back each horizon step reads, in steps.
        description: Human-readable rule.
    """

    name: str
    offsets: tuple[int, ...]
    description: str

    def forecast(self, series: np.ndarray, origin: int) -> np.ndarray:
        """Forecast every horizon step from one origin.

        Args:
            series: One series' full history.
            origin: Forecast origin index ``t``.

        Returns:
            One value per horizon step.

        Raises:
            IndexError: If a required past value lies before the start of the
                record, which would mean reaching outside the data.
        """
        offsets = np.asarray(self.offsets, dtype=np.int64)
        indices = origin - offsets
        if int(indices.min()) < 0:
            raise IndexError(
                f"origin {origin} is too early for the {self.name} rule, which reads "
                f"back to t-{int(offsets.max())}"
            )
        return np.asarray(series, dtype=np.float64)[indices]


def _day_offsets(horizons: tuple[int, ...]) -> tuple[int, ...]:
    """For horizon h, read ``t - (96 - h)`` so the forecast is the value one day
    before the *target* step, not one day before the origin."""
    return tuple(96 - h for h in horizons)


def _week_offsets(horizons: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(672 - h for h in horizons)


def _last_offsets(horizons: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(0 for _ in horizons)


#: The naive rules this project evaluates, in reporting order.
NAIVE_MODELS: tuple[str, ...] = (
    "naive_last_value",
    "naive_seasonal_day",
    "naive_seasonal_week",
)


def fit_naive(name: str, horizons: tuple[int, ...]) -> NaiveForecaster:
    """Return the named naive rule for the given horizons.

    Args:
        name: One of :data:`NAIVE_MODELS`, or ``"naive_drift"``.
        horizons: Forecast horizons in steps.

    Raises:
        KeyError: If the model name is unknown, so a typo cannot silently produce
            a missing row in the comparison table.
    """
    if name == "naive_last_value":
        return NaiveForecaster(
            name=name,
            offsets=_last_offsets(horizons),
            description="repeat the value at the forecast origin",
        )
    if name == "naive_seasonal_day":
        return NaiveForecaster(
            name=name,
            offsets=_day_offsets(horizons),
            description="value at the same clock time on the previous day (t - (96 - h))",
        )
    if name == "naive_seasonal_week":
        return NaiveForecaster(
            name=name,
            offsets=_week_offsets(horizons),
            description="value at the same clock time one week earlier (t - (672 - h))",
        )
    if name == "naive_drift":
        return NaiveForecaster(
            name=name,
            offsets=_last_offsets(horizons),
            description="last value plus the mean change over the previous hour",
        )
    raise KeyError(f"unknown naive model {name!r}; available: {list(NAIVE_MODELS) + ['naive_drift']}")


def naive_forecast(
    series: np.ndarray,
    origins: np.ndarray,
    horizons: tuple[int, ...],
    model: str,
) -> np.ndarray:
    """Vectorised forecast for many origins of one series.

    Args:
        series: One series' full history.
        origins: Forecast origins.
        horizons: Forecast horizons.
        model: One of :func:`fit_naive`'s names.

    Returns:
        ``(len(origins), len(horizons))`` predictions.

    Raises:
        ValueError: If any origin is too early for the rule's offsets.
    """
    rule = fit_naive(model, horizons)
    offsets = np.asarray(rule.offsets, dtype=np.int64)
    origin_array = np.asarray(origins, dtype=np.int64)
    index = origin_array[:, None] - offsets[None, :]
    if int(index.min()) < 0:
        raise ValueError(
            f"origin {int(origin_array.min())} is too early for {model}, which reads "
            f"back to t-{int(offsets.max())}"
        )
    values = np.asarray(series, dtype=np.float64)
    if model == "naive_drift":
        window = 4
        history = np.stack([values[origin_array[:, None] - k] for k in range(1, window + 1)], axis=1)
        mean_change = (values[origin_array] - values[origin_array - window]) / window
        return values[index] + mean_change[:, None] * np.arange(
            1, len(horizons) + 1, dtype=np.float64
        )[None, :]
    return values[index]