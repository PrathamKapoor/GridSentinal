"""The calibration split, and an honest account of what it may and may not touch.

Uncertainty needs data that was not used to fit the point forecast, and - for the conformal
methods - data that was not used to fit the scale either. This module fixes that
partition once, for the whole phase, so no method can quietly get a better split than its
neighbour.

The partition
-------------

.. code-block:: text

    TRAIN          Phase 4/5 split. 950,400 rows.  The experts were fitted here.
                            |
    VALIDATION     Phase 5/7 split. 179,520 rows.  Expert predictions are out-of-sample.
        +-- CAL_FIT      first 50%, 89,760 rows.  Scale functions and quantile models.
        +-- CAL_CONF     last  50%, 89,760 rows.  Conformal conformity scores.
                            |
    TEST           Phase 5/7 split. 179,520 rows.  Read once, for the final evaluation.

Two deliberate choices, both stated because the alternative would have been wrong in a way
no test would catch.

**Why validation is halved rather than using Phase 7's own 70/30 router split.** Phase 7's
router was fitted on the first 70% of validation and selected on the last 30%. Its fitted
fixed-ensemble weights therefore come from those first 70% of rows. Reusing Phase 7's split
would mean the scale model is fitted on rows the fixed ensemble's three numbers per horizon
were also fitted on - a mild optimism for three parameters, but Phase 7's cross-fitting
check (§14 of its design doc) measured that in-sample expert behaviour changes the
achievable headroom by up to 2x, so "mild" was measured to be the wrong word to rely on.

**Why the overlap is measured anyway rather than assumed negligible.**
:func:`split_parity_check` reports the fixed ensemble's MAE on rows its own weights saw
(CAL_FIT), rows they did not (CAL_CONF), and on test. If the three agree, the overlap is
immaterial and that is a number rather than an assurance. If they disagree, the partition
is wrong and the report says so.

The test split is never touched by anything in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["CalibrationSplit", "build_calibration_split", "split_parity_check"]


@dataclass(frozen=True, slots=True)
class CalibrationSplit:
    """Which panel rows each part of Phase 8 may use.

    Attributes:
        calibration_fit: Rows on which scale functions and quantile models are fitted.
        calibration_conformity: Rows supplying conformal conformity scores. Disjoint from
            ``calibration_fit`` by construction.
        test: The sealed evaluation rows.
        validation: Every validation row, the union of the two calibration parts.
        horizons: Horizon steps.
        cal_fraction: The share of validation used for fitting, before the conformity part.
    """

    calibration_fit: np.ndarray
    calibration_conformity: np.ndarray
    test: np.ndarray
    validation: np.ndarray
    horizons: tuple[int, ...]
    cal_fraction: float

    def __post_init__(self) -> None:
        fit = np.asarray(self.calibration_fit, dtype=np.int64)
        conform = np.asarray(self.calibration_conformity, dtype=np.int64)
        test = np.asarray(self.test, dtype=np.int64)
        validation = np.asarray(self.validation, dtype=np.int64)
        if fit.size == 0 or conform.size == 0 or test.size == 0:
            raise ValueError(
                f"every split must be non-empty; got fit={fit.size}, "
                f"conformity={conform.size}, test={test.size}"
            )
        overlap = np.intersect1d(fit, conform)
        if overlap.size:
            raise ValueError(
                f"the scale-fitting rows and the conformity rows overlap on "
                f"{overlap.size} rows; a scale fitted on rows whose residuals it then "
                f"normalises depends on the very values it is meant to predict"
            )
        for name, rows in (
            ("calibration_fit", fit),
            ("calibration_conformity", conform),
        ):
            stray = np.intersect1d(rows, test)
            if stray.size:
                raise ValueError(
                    f"{name} contains {stray.size} test rows; the test split is sealed"
                )
        calibration_union = np.union1d(fit, conform)
        if not np.array_equal(calibration_union, np.unique(validation)):
            raise ValueError(
                "calibration_fit and calibration_conformity must together be the whole "
                f"validation split: their union holds {calibration_union.size} rows while "
                f"validation holds {np.unique(validation).size}. A calibration row outside "
                "validation would be either a training row - where the point forecast is "
                "in-sample - or a sealed test row"
            )

    def describe(self) -> dict[str, Any]:
        return {
            "calibration_fit_rows": int(self.calibration_fit.size),
            "calibration_conformity_rows": int(self.calibration_conformity.size),
            "validation_rows": int(self.validation.size),
            "test_rows": int(self.test.size),
            "cal_fraction": float(self.cal_fraction),
            "horizons": [int(h) for h in self.horizons],
            "ordering": "chronological within the validation split",
            "guarantee": (
                "the scale functions are fitted on calibration_fit and their conformal "
                "multipliers come from calibration_conformity, which is disjoint; the "
                "test rows are read once"
            ),
        }


def build_calibration_split(
    *,
    validation_rows: np.ndarray,
    test_rows: np.ndarray,
    origins: np.ndarray,
    horizons: tuple[int, ...],
    cal_fraction: float = 0.5,
) -> CalibrationSplit:
    """Split validation chronologically into a fitting and a conformity portion.

    Args:
        validation_rows: Panel positions of the validation split.
        test_rows: Panel positions of the sealed test split.
        origins: ``[N]`` forecast origin index, used to order chronologically. Sorting on
            position alone would be wrong if the panel were ever reordered, and the
            conformal guarantee depends on order.
        horizons: Horizon steps.
        cal_fraction: Share of validation used for fitting, in ``(0, 1)``.

    Returns:
        The split.

    Raises:
        ValueError: If ``cal_fraction`` is outside ``(0, 1)``, or either input is empty.
    """
    if not 0.0 < float(cal_fraction) < 1.0:
        raise ValueError(f"cal_fraction must lie in (0, 1), got {cal_fraction}")
    validation = np.asarray(validation_rows, dtype=np.int64)
    test = np.asarray(test_rows, dtype=np.int64)
    if validation.size == 0 or test.size == 0:
        raise ValueError(
            f"both splits must be non-empty; got validation={validation.size}, "
            f"test={test.size}"
        )
    all_origins = np.asarray(origins, dtype=np.int64)
    order = np.argsort(all_origins[validation], kind="mergesort")
    ordered = validation[order]
    cut = int(round(ordered.size * float(cal_fraction)))
    cut = max(1, min(cut, ordered.size - 1))
    return CalibrationSplit(
        calibration_fit=ordered[:cut],
        calibration_conformity=ordered[cut:],
        test=test,
        validation=ordered,
        horizons=tuple(int(h) for h in horizons),
        cal_fraction=float(cal_fraction),
    )


def split_parity_check(
    *,
    point_kw: np.ndarray,
    actual_kw: np.ndarray,
    split: CalibrationSplit,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    """Report the point forecast's error on each split, to expose any split optimism.

    Three numbers per horizon: rows the fixed-ensemble weights were fitted on
    (``calibration_fit``), rows they were not (``calibration_conformity``), and the sealed
    test rows. If the first is materially lower than the others, the uncertainty methods
    are being fitted on optimistic residuals and every width they produce is too narrow.

    Returns:
        Per-horizon MAE on each split, plus the relative gap between the fitting and
        conformity portions.
    """
    from ..metrics import mae

    point = np.asarray(point_kw, dtype=np.float64)
    actual = np.asarray(actual_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)
    out: dict[str, Any] = {}
    for column, horizon in enumerate(horizons):
        entry: dict[str, Any] = {}
        for label, rows in (
            ("calibration_fit", split.calibration_fit),
            ("calibration_conformity", split.calibration_conformity),
            ("validation_all", split.validation),
            ("test", split.test),
        ):
            entry[f"{label}_mae_kw"] = float(
                mae(actual[rows, column], point[rows, column])
            )
            entry[f"{label}_rows"] = int(rows.size)
        reference = entry["calibration_conformity_mae_kw"]
        entry["fit_vs_conformity_pct"] = (
            100.0 * (entry["calibration_fit_mae_kw"] - reference) / reference
            if reference > 0
            else None
        )
        entry["test_vs_conformity_pct"] = (
            100.0 * (entry["test_mae_kw"] - reference) / reference if reference > 0 else None
        )
        out[str(horizon)] = entry
    return out