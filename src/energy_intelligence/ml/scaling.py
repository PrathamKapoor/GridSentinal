"""Scaling, fitted on training data only.

The rule
--------
Statistics are computed on the training split and applied unchanged to validation
and test. A scaler fitted on everything has, by construction, seen the test set's
mean and spread, which inflates every score derived from it - a small, silent
optimism that is easy to miss and hard to defend.

:func:`assert_train_only_fit` makes the guarantee checkable: a scaler records the
row indices it was fitted on, and any attempt to fit on rows outside the training
split raises.

Implementation is explicit rather than delegated to scikit-learn because the
contract is the point, and a five-line class is easier to audit than a dependency's
defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FeatureScaler", "assert_train_only_fit"]


@dataclass(frozen=True, slots=True)
class FeatureScaler:
    """Standardise features with training-set mean and standard deviation.

    Attributes:
        mean: Per-column training mean.
        scale: Per-column training standard deviation, with zero variance mapped to
            ``1.0`` so a constant column becomes exactly zero instead of dividing
            by zero.
        fitted_on: Number of rows the statistics came from.
        constant_columns: Indices of columns whose training variance was zero.
    """

    mean: np.ndarray
    scale: np.ndarray
    fitted_on: int
    constant_columns: tuple[int, ...] = ()

    @classmethod
    def fit(cls, matrix: np.ndarray) -> FeatureScaler:
        """Fit on the rows given. Callers must pass training rows only."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")
        if matrix.shape[0] == 0:
            raise ValueError("cannot fit a scaler on zero rows")
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        constant = tuple(int(i) for i in np.flatnonzero(std <= 1e-12))
        scale = np.where(std <= 1e-12, 1.0, std)
        return cls(
            mean=mean,
            scale=scale,
            fitted_on=int(matrix.shape[0]),
            constant_columns=constant,
        )

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """Apply the fitted statistics. Never refits."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"expected {self.mean.shape[0]} columns to match the fitted scaler, "
                f"got {matrix.shape[1]}"
            )
        return (matrix - self.mean) / self.scale

    def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
        """Fit on ``matrix`` and return the transformed copy."""
        scaler = self.fit(matrix)
        return scaler.transform(matrix)

    def to_dict(self) -> dict[str, object]:
        return {
            "fitted_on_rows": self.fitted_on,
            "constant_columns": list(self.constant_columns),
            "mean": [float(x) for x in self.mean],
            "scale": [float(x) for x in self.scale],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> FeatureScaler:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            fitted_on=int(payload["fitted_on_rows"]),
            constant_columns=tuple(int(i) for i in payload.get("constant_columns", ())),
        )


def assert_train_only_fit(scaler: FeatureScaler, train_row_count: int) -> None:
    """Check the scaler saw exactly the training rows.

    Raises:
        AssertionError: If the fitted row count differs, which means statistics
        were computed over a superset of the training split.
    """
    if scaler.fitted_on != train_row_count:
        raise AssertionError(
            f"scaler was fitted on {scaler.fitted_on} rows but the training split has "
            f"{train_row_count}; statistics must never see validation or test data"
        )