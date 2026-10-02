"""Chronological splitting, and the leakage rules that go with it.

Why this module exists
----------------------
Randomly shuffling a year of energy data before splitting produces a model that
has already seen the future. Every sample here is therefore keyed by its
**forecast origin** on the native grid, and every split boundary is a grid index,
so "training data" and "the past" mean the same thing.

Two distinct leaks are prevented, and both are tested:

1. **Split leakage.** A training sample's *entire* lookback window and its entire
   target window must lie inside the training index range. A sample at the very
   end of training whose lookback reaches into validation would import future
   information through its own inputs. See :func:`split_boundaries` and
   :func:`sample_splits`.
2. **Target leakage.** A feature may only read values at or before its origin.
   That is enforced in :mod:`features` by construction, not by review.

Proportions are a configuration choice, not a law of nature. The default is
70 / 15 / 15 by time, which for one year of 15-minute data gives roughly 256 days
of training, 55 days of validation and 55 days of test. The test span contains the
dataset's own peak timepoint (grid index 19553 = 2018-07-23), so peak-period error
is measured on data no model has seen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SPLIT_TRAIN",
    "SPLIT_VALIDATION",
    "SPLIT_TEST",
    "SPLIT_NAMES",
    "SplitBoundaries",
    "split_boundaries",
    "sample_splits",
    "split_summary",
]

SPLIT_TRAIN = 0
SPLIT_VALIDATION = 1
SPLIT_TEST = 2

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class SplitBoundaries:
    """Half-open grid-index ranges ``[lo, hi)`` per split.

    ``train`` covers the earliest indices. Ranges are contiguous and together span
    the full axis, so no grid step is unassigned; a sample's *windows* are then
    additionally required to fit inside one range.
    """

    train: tuple[int, int]
    validation: tuple[int, int]
    test: tuple[int, int]

    def as_tuple(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        return (self.train, self.validation, self.test)

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }


def split_boundaries(
    total_steps: int,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> SplitBoundaries:
    """Split the grid chronologically into three contiguous ranges.

    Args:
        total_steps: Length of the native time axis.
        train_fraction: Share of the axis used for training, from the start.
        validation_fraction: Share used for validation, immediately after training.

    Returns:
        The three contiguous ranges.

    Raises:
        ValueError: If the fractions do not form three non-empty ranges. An empty
            test range would silently remove the only honest evaluation data.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must lie in (0, 1), got {train_fraction}")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must lie in (0, 1), got {validation_fraction}")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "train_fraction + validation_fraction must leave a non-empty test range; got "
            f"{train_fraction} + {validation_fraction}"
        )

    train_hi = int(round(total_steps * train_fraction))
    validation_hi = int(round(total_steps * (train_fraction + validation_fraction)))
    if train_hi < 1 or validation_hi <= train_hi or validation_hi >= total_steps:
        raise ValueError(
            f"fractions produce an empty split for {total_steps} steps "
            f"(train<{train_hi}, validation<{validation_hi})"
        )
    return SplitBoundaries(
        train=(0, train_hi),
        validation=(train_hi, validation_hi),
        test=(validation_hi, total_steps),
    )


def sample_splits(
    origins: np.ndarray,
    boundaries: SplitBoundaries,
    *,
    lookback: int,
    max_horizon: int,
) -> np.ndarray:
    """Assign each sample the split its **entire window** fits inside.

    A sample with forecast origin ``t`` is train-only when both
    ``t - lookback >= train_lo`` and ``t + max_horizon < train_hi``. Otherwise its
    inputs or its targets would reach across a boundary into later data.

    A sample whose windows straddle two ranges is **dropped** rather than assigned
    to either, which is why the per-split sample counts are slightly smaller than
    a naive index count would suggest.

    Args:
        origins: Forecast-origin grid index per sample.
        boundaries: The three ranges.
        lookback: Steps of history a sample consumes.
        max_horizon: Largest horizon step the sample's target window reaches.

    Returns:
        An ``int8`` array of split codes, with ``-1`` marking a dropped sample.

    Raises:
        ValueError: If ``lookback`` or ``max_horizon`` is not positive, since that
            would mean the windows are empty.
    """
    if lookback <= 0:
        raise ValueError(f"lookback must be positive, got {lookback}")
    if max_horizon <= 0:
        raise ValueError(f"max_horizon must be positive, got {max_horizon}")

    codes = np.full(origins.shape, -1, dtype=np.int8)
    for code, (lo, hi) in enumerate(boundaries.as_tuple()):
        fits = (origins - lookback >= lo) & (origins + max_horizon < hi)
        codes[fits] = code
    return codes


def split_summary(codes: np.ndarray) -> dict[str, object]:
    """Count samples per split, including how many were dropped."""
    kept = codes[codes >= 0]
    dropped = int(codes.size - kept.size)
    return {
        "total_samples": int(codes.size),
        "dropped_crossing_boundary": dropped,
        "train": int(np.count_nonzero(kept == SPLIT_TRAIN)),
        "validation": int(np.count_nonzero(kept == SPLIT_VALIDATION)),
        "test": int(np.count_nonzero(kept == SPLIT_TEST)),
    }