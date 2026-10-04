"""Temporal folds and the common-sample accounting that keeps the ablation honest.

Two failure modes this module exists to prevent.

**Leakage through the final test split.** The final test range is never returned to a caller
of :func:`ablation_folds`, and :func:`assert_final_test_denied` is the runtime guard. Phase 10
is a feature study: choosing features on test data would make the eventual benchmark
meaningless, and the guard makes that a raised error rather than a review comment.

**The ablation silently becoming a missing-row study.** Feature sets with more features are
built from longer histories, so the largest set naturally has the *fewest* usable origins. If
each set were scored on whatever rows it happened to have, a worse model could lose rows and
look better. Every run therefore reports its usable ``(origin, series)`` identity, and
:func:`common_timestamps` intersects those identities across feature sets so the comparison can
be made on shared rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Fold",
    "CONFIRMATION_FOLDS",
    "SELECTION_FOLDS",
    "ablation_folds",
    "common_timestamps",
    "SampleAccounting",
    "assert_final_test_denied",
    "FinalTestAccessError",
]


class FinalTestAccessError(RuntimeError):
    """Raised when Phase 10 code path reaches for the final test split.

    A dedicated type so a test can assert on it and so an operator sees an unmistakable
    reason rather than a shape mismatch.
    """


@dataclass(frozen=True, slots=True)
class Fold:
    """One contiguous block of forecast origins.

    Attributes:
        fold_id: ``F01``..``F06``.
        role: ``ABLATION`` for F01-F04, ``CONFIRMATION`` for F05-F06.
        start: Inclusive origin index.
        stop: Exclusive origin index.
    """

    fold_id: str
    role: str
    start: int
    stop: int

    @property
    def size(self) -> int:
        return int(self.stop - self.start)

    def contains(self, origin: int) -> bool:
        return self.start <= int(origin) < self.stop

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "role": self.role,
            "start": int(self.start),
            "stop": int(self.stop),
            "origins": self.size,
        }


#: The four folds that may be used to SELECT a feature set.
SELECTION_FOLDS: tuple[str, ...] = ("F01", "F02", "F03", "F04")
#: The two folds reserved for post-selection confirmation.
CONFIRMATION_FOLDS: tuple[str, ...] = ("F05", "F06")


def ablation_folds(validation_start: int, validation_stop: int) -> tuple[Fold, ...]:
    """Split the validation range into F01-F06, contiguously and in order.

    F01-F04 tile the first half of validation and F05-F06 the second half. Every feature set
    is scored on the *same* origin ranges, so no set can be advantaged by being evaluated on a
    different, later, or easier period.

    The final test range is not an input and is never produced: validation is the outer bound,
    and anything at or beyond ``validation_stop`` is rejected by
    :func:`assert_final_test_denied`.

    Args:
        validation_start: First origin index of the validation split.
        validation_stop: One past the last origin index of validation.

    Returns:
        Six :class:`Fold` objects in order.

    Raises:
        ValueError: If the range is too small to split six ways.
    """
    start, stop = int(validation_start), int(validation_stop)
    if stop <= start:
        raise ValueError(f"validation range is empty: [{start}, {stop})")
    width = stop - start
    if width < 6:
        raise ValueError(
            f"validation range [{start}, {stop}) has {width} origins, too few to split into "
            f"six folds; at least one origin per fold is required"
        )
    half = width // 2
    quarter = half // 4
    if quarter < 1:
        raise ValueError(
            f"validation range [{start}, {stop}) has {width} origins; the first half "
            f"({half}) cannot be divided into four folds"
        )
    folds: list[Fold] = []
    cursor = start
    for index, fold_id in enumerate(SELECTION_FOLDS):
        edge = start + quarter * (index + 1)
        folds.append(Fold(fold_id, "ABLATION", cursor, edge))
        cursor = edge
    remaining = stop - cursor
    third = max(remaining // 3, 1)
    for offset, fold_id in enumerate(CONFIRMATION_FOLDS):
        edge = stop if offset == len(CONFIRMATION_FOLDS) - 1 else cursor + third
        folds.append(Fold(fold_id, "CONFIRMATION", cursor, edge))
        cursor = edge
    assert_final_test_denied(stop)
    return tuple(folds)


def assert_final_test_denied(validation_stop: int, requested_stop: int | None = None) -> None:
    """Refuse any origin range that reaches into or past the final test split.

    This is the runtime guard behind the protocol freeze's
    ``FINAL_TEST_FEATURE_SELECTION_ACCESS = NO``. It is cheap, it takes no data, and it is
    called from the fold builder itself so a future caller cannot forget it.

    Args:
        validation_stop: One past the last validation origin.
        requested_stop: The stop index actually requested, when different.

    Raises:
        FinalTestAccessError: If the requested range extends beyond validation.
    """
    limit = int(validation_stop)
    if requested_stop is not None and int(requested_stop) > limit:
        raise FinalTestAccessError(
            f"requested origin range stops at {int(requested_stop)}, past the validation "
            f"split boundary {limit}. Phase 10 must not read the final test split for "
            f"feature selection or evaluation; F01-F04 select and F05-F06 confirm."
        )


@dataclass(frozen=True, slots=True)
class SampleAccounting:
    """One feature set's usable rows on one fold, with their identity.

    Attributes:
        target: Target id.
        fold_id: Fold identifier.
        feature_set: Feature set id.
        origins: Sorted unique forecast-origin step indices.
        series: Series index per row, aligned with the flattened origin-major order.
        dropped: Rows excluded because a feature was unavailable at the origin.
    """

    target: str
    fold_id: str
    feature_set: str
    origins: np.ndarray
    series: np.ndarray
    dropped: int

    @property
    def n_rows(self) -> int:
        return int(self.origins.size)

    def identity(self) -> np.ndarray:
        """A hashable per-row identity, so two sets can be intersected exactly.

        Returns:
            ``[n_rows]`` int64 keys combining origin and series.
        """
        return self.origins.astype(np.int64) * 100_000 + self.series.astype(np.int64)

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "fold": self.fold_id,
            "feature_set": self.feature_set,
            "sample_count": self.n_rows,
            "unique_origins": int(np.unique(self.origins).size),
            "unique_series": int(np.unique(self.series).size),
            "first_origin": int(self.origins.min()) if self.n_rows else None,
            "last_origin": int(self.origins.max()) if self.n_rows else None,
            "dropped_rows": int(self.dropped),
            "timestamp_identity": (
                "origin_step x 100000 + series_index; intersected across feature sets so "
                "the comparison is made on shared rows"
            ),
        }


def common_timestamps(
    accounts: dict[str, SampleAccounting],
) -> tuple[np.ndarray, dict[str, object]]:
    """Intersect the usable rows of several feature sets.

    Larger feature sets read longer histories, so they legitimately have fewer usable origins.
    Scoring each set on whatever rows it has would compare a model on many rows against a
    model on few, which is a missing-row study wearing an ablation's clothes.

    Args:
        accounts: ``{feature_set_id: SampleAccounting}`` for one target and fold.

    Returns:
        ``(common_identity, report)``. The array is empty when the sets share no rows, which
        the caller must treat as a failure rather than as an empty comparison.
    """
    if not accounts:
        raise ValueError("no sample accounts were supplied")
    ids = list(accounts)
    identity = accounts[ids[0]].identity()
    for set_id in ids[1:]:
        identity = np.intersect1d(identity, accounts[set_id].identity(), assume_unique=False)
    per_set = {set_id: accounts[set_id].n_rows for set_id in ids}
    common = int(identity.size)
    report = {
        "per_set_rows": per_set,
        "common_rows": common,
        "smallest_set_rows": min(per_set.values()),
        "common_share_of_smallest": (
            common / min(per_set.values()) if min(per_set.values()) else 0.0
        ),
        "binding_set": min(per_set, key=lambda k: per_set[k]),
        "usable": bool(common > 0),
        "note": (
            "rows shared by every feature set. Where the share is below 1.0 the sets differ "
            "in usable history length, and the ablation must be read on the common rows"
        ),
    }
    return identity, report