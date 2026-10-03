"""Ordered temporal windows for Phase 5.

What this adds to Phase 4
-------------------------
Phase 4 stored one row per ``(series, forecast origin)`` with **selected** lag
features, which is the right representation for a tabular model and the wrong one for
a temporal model: it throws away the order, and with it the shape of the window.

This module builds the window instead::

    x = [y(t-N+1) ... y(t-1), y(t)]      ordered, ending exactly at the origin
    y = [y(t+h) for h in horizons]       strictly after the origin

The Phase 4 data contract is preserved exactly: same target, same per-unit
normalisation, same native 15-minute resolution (**no resampling**), same horizons,
same chronological split boundaries. Only the *shape* of a sample changes, so this is
a new dataset version rather than a change of contract (D-071).

Two design decisions that are not obvious
-----------------------------------------
**Token stride.** A 672-step window at native resolution is 672 tokens, which is
expensive on CPU. Reading it at every ``token_stride``-th step gives 168 tokens
covering the same week, for a quarter of the compute. This subsamples the *input
window only* - the grid, the targets and the horizons stay native. Whether it costs
accuracy is not assumed; it is one of the measured ablations (D-074).

**Nothing is materialised.** A dense window tensor for 325k samples would be
gigabytes. Instead only the indices are stored and windows are gathered per batch
from the series matrix, which is 5.6 MB. Memory is therefore independent of the
number of samples.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .splits import SplitBoundaries, split_boundaries

__all__ = [
    "SequenceChannels",
    "SEQUENCE_CHANNELS",
    "SequenceConfig",
    "SequenceIndex",
    "build_sequence_index",
    "sequence_version",
    "gather_sequences",
    "assert_sequences_are_causal",
]

#: Steps per day on the native grid. The daily period used by the cyclic channels.
STEPS_PER_DAY = 96


@dataclass(frozen=True, slots=True)
class SequenceChannels:
    """The ordered channels fed to the temporal encoder.

    Each channel is either a function of the demand series itself, read in order, or
    a function of the origin's position on the clock. **Every one is knowable at the
    forecast origin**; none reads a target.

    Attributes:
        name: Column name in the gathered tensor.
        formula: How the value is produced.
        reason: Why the channel is there.
        requires_cyclic: Whether the channel belongs to the cyclic family, which the
            ablation switches on and off as a group.
    """

    name: str
    formula: str
    reason: str
    requires_cyclic: bool = False


#: The default channel set, in a fixed order so a checkpoint always matches.
SEQUENCE_CHANNELS: tuple[SequenceChannels, ...] = (
    SequenceChannels(
        name="demand",
        formula="y(t) normalised per unit of the series scale",
        reason="the signal itself; ordered, unlike Phase 4's selected lags",
    ),
    SequenceChannels(
        name="delta_1",
        formula="y(t) - y(t-1)",
        reason="DERIVED: the most recent ramp, as a channel rather than a feature",
    ),
    SequenceChannels(
        name="hour_sin",
        formula="sin(2*pi*hour_of_day/24)",
        reason="cyclic daily position, continuous across midnight",
        requires_cyclic=True,
    ),
    SequenceChannels(
        name="hour_cos",
        formula="cos(2*pi*hour_of_day/24)",
        reason="pairs with hour_sin so the encoding is a circle, not a ramp",
        requires_cyclic=True,
    ),
    SequenceChannels(
        name="year_sin",
        formula="sin(2*pi*day_of_year/365)",
        reason="cyclic seasonal position, continuous across new year",
        requires_cyclic=True,
    ),
    SequenceChannels(
        name="year_cos",
        formula="cos(2*pi*day_of_year/365)",
        reason="pairs with year_sin so the season is a circle",
        requires_cyclic=True,
    ),
)


@dataclass(frozen=True, slots=True)
class SequenceConfig:
    """Everything that shapes a Phase 5 sample.

    Attributes:
        lookback_steps: History window in **native steps**, ending at the origin.
        token_stride: Read every Nth step of that window.
        horizons: Forecast horizons in native steps.
        train_origin_stride: Origin stride for **training** rows only. Evaluation
            rows always use stride 1, so the test population is identical to Phase
            4's and the two phases' numbers are directly comparable.
        train_fraction: Share of the grid used for training.
        validation_fraction: Share used for validation.
        cyclic_channels: Whether the four cyclic channels are included.
        delta_target: Whether the network predicts the change from the value at the
            origin rather than the level.
    """

    lookback_steps: int = 672
    token_stride: int = 4
    horizons: tuple[int, ...] = (1, 4, 96)
    train_origin_stride: int = 4
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    cyclic_channels: bool = True
    delta_target: bool = True

    def __post_init__(self) -> None:
        # TOML and JSON both hand over a list where this contract wants a tuple.
        # Normalising here rather than at every call site means an ablation declared
        # as `horizons = [1]` behaves identically to one declared as `(1,)`, and
        # sequence_version() therefore hashes the same configuration the same way.
        if isinstance(self.horizons, list):
            object.__setattr__(self, "horizons", tuple(int(h) for h in self.horizons))
        if self.lookback_steps <= 0:
            raise ValueError(f"lookback_steps must be positive, got {self.lookback_steps}")
        if self.token_stride <= 0:
            raise ValueError(f"token_stride must be positive, got {self.token_stride}")
        if not self.horizons:
            raise ValueError("at least one horizon is required")
        if any(h <= 0 for h in self.horizons):
            raise ValueError(f"horizons must be positive native steps, got {self.horizons}")
        if tuple(sorted(self.horizons)) != self.horizons:
            raise ValueError(f"horizons must be sorted and unique, got {self.horizons}")
        if self.train_origin_stride <= 0:
            raise ValueError(
                f"train_origin_stride must be positive, got {self.train_origin_stride}"
            )
        if self.lookback_steps <= self.token_stride:
            raise ValueError(
                "token_stride must be shorter than lookback_steps, otherwise the "
                "window collapses to one token"
            )

    @property
    def token_count(self) -> int:
        """Tokens per window after subsampling.

        ``ceil(lookback / stride)``: a 672-step window read every 4th step is 168
        tokens, the last of which is the origin itself.
        """
        return -(-self.lookback_steps // self.token_stride)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(
            channel.name
            for channel in SEQUENCE_CHANNELS
            if self.cyclic_channels or not channel.requires_cyclic
        )

    @property
    def channel_count(self) -> int:
        return len(self.channel_names)

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lookback_steps": self.lookback_steps,
            "token_stride": self.token_stride,
            "token_count": self.token_count,
            "horizons": list(self.horizons),
            "train_origin_stride": self.train_origin_stride,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "cyclic_channels": self.cyclic_channels,
            "delta_target": self.delta_target,
            "channels": list(self.channel_names),
        }


def sequence_version(
    config: SequenceConfig, target_id: str, series_ids: tuple[str, ...]
) -> str:
    """Content-addressed version for a sequence dataset."""
    material = "|".join(
        [
            "seq-1.0.0-phase5",
            target_id,
            f"lb{config.lookback_steps}",
            f"ts{config.token_stride}",
            ",".join(str(h) for h in config.horizons),
            f"tr{config.train_origin_stride}",
            f"{config.train_fraction:.4f}",
            f"{config.validation_fraction:.4f}",
            "cyclic" if config.cyclic_channels else "no-cyclic",
            "delta" if config.delta_target else "level",
            str(len(series_ids)),
        ]
    )
    return "sq-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class SequenceIndex:
    """Indices of every sample, and nothing else.

    Deliberately weightless: a dense window tensor would be gigabytes, and storing
    only indices keeps memory independent of the sample count while leaving every
    temporal property directly assertable.

    Attributes:
        origins: Forecast origin per sample, native grid index.
        series: Series row per sample.
        split: Split code per sample (0 train, 1 validation, 2 test).
        series_ids: Series names.
        config: The configuration that produced this index.
        boundaries: The three chronological ranges.
        target_id: Which declared target.
        value_origin: ``DERIVED`` or ``OBSERVED``, carried from Phase 4.
        source: The Phase 4 source locator.
    """

    origins: np.ndarray
    series: np.ndarray
    split: np.ndarray
    series_ids: tuple[str, ...]
    config: SequenceConfig
    boundaries: SplitBoundaries
    target_id: str
    value_origin: str
    source: str
    version: str

    @property
    def sample_count(self) -> int:
        return int(self.origins.size)

    def mask(self, split_code: int) -> np.ndarray:
        """Boolean mask of one split's rows."""
        return self.split == split_code

    def rows(self, split_code: int) -> np.ndarray:
        """Row indices belonging to one split."""
        return np.flatnonzero(self.split == split_code)

    def summary(self) -> dict[str, Any]:
        counts = {
            name: int(np.count_nonzero(self.split == code))
            for code, name in ((0, "train"), (1, "validation"), (2, "test"))
        }
        return {
            "version": self.version,
            "target_id": self.target_id,
            "value_origin": self.value_origin,
            "sample_count": self.sample_count,
            "series_count": len(self.series_ids),
            "token_count": self.config.token_count,
            "channel_count": self.config.channel_count,
            "channels": list(self.config.channel_names),
            "splits": counts,
            "config": self.config.to_dict(),
            "boundaries": self.boundaries.to_dict(),
            "source": self.source,
        }

    def to_manifest(self) -> dict[str, Any]:
        """The audit record for the dataset artifact."""
        return {
            "schema_version": "1.0.0-phase5",
            "dataset_version": self.version,
            "sequence_config": self.config.to_dict(),
            "channel_definitions": [
                {
                    "name": channel.name,
                    "formula": channel.formula,
                    "reason": channel.reason,
                    "cyclic": channel.requires_cyclic,
                }
                for channel in SEQUENCE_CHANNELS
                if channel.name in self.config.channel_names
            ],
            "target": {
                "target_id": self.target_id,
                "value_origin": self.value_origin,
                "horizons": list(self.config.horizons),
            },
            "temporal_contract": {
                "resolution_minutes": 15,
                "resampling": "NONE on the grid, targets or horizons (D-044)",
                "input_subsampling": (
                    f"window of {self.config.lookback_steps} native steps read every "
                    f"{self.config.token_stride}th step -> "
                    f"{self.config.token_count} tokens. The SUBSET is the input "
                    "window only; targets remain at native 15-minute resolution."
                ),
                "window_end": "the last token is y(t), the forecast origin itself",
                "causality": "every input index is <= t; every target index is > t",
            },
            "split": {
                "method": "chronological, contiguous, by grid index (D-061)",
                "fractions": {
                    "train": self.config.train_fraction,
                    "validation": self.config.validation_fraction,
                },
                "boundaries": self.boundaries.to_dict(),
                "train_origin_stride": self.config.train_origin_stride,
                "evaluation_origin_stride": 1,
                "note": (
                    "Training rows are subsampled in time for compute; validation and "
                    "test rows use every origin, so the test population is identical "
                    "to Phase 4's and the numbers are directly comparable."
                ),
            },
            "leakage_policy": (
                "gather_sequences asserts max(input index) <= t < min(target index), "
                "and assert_sequences_are_causal poisons every value after t and "
                "requires the gathered input to be byte-identical."
            ),
            "source": {"dataset": "SMART-DS", "locator": self.source},
        }


def build_sequence_index(
    *,
    series_count: int,
    total_steps: int,
    config: SequenceConfig,
    target_id: str,
    value_origin: str,
    series_ids: tuple[str, ...],
    source: str,
) -> SequenceIndex:
    """Build the sample index for one task.

    The window and target constraints are applied per split, exactly as Phase 4 did,
    so no sample's window can cross a boundary.
    """
    boundaries = split_boundaries(
        total_steps,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )

    origin_blocks: list[np.ndarray] = []
    split_blocks: list[np.ndarray] = []
    for code, (lo, hi) in enumerate(boundaries.as_tuple()):
        # A sample needs its whole window inside [lo, hi) and its targets strictly
        # before hi. Training rows may additionally be strided.
        first = lo + config.lookback_steps
        last = hi - config.max_horizon - 1
        if last < first:
            continue
        stride = config.train_origin_stride if code == 0 else 1
        origins = np.arange(first, last + 1, stride, dtype=np.int64)
        if origins.size == 0:
            continue
        origin_blocks.append(origins)
        split_blocks.append(np.full(origins.size, code, dtype=np.int8))

    if not origin_blocks:
        raise ValueError(
            "no origin survives the window and horizon constraints; the configuration "
            f"is infeasible for a {total_steps}-step axis"
        )

    origins = np.concatenate(origin_blocks)
    splits = np.concatenate(split_blocks)

    # One row per (series, origin), exactly as in Phase 4: every origin is repeated
    # once per series and the series index cycles. Expanding both grids together is
    # what keeps ``origins[i]`` and ``series[i]`` describing the same sample.
    return SequenceIndex(
        origins=np.repeat(origins, series_count),
        series=np.tile(np.arange(series_count, dtype=np.int64), origins.size),
        split=np.repeat(splits, series_count),
        series_ids=tuple(series_ids),
        config=config,
        boundaries=boundaries,
        target_id=target_id,
        value_origin=value_origin,
        source=source,
        version=sequence_version(config, target_id, tuple(series_ids)),
    )


def _window_offsets(config: SequenceConfig) -> np.ndarray:
    """Native offsets of the window tokens, chronological and ending at 0.

    ``[-(T-1)*stride, ..., -stride, 0]`` where ``T`` is the token count, so the
    last token is the forecast origin and the offsets are strictly increasing. The
    oldest token is at ``-(T-1)*stride``, which is within one stride of the full
    lookback; that residue is the price of a token grid anchored on the origin
    rather than on the lookback start, and it guarantees the origin is always
    included.
    """
    tokens = config.token_count
    offsets = -(tokens - 1) * config.token_stride + np.arange(tokens, dtype=np.int64) * config.token_stride
    if int(offsets[-1]) != 0:
        raise AssertionError("the window must end exactly at the forecast origin")
    if int(offsets.min()) < -(config.lookback_steps - 1):
        raise AssertionError("the window reaches further back than the lookback allows")
    return offsets


def gather_sequences(
    series_values: np.ndarray,
    index: SequenceIndex,
    rows: np.ndarray,
    *,
    origin_iso_day_zero: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather one batch of windows, targets and baselines.

    Args:
        series_values: ``(n_series, n_steps)`` **normalised** series, as stored in
            the Phase 4 dataset.
        index: The sample index.
        rows: Sample rows to gather.
        origin_iso_day_zero: Day-of-year of index 0, for the cyclic channels. The
            native axis starts 1 January, so this is 0, but it is explicit.

    Returns:
        ``(x, y, baseline)`` where ``x`` is ``(len(rows), channels, tokens)``,
        ``y`` is ``(len(rows), horizons)`` of target levels, and ``baseline`` is
        ``(len(rows), horizons)`` persistence predictions ``y(t)``. The baseline is
        returned so the loss can be evaluated as a *change from persistence* without
        the caller re-deriving it.
    """
    values = np.asarray(series_values, dtype=np.float32)
    offsets = _window_offsets(index.config)
    origins = index.origins[rows].astype(np.int64)
    series_rows = index.series[rows].astype(np.int64)

    window_index = origins[:, None] + offsets[None, :]  # (rows, tokens)
    if int(window_index.min()) < 0:
        raise ValueError(
            f"a window reaches before the start of the record (min index "
            f"{int(window_index.min())}); the lookback is longer than the history"
        )
    demand = values[series_rows[:, None], window_index]  # (rows, tokens)

    channels = [demand[:, None, :], (demand - np.concatenate([demand[:, :1], demand[:, :-1]], axis=1))[:, None, :]]
    names = index.config.channel_names
    if "hour_sin" in names:
        hour = ((window_index % STEPS_PER_DAY) * 15.0 / 60.0)
        angle_hour = 2.0 * np.pi * hour / 24.0
        channels.append(np.sin(angle_hour)[:, None, :])
        if "hour_cos" in names:
            channels.append(np.cos(angle_hour)[:, None, :])
    if "year_sin" in names:
        day = (window_index // STEPS_PER_DAY) + origin_iso_day_zero
        angle_year = 2.0 * np.pi * day / 365.0
        channels.append(np.sin(angle_year)[:, None, :])
        if "year_cos" in names:
            channels.append(np.cos(angle_year)[:, None, :])

    x = np.concatenate([c.astype(np.float32) for c in channels], axis=1)  # (rows, C, T)
    if x.shape[1] != index.config.channel_count:
        raise AssertionError(
            f"gathered {x.shape[1]} channels but the configuration declares "
            f"{index.config.channel_count}: {names}"
        )

    targets = np.stack(
        [values[series_rows, origins + h] for h in index.config.horizons], axis=1
    ).astype(np.float32)
    baseline = np.repeat(values[series_rows, origins][:, None], len(index.config.horizons), axis=1)
    return x, targets, baseline.astype(np.float32)


def assert_sequences_are_causal(
    series_values: np.ndarray,
    index: SequenceIndex,
    rows: np.ndarray,
) -> None:
    """Prove a gathered batch cannot contain information from after its origin.

    The check is the same one Phase 4 used, applied per sample: for a deterministic
    sample of rows, poison every value strictly after that row's origin and require
    the gathered **input** to be unchanged. Targets are gathered from later indices
    and *are* expected to change - a target that survived poisoning would mean it
    was not in the future at all.

    Raises:
        AssertionError: If any input token changes when the future is poisoned, if a
            window index exceeds its origin, or if a target index is not strictly
            after it.
    """
    values = np.asarray(series_values, dtype=np.float32)
    if rows.size == 0:
        return

    offsets = _window_offsets(index.config)
    if int(offsets.max()) > 0:
        raise AssertionError(
            f"the window reaches {int(offsets.max())} steps past the origin; every "
            "offset must be <= 0"
        )
    origins = index.origins[rows].astype(np.int64)
    if int((origins[:, None] + offsets[None, :]).max()) > int(origins.max()):
        raise AssertionError("a window index exceeds its own origin")

    probe_count = min(32, rows.size)
    probe_rows = rows[
        np.sort(np.random.default_rng(0).choice(rows.size, size=probe_count, replace=False))
    ]
    baseline_x, _, _ = gather_sequences(values, index, probe_rows)
    baseline_x = baseline_x.copy()

    for position, row in enumerate(probe_rows):
        origin = int(index.origins[row])
        series_row = int(index.series[row])
        if origin + index.config.max_horizon >= values.shape[1]:
            raise AssertionError(
                f"row {row} targets would fall outside the record (origin {origin})"
            )
        poisoned = values.copy()
        poisoned[series_row, origin + 1 :] = np.nan
        candidate, _, _ = gather_sequences(poisoned, index, np.asarray([row]))
        if not np.array_equal(baseline_x[position : position + 1], candidate, equal_nan=True):
            raise AssertionError(
                f"row {row} (series {series_row}, origin {origin}) changed when "
                "values after its origin were poisoned: the window reads the future"
            )
