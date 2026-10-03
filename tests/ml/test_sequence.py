"""Tests for ordered temporal window construction.

Temporal correctness is the entire value of this module, so the tests here are
written against explicit index arithmetic rather than against the builder's own
output: if the offsets were wrong in a self-consistent way, comparing the builder to
itself would still pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import build_series

from energy_intelligence.ml.sequence import (
    SEQUENCE_CHANNELS,
    SequenceConfig,
    assert_sequences_are_causal,
    build_sequence_index,
    gather_sequences,
    sequence_version,
)


def _index(config: SequenceConfig | None = None, *, series: int = 3, steps: int = 20000):
    resolved = config or SequenceConfig()
    return build_sequence_index(
        series_count=series,
        total_steps=steps,
        config=resolved,
        target_id="customer_load",
        value_origin="DERIVED",
        series_ids=tuple(f"s{i}" for i in range(series)),
        source="synthetic/for-tests",
    )


# ======================================================================
# Configuration
# ======================================================================


def test_horizons_from_a_config_file_are_normalised_to_a_tuple() -> None:
    """TOML hands over a list; the contract is a tuple, so it is normalised once here.

    Without this, an ablation declared ``horizons = [1]`` fails validation while
    ``horizons = (1,)`` passes - a difference that has nothing to do with the data.
    """
    from_list = SequenceConfig(horizons=[1, 4])       # type: ignore[arg-type]
    from_tuple = SequenceConfig(horizons=(1, 4))
    assert from_list.horizons == from_tuple.horizons
    assert isinstance(from_list.horizons, tuple)


def test_token_count_is_ceiling_of_lookback_over_stride() -> None:
    assert SequenceConfig(lookback_steps=672, token_stride=4).token_count == 168
    assert SequenceConfig(lookback_steps=96, token_stride=4).token_count == 24
    # A lookback that is not a multiple of the stride still rounds up, and the origin
    # is always the last token.
    assert SequenceConfig(lookback_steps=100, token_stride=4).token_count == 25


def test_config_rejects_impossible_shapes() -> None:
    with pytest.raises(ValueError, match="lookback_steps must be positive"):
        SequenceConfig(lookback_steps=0)
    with pytest.raises(ValueError, match="token_stride must be positive"):
        SequenceConfig(token_stride=0)
    with pytest.raises(ValueError, match="must be shorter than lookback_steps"):
        SequenceConfig(lookback_steps=96, token_stride=96)
    with pytest.raises(ValueError, match="sorted and unique"):
        SequenceConfig(horizons=(4, 1))
    with pytest.raises(ValueError, match="at least one horizon"):
        SequenceConfig(horizons=())
    with pytest.raises(ValueError, match="train_origin_stride must be positive"):
        SequenceConfig(train_origin_stride=0)


def test_cyclic_channels_are_a_group() -> None:
    with_cyclic = SequenceConfig(cyclic_channels=True)
    without = SequenceConfig(cyclic_channels=False)
    assert "hour_sin" in with_cyclic.channel_names
    assert "hour_sin" not in without.channel_names
    assert without.channel_count == with_cyclic.channel_count - 4
    # The signal channels survive.
    assert "demand" in without.channel_names and "delta_1" in without.channel_names


def test_every_channel_declares_a_formula_and_a_reason() -> None:
    for channel in SEQUENCE_CHANNELS:
        assert channel.formula and channel.reason


def test_version_changes_with_every_shape_decision() -> None:
    ids = ("s0", "s1")
    base = sequence_version(SequenceConfig(), "customer_load", ids)
    variants = [
        sequence_version(SequenceConfig(lookback_steps=96), "customer_load", ids),
        sequence_version(SequenceConfig(token_stride=1), "customer_load", ids),
        sequence_version(SequenceConfig(horizons=(1, 4)), "customer_load", ids),
        sequence_version(SequenceConfig(train_origin_stride=16), "customer_load", ids),
        sequence_version(SequenceConfig(cyclic_channels=False), "customer_load", ids),
        sequence_version(SequenceConfig(delta_target=False), "customer_load", ids),
        sequence_version(SequenceConfig(), "feeder_load", ids),
    ]
    assert len({base, *variants}) == len(variants) + 1


# ======================================================================
# Index construction
# ======================================================================


def test_index_expands_one_row_per_series_and_origin() -> None:
    index = _index(SequenceConfig(train_origin_stride=4), series=3, steps=20000)
    assert index.origins.size == index.series.size == index.split.size
    assert index.sample_count % 3 == 0
    # Rows are stored series-major: every origin repeated once per series.
    first_block = index.origins[:3]
    assert len(set(first_block.tolist())) == 1
    assert index.series[:3].tolist() == [0, 1, 2]


def test_evaluation_splits_use_every_origin_while_training_is_strided() -> None:
    """This is what makes the Phase 5 test numbers comparable to Phase 4's."""
    config = SequenceConfig(train_origin_stride=4, lookback_steps=672, horizons=(1, 4, 96))
    index = _index(config, series=2, steps=35040)
    bounds = index.boundaries.to_dict()
    train_span = bounds["train"][1] - bounds["train"][0]
    expected_train = (train_span - 672 - 96) // 4
    assert index.rows(0).size == expected_train * 2
    # Validation and test are NOT strided.
    assert index.rows(1).size == (bounds["validation"][1] - bounds["validation"][0] - 672 - 96) * 2
    assert index.rows(2).size == (bounds["test"][1] - bounds["test"][0] - 672 - 96) * 2


def test_no_window_reaches_before_its_split() -> None:
    index = _index(SequenceConfig(), series=2, steps=35040)
    bounds = index.boundaries.to_dict()
    for code, name in ((0, "train"), (1, "validation"), (2, "test")):
        rows = index.rows(code)
        assert rows.size, name
        origins = index.origins[rows]
        lo, hi = bounds[name]
        assert origins.min() - index.config.lookback_steps >= lo
        assert origins.max() + index.config.max_horizon < hi


def test_infeasible_axis_is_refused_with_a_reason() -> None:
    with pytest.raises(ValueError, match="no origin survives"):
        _index(SequenceConfig(lookback_steps=672, horizons=(96,)), series=1, steps=700)


# ======================================================================
# Window contents
# ======================================================================


def test_window_is_chronological_and_ends_at_the_origin() -> None:
    series = build_series(series_count=2, steps=20000)
    values = series.normalised().astype(np.float32)
    config = SequenceConfig(lookback_steps=96, token_stride=4, train_origin_stride=64)
    index = _index(config, series=2, steps=20000)
    row = int(index.rows(1)[5])
    x, targets, baseline = gather_sequences(values, index, np.asarray([row]))

    origin = int(index.origins[row])
    series_row = int(index.series[row])
    # token_count tokens ending at the origin: offsets -(T-1)*stride .. 0 step stride
    first_offset = -(config.token_count - 1) * config.token_stride
    expected = values[series_row, origin + first_offset : origin + 1 : config.token_stride]
    assert x[0, 0].tolist() == pytest.approx(expected.tolist())
    assert expected.size == config.token_count
    # The last token IS the origin: the persistence baseline is its value.
    assert baseline[0, 0] == pytest.approx(float(values[series_row, origin]), rel=1e-6)
    assert x[0, 0, -1] == pytest.approx(float(values[series_row, origin]), rel=1e-6)


def test_targets_are_strictly_after_the_origin() -> None:
    series = build_series(series_count=2, steps=20000)
    values = series.normalised().astype(np.float32)
    config = SequenceConfig(horizons=(1, 4, 96), train_origin_stride=64)
    index = _index(config, series=2, steps=20000)
    row = int(index.rows(1)[7])
    _, targets, baseline = gather_sequences(values, index, np.asarray([row]))
    origin = int(index.origins[row])
    series_row = int(index.series[row])
    for position, horizon in enumerate(config.horizons):
        assert targets[0, position] == pytest.approx(
            float(values[series_row, origin + horizon]), rel=1e-6
        )
        assert baseline[0, position] == pytest.approx(float(values[series_row, origin]), rel=1e-6)


def test_delta_channel_is_the_first_difference() -> None:
    series = build_series(series_count=2, steps=20000)
    values = series.normalised().astype(np.float32)
    config = SequenceConfig(lookback_steps=96, token_stride=4, cyclic_channels=False,
                            train_origin_stride=64)
    index = _index(config, series=2, steps=20000)
    x, _, _ = gather_sequences(values, index, np.asarray([int(index.rows(1)[0])]))
    demand = x[0, 0]
    delta = x[0, 1]
    assert delta[1:].tolist() == pytest.approx(np.diff(demand).tolist(), rel=1e-5)
    assert delta[0] == pytest.approx(0.0, abs=1e-6), "the first token has no predecessor"


def test_cyclic_channels_use_the_native_clock() -> None:
    series = build_series(series_count=1, steps=20000)
    values = series.normalised().astype(np.float32)
    config = SequenceConfig(lookback_steps=96, token_stride=4, train_origin_stride=64)
    index = _index(config, series=1, steps=20000)
    row = int(index.rows(1)[3])
    x, _, _ = gather_sequences(values, index, np.asarray([row]))
    origin = int(index.origins[row])
    positions = config.channel_names.index("hour_sin")
    for token in range(config.token_count):
        absolute_step = origin - (config.token_count - 1 - token) * config.token_stride
        hour = (absolute_step % 96) * 15 / 60
        assert x[0, positions, token] == pytest.approx(np.sin(2 * np.pi * hour / 24), abs=1e-5)
        assert x[0, positions + 1, token] == pytest.approx(np.cos(2 * np.pi * hour / 24), abs=1e-5)


def test_channel_count_matches_the_configuration() -> None:
    series = build_series(series_count=1, steps=20000)
    values = series.normalised().astype(np.float32)
    for cyclic in (True, False):
        config = SequenceConfig(cyclic_channels=cyclic, train_origin_stride=64)
        index = _index(config, series=1, steps=20000)
        x, _, _ = gather_sequences(values, index, np.asarray([int(index.rows(1)[0])]))
        assert x.shape == (1, config.channel_count, config.token_count)


def test_gather_refuses_a_window_before_the_record() -> None:
    """A window that would read before index 0 is refused, not clamped.

    ``build_sequence_index`` already prevents this, so the guard is exercised with a
    hand-built index whose origins are too early for its window - which is exactly
    the situation the check exists for.
    """
    from energy_intelligence.ml.sequence import SequenceIndex

    series = build_series(series_count=1, steps=20000)
    values = series.normalised().astype(np.float32)
    config = SequenceConfig(lookback_steps=96, token_stride=4, train_origin_stride=64)
    bad = SequenceIndex(
        origins=np.asarray([10], dtype=np.int64),
        series=np.asarray([0], dtype=np.int64),
        split=np.asarray([1], dtype=np.int8),
        series_ids=("s0",),
        config=config,
        boundaries=_index(config, series=1, steps=20000).boundaries,
        target_id="customer_load",
        value_origin="DERIVED",
        source="synthetic",
        version="sq-bad",
    )
    with pytest.raises(ValueError, match="reaches before the start"):
        gather_sequences(values, bad, np.asarray([0]))


# ======================================================================
# Causality
# ======================================================================


def test_causality_guard_passes_on_real_shaped_data() -> None:
    series = build_series(series_count=2, steps=20000)
    values = series.normalised().astype(np.float32)
    index = _index(SequenceConfig(train_origin_stride=64), series=2, steps=20000)
    assert_sequences_are_causal(values, index, index.rows(2)[:32])


def test_causality_guard_catches_a_leaking_window() -> None:
    """The guard must fail on a window that includes its own future.

    Built by hand rather than by corrupting the module, so the test proves the guard
    detects leakage rather than proving the builder is currently correct.
    """
    series = build_series(series_count=2, steps=20000)
    values = series.normalised().astype(np.float32)
    index = _index(SequenceConfig(train_origin_stride=64), series=2, steps=20000)
    rows = index.rows(2)[:16]

    # A deliberately leaky builder that reads one step past the origin.
    import energy_intelligence.ml.sequence as sequence_module

    original = sequence_module._window_offsets

    def leaky_offsets(config):
        offsets = original(config)
        return np.append(offsets, 1)  # one step into the future

    sequence_module._window_offsets = leaky_offsets
    try:
        # The guard refuses the forward offset before it even gathers, naming the
        # violation.
        with pytest.raises(AssertionError, match="past the origin"):
            assert_sequences_are_causal(values, index, rows)
    finally:
        sequence_module._window_offsets = original


def test_causality_guard_handles_an_empty_selection() -> None:
    series = build_series(series_count=1, steps=20000)
    values = series.normalised().astype(np.float32)
    index = _index(SequenceConfig(train_origin_stride=64), series=1, steps=20000)
    assert_sequences_are_causal(values, index, np.asarray([], dtype=np.int64))


def test_causality_guard_rejects_a_forward_window_even_before_gathering() -> None:
    series = build_series(series_count=1, steps=20000)
    values = series.normalised().astype(np.float32)
    index = _index(SequenceConfig(train_origin_stride=64), series=1, steps=20000)
    import energy_intelligence.ml.sequence as sequence_module

    original = sequence_module._window_offsets
    sequence_module._window_offsets = lambda config: np.asarray([0, 1, 2])
    try:
        with pytest.raises(AssertionError, match="past the origin"):
            assert_sequences_are_causal(values, index, index.rows(2)[:4])
    finally:
        sequence_module._window_offsets = original


# ======================================================================
# Manifest
# ======================================================================


def test_manifest_records_the_whole_contract() -> None:
    index = _index(SequenceConfig(train_origin_stride=4), series=2, steps=35040)
    manifest = index.to_manifest()
    assert manifest["schema_version"].startswith("1.")
    assert manifest["dataset_version"] == index.version
    assert manifest["sequence_config"]["lookback_steps"] == 672
    assert manifest["temporal_contract"]["resampling"].startswith("NONE")
    assert "input window only" in manifest["temporal_contract"]["input_subsampling"]
    assert manifest["split"]["evaluation_origin_stride"] == 1
    assert manifest["leakage_policy"]
    assert len(manifest["channel_definitions"]) == index.config.channel_count
    for definition in manifest["channel_definitions"]:
        assert definition["formula"] and definition["reason"]


def test_summary_exposes_splits_and_shape() -> None:
    index = _index(SequenceConfig(train_origin_stride=4), series=2, steps=35040)
    summary = index.summary()
    assert summary["sample_count"] == index.sample_count
    assert summary["token_count"] == index.config.token_count
    assert set(summary["splits"]) == {"train", "validation", "test"}
    assert summary["value_origin"] == "DERIVED"
