"""Phase 10: controlled feature ablation.

Which predefined feature families actually contribute predictive information when the model
configuration is held fixed? Everything else - model, hyperparameters, training procedure,
scaling, seed, horizon, folds, metrics - is held fixed by construction, so the only thing
that varies between two rows of the ablation table is the feature set.

The package is split so the rules are testable without running anything:

``sets``       the five nested feature sets, built from the existing catalogue
``folds``      F01-F06, and the guard that denies the final test split
``freeze``     the protocol frozen before official execution
``runner``     one fit-and-score cell, with timestamp-level evidence retained
``selection``  F01-F04 selection, the frozen tie-break, and incremental effects
"""

from __future__ import annotations

from .folds import (
    CONFIRMATION_FOLDS,
    SELECTION_FOLDS,
    FinalTestAccessError,
    Fold,
    SampleAccounting,
    ablation_folds,
    assert_final_test_denied,
    common_timestamps,
)
from .freeze import (
    FREEZE_PATH,
    SELECTION_FREEZE_PATH,
    FinalTestPolicy,
    Phase10Protocol,
    build_protocol,
    load_freeze,
    protocol_hash,
    write_protocol_freeze,
    write_selection_freeze,
)
from .runner import FoldResult, TimestampRows, primary_model_for, run_one
from .sets import FEATURE_SETS, FEATURE_SET_IDS, SET_ORDER, WEATHER_FEATURES, checksum, describe

__all__ = [
    "CONFIRMATION_FOLDS",
    "SELECTION_FOLDS",
    "FEATURE_SETS",
    "FEATURE_SET_IDS",
    "FREEZE_PATH",
    "FinalTestAccessError",
    "FinalTestPolicy",
    "Fold",
    "FoldResult",
    "Phase10Protocol",
    "SELECTION_FREEZE_PATH",
    "SET_ORDER",
    "SampleAccounting",
    "TimestampRows",
    "WEATHER_FEATURES",
    "ablation_folds",
    "assert_final_test_denied",
    "build_protocol",
    "checksum",
    "common_timestamps",
    "describe",
    "load_freeze",
    "primary_model_for",
    "protocol_hash",
    "run_one",
    "write_protocol_freeze",
    "write_selection_freeze",
]
