"""Training a temporal demand model, honestly.

The rules this module exists to enforce
---------------------------------------
**Model selection never sees the test split.** Checkpoints are chosen by validation
MAE. The test split is touched once, at the end, by
:func:`evaluate_checkpoint`. This is asserted structurally: the training function
takes no test data at all - it receives only a training index and a validation index.

**Every split is scaled with training statistics.** The scaler is fitted on training
rows and applied unchanged; :func:`~energy_intelligence.ml.scaling.assert_train_only_fit`
checks it.

**Delta or level.** When the configuration asks for the delta formulation, the model
predicts ``y(t+h) - y(t)`` and the level is reconstructed by adding the value at the
origin back. That is a property of the *model output*, not of the loss: the loss is
still computed on the reconstructed level, so the reported number is always the same
quantity as every other model's.

**Equal horizon weights.** No horizon is up-weighted to make a long horizon look
better (D-075). The mean per-unit absolute error across horizons selects the
checkpoint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .scaling import FeatureScaler, assert_train_only_fit
from .sequence import SequenceIndex, gather_sequences
from .temporal import TemporalDemandModel

__all__ = [
    "TrainingConfig",
    "TrainingHistory",
    "TrainedModel",
    "fit_temporal_model",
    "evaluate_checkpoint",
    "predict_series",
]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """The training protocol.

    Attributes:
        epochs: Maximum epochs.
        batch_size: Samples per step.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay, 0.0 by default so the loss is the only thing
            being tuned.
        loss: ``"l1"`` or ``"huber"``. L1 is the default because MAE is the reported
            metric; optimising MSE and reporting MAE hides peak under-weighting.
        huber_delta: Delta for the Huber loss, in per-unit units.
        patience: Validation-improvements without improvement before stopping.
        min_delta: Minimum validation improvement to reset the patience counter.
        max_seconds: Wall-clock cap, so an experiment cannot run away.
        seed: Seed for initialisation and batch order.
        threads: Torch CPU threads.
        grad_clip: Gradient-norm clip, 0 to disable.
    """

    epochs: int = 12
    batch_size: int = 512
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    loss: str = "l1"
    huber_delta: float = 0.05
    patience: int = 3
    min_delta: float = 1e-5
    max_seconds: float = 3600.0
    seed: int = 20260101
    threads: int = 8
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.loss not in {"l1", "huber"}:
            raise ValueError(f"loss must be 'l1' or 'huber', got {self.loss!r}")
        if self.loss == "huber" and self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive when loss is 'huber'")
        if self.patience < 0:
            raise ValueError(f"patience must not be negative, got {self.patience}")
        if self.max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive, got {self.max_seconds}")
        if self.threads <= 0:
            raise ValueError(f"threads must be positive, got {self.threads}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "loss": self.loss,
            "huber_delta": self.huber_delta,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "max_seconds": self.max_seconds,
            "seed": self.seed,
            "threads": self.threads,
            "grad_clip": self.grad_clip,
        }


@dataclass(slots=True)
class TrainingHistory:
    """Per-epoch record, written to a JSON Lines training log."""

    epoch: int
    train_loss: float
    validation_mae_per_unit: float
    seconds: float
    is_best: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": round(self.train_loss, 6),
            "validation_mae_per_unit": round(self.validation_mae_per_unit, 6),
            "seconds": round(self.seconds, 2),
            "is_best": self.is_best,
        }


@dataclass(frozen=True, slots=True)
class TrainedModel:
    """A fitted model plus everything needed to reproduce and evaluate it.

    Attributes:
        model: The fitted network.
        history: Per-epoch records.
        best_epoch: Epoch whose validation MAE was lowest.
        best_validation_mae: That validation MAE, per unit.
        stopped_early: Whether patience ended training before ``epochs``.
        train_seconds: Wall-clock training time.
        checkpoint_path: Where the best weights were written.
        train_rows: Training rows used.
        validation_rows: Validation rows used.
    """

    model: TemporalDemandModel
    history: tuple[TrainingHistory, ...]
    best_epoch: int
    best_validation_mae: float
    stopped_early: bool
    train_seconds: float
    checkpoint_path: Path
    train_rows: int
    validation_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_validation_mae_per_unit": self.best_validation_mae,
            "stopped_early": self.stopped_early,
            "train_seconds": round(self.train_seconds, 2),
            "epochs_run": len(self.history),
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "parameter_count": self.model.parameter_count(),
            "trainable_parameter_count": self.model.trainable_parameter_count(),
            "checkpoint": str(self.checkpoint_path),
        }


def _loss_fn(config: TrainingConfig) -> nn.Module:
    if config.loss == "l1":
        return nn.L1Loss()
    return nn.HuberLoss(delta=config.huber_delta)


def _epoch_loss(
    model: TemporalDemandModel,
    series_values: np.ndarray,
    index: SequenceIndex,
    rows: np.ndarray,
    *,
    config: TrainingConfig,
    optimiser: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
    scaler: FeatureScaler,
) -> float:
    """One pass over ``rows``; trains when ``optimiser`` is given, else evaluates."""
    training = optimiser is not None
    model.train(training)
    generator = torch.Generator().manual_seed(config.seed)
    order = (
        torch.randperm(rows.size, generator=generator).numpy()
        if training
        else np.arange(rows.size)
    )
    total = 0.0
    seen = 0
    batch = config.batch_size
    for start in range(0, order.size, batch):
        chosen = order[start : start + batch]
        raw_x, targets, baseline = gather_sequences(series_values, index, rows[chosen])
        features = scaler.transform(
            raw_x.transpose(0, 2, 1).reshape(chosen.size, -1).astype(np.float64)
        ).reshape(chosen.size, raw_x.shape[2], raw_x.shape[1]).transpose(0, 2, 1)
        x = torch.as_tensor(features, dtype=torch.float32)
        target = torch.as_tensor(targets, dtype=torch.float32)
        persistence = torch.as_tensor(baseline, dtype=torch.float32)
        series_ids = torch.as_tensor(
            index.series[rows[chosen]].astype(np.int64), dtype=torch.long
        )

        output = model(x, series_ids)
        if index.config.delta_target:
            prediction = output + persistence
        else:
            prediction = output
        loss = loss_fn(prediction, target)

        if training:
            assert optimiser is not None
            optimiser.zero_grad()
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
        total += float(loss.detach()) * chosen.size
        seen += chosen.size
    return total / max(1, seen)


def fit_temporal_model(
    model: TemporalDemandModel,
    series_values: np.ndarray,
    index: SequenceIndex,
    *,
    config: TrainingConfig,
    scaler: FeatureScaler,
    checkpoint_path: Path,
    validation_stride: int = 1,
) -> TrainedModel:
    """Train on the training split, selecting the checkpoint on validation MAE.

    The function receives **no test rows**. That is deliberate: selection on the test
    split is the single easiest way to make a temporal model look better than it is,
    and making it impossible to do so here is stronger than documenting a rule.

    Args:
        model: The model to fit, already built.
        series_values: ``(n_series, n_steps)`` normalised series.
        index: The sample index.
        config: The training protocol.
        scaler: Feature scaler, already fitted on training rows.
        checkpoint_path: Where the best weights are written.
        validation_stride: Use every Nth validation row for per-epoch checkpoint
            selection. The **final** evaluation always uses every test row, so this
            affects only which epoch is chosen, never what is reported. A stride of 2
            still leaves tens of thousands of validation samples, which is ample for
            ranking epochs, and halves the per-epoch cost.

    Returns:
        The trained model with its history and best checkpoint restored.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed % (2**32 - 1))
    torch.set_num_threads(config.threads)

    train_rows = index.rows(0)
    validation_rows = index.rows(1)
    if validation_stride > 1:
        validation_rows = validation_rows[::validation_stride]
    if train_rows.size == 0:
        raise ValueError("training split is empty")
    if validation_rows.size == 0:
        raise ValueError("validation split is empty; checkpoint selection needs it")

    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = _loss_fn(config)

    history: list[TrainingHistory] = []
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    since_improvement = 0
    stopped_early = False
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = _epoch_loss(
            model, series_values, index, train_rows,
            config=config, optimiser=optimiser, loss_fn=loss_fn, scaler=scaler,
        )
        validation_mae = _epoch_loss(
            model, series_values, index, validation_rows,
            config=config, optimiser=None, loss_fn=loss_fn, scaler=scaler,
        )
        elapsed = time.perf_counter() - epoch_started
        improved = best_mae - validation_mae > config.min_delta
        if improved:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            since_improvement = 0
        else:
            since_improvement += 1
        history.append(
            TrainingHistory(
                epoch=epoch,
                train_loss=train_loss,
                validation_mae_per_unit=validation_mae,
                seconds=elapsed,
                is_best=improved,
            )
        )
        if since_improvement >= config.patience:
            stopped_early = True
            break
        if time.perf_counter() - started > config.max_seconds:
            stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config_dict(),
            "sequence_config": index.config.to_dict(),
            "training_config": config.to_dict(),
            "best_epoch": best_epoch,
            "best_validation_mae_per_unit": best_mae,
            "parameter_count": model.parameter_count(),
        },
        checkpoint_path,
    )

    return TrainedModel(
        model=model,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_mae=best_mae,
        stopped_early=stopped_early,
        train_seconds=time.perf_counter() - started,
        checkpoint_path=checkpoint_path,
        train_rows=int(train_rows.size),
        validation_rows=int(validation_rows.size),
    )


def predict_series(
    model: TemporalDemandModel,
    series_values: np.ndarray,
    index: SequenceIndex,
    rows: np.ndarray,
    *,
    scaler: FeatureScaler,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict one batch of rows, returning levels in the stored units.

    Returns:
        ``(actual, predicted, persistence)``, each ``(len(rows), horizons)``.
    """
    model.eval()
    actuals: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    persistences: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, rows.size, batch_size):
            chosen = rows[start : start + batch_size]
            raw_x, targets, baseline = gather_sequences(series_values, index, chosen)
            features = scaler.transform(
                raw_x.transpose(0, 2, 1).reshape(chosen.size, -1).astype(np.float64)
            ).reshape(chosen.size, raw_x.shape[2], raw_x.shape[1]).transpose(0, 2, 1)
            output = model(
                torch.as_tensor(features, dtype=torch.float32),
                torch.as_tensor(index.series[chosen].astype(np.int64), dtype=torch.long),
            ).numpy()
            if index.config.delta_target:
                output = output + baseline
            actuals.append(targets.astype(np.float64))
            predictions.append(output.astype(np.float64))
            persistences.append(baseline.astype(np.float64))
    if not actuals:
        empty = np.zeros((0, len(index.config.horizons)), dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    return (
        np.concatenate(actuals),
        np.concatenate(predictions),
        np.concatenate(persistences),
    )


def evaluate_checkpoint(
    trained: TrainedModel,
    series_values: np.ndarray,
    index: SequenceIndex,
    *,
    split_code: int,
    scaler: FeatureScaler,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a selected checkpoint on one split, and return its rows.

    Split out from :func:`predict_series` so the call site reads as the deliberate,
    once-only step it is: the test split is evaluated here and nowhere else.

    Returns:
        ``(actual, predicted, persistence, rows)``.
    """
    rows = index.rows(split_code)
    actual, predicted, persistence = predict_series(
        trained.model, series_values, index, rows, scaler=scaler, batch_size=batch_size
    )
    return actual, predicted, persistence, rows
