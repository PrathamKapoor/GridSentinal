"""Training the router, without letting it see a future.

The loss is the routed forecast's own absolute error in kW. Not a classification
cross-entropy against "which expert was best", and not a regression onto oracle weights.
That choice is the phase's main methodological decision and it is worth stating why,
because the alternatives were not rejected on convenience.

*Why not classify the best expert.* A classification target discards the magnitude of
the win. If the GBM beats persistence by 0.0001 kW on a row and by 2 kW on the next,
both rows are labelled "GBM" and the router is told the two situations are equally
important. Worse, the achievable forecast from a label is "use that expert", which is
the *hard* routing upper bound - it throws away the possibility that a blend beats both.
Measured against that ceiling, a soft router looks bad for doing something useful.

*Why not regress oracle weights.* The oracle weight vector is a label computed from the
realised target. Regressing on it teaches the router to imitate hindsight, and the
hindsight weights are not what minimise expected error under uncertainty.

*Why the direct objective.* The question is "does routing produce a better forecast",
so the quantity being optimised is the forecast error. Weights are free to become
one-hot (hard routing) or fractional (hedging) exactly when that minimises error, and
nothing has to be argued about the gap between a surrogate label and the real goal.

The rows this trains on are the **validation** split, and the experts' predictions on
those rows are out-of-sample with respect to the experts' own fitting. Early stopping
uses the later part of that same split, so the router is never selected on rows it was
fitted on, and the test split is not touched at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import EnergyRouterNet, RouterArchitecture, build_router

__all__ = [
    "RouterTrainingConfig",
    "RouterHistory",
    "TrainedRouter",
    "fit_router",
    "load_router",
    "router_weights_for",
]


@dataclass(frozen=True, slots=True)
class RouterTrainingConfig:
    """The router's training protocol.

    Attributes:
        epochs: Maximum passes over the router-training rows.
        batch_size: Rows per step.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        temperature: Softmax temperature; part of the architecture, repeated here so the
            experiment record carries it next to the optimiser.
        seed: Seed for initialisation and batching.
        threads: Torch CPU threads.
        patience: Early-stopping patience in epochs.
        min_delta: Minimum validation improvement that counts as progress.
        max_seconds: Wall-clock cap.
        train_fraction_of_validation: Share of the validation split, chronologically,
            that fits the router. The remainder is the early-stopping set. Leaving both
            out and early stopping on the rows being fitted would let the router pick an
            epoch that merely memorises them.
    """

    epochs: int = 30
    batch_size: int = 4096
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    temperature: float = 1.0
    seed: int = 20260101
    threads: int = 8
    patience: int = 5
    min_delta: float = 1e-6
    max_seconds: float = 900.0
    train_fraction_of_validation: float = 0.7

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must not be negative")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.threads <= 0:
            raise ValueError("threads must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must not be negative")
        if not 0.0 < self.train_fraction_of_validation < 1.0:
            raise ValueError("train_fraction_of_validation must lie in (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "temperature": self.temperature,
            "seed": self.seed,
            "threads": self.threads,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "max_seconds": self.max_seconds,
            "train_fraction_of_validation": self.train_fraction_of_validation,
        }


@dataclass(frozen=True, slots=True)
class RouterHistory:
    """One epoch's outcome, as written to the training log."""

    epoch: int
    train_loss: float
    validation_mae_kw: float
    validation_mae_kw_by_horizon: dict[int, float]
    seconds: float
    is_best: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "validation_mae_kw": self.validation_mae_kw,
            "validation_mae_kw_by_horizon": {
                str(h): v for h, v in self.validation_mae_kw_by_horizon.items()
            },
            "seconds": self.seconds,
            "is_best": self.is_best,
        }


@dataclass(frozen=True, slots=True)
class TrainedRouter:
    """A fitted router and everything needed to reproduce its predictions.

    Attributes:
        net: The router.
        scaler: Feature scaler fitted on router-training rows only.
        feature_names: The columns fed to the net.
        history: Per-epoch record.
        best_epoch: Epoch with the lowest early-stopping MAE.
        best_validation_mae_kw: That MAE, averaged over horizons.
        stopped_early: Whether patience ran out.
        train_seconds: Total training wall time.
        train_rows: Rows used to fit.
        validation_rows: Rows used to select the epoch.
        checkpoint_path: Where the router was saved.
        architecture: The shape, repeated for the record.
    """

    net: EnergyRouterNet
    scaler: Any
    feature_names: tuple[str, ...]
    history: tuple[RouterHistory, ...]
    best_epoch: int
    best_validation_mae_kw: float
    stopped_early: bool
    train_seconds: float
    train_rows: int
    validation_rows: int
    checkpoint_path: Path | None
    architecture: dict[str, Any]

    @property
    def parameter_count(self) -> int:
        return self.net.parameter_count()

    @property
    def best_validation_mae_kw_by_horizon(self) -> dict[int, float]:
        """Per-horizon MAE on the selection rows, at the selected epoch."""
        for item in self.history:
            if item.epoch == self.best_epoch:
                return dict(item.validation_mae_kw_by_horizon)
        if self.history:  # pragma: no cover - best_epoch always names a history entry
            return dict(self.history[-1].validation_mae_kw_by_horizon)
        return {}  # pragma: no cover

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "parameters": self.parameter_count,
            "best_epoch": self.best_epoch,
            "best_validation_mae_kw": self.best_validation_mae_kw,
            "stopped_early": self.stopped_early,
            "train_seconds": round(self.train_seconds, 3),
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "epochs_run": len(self.history),
            "router_features": list(self.feature_names),
            "history": [item.to_dict() for item in self.history],
        }


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.abs(np.asarray(actual, np.float64) - np.asarray(predicted, np.float64)).mean())


def router_weights_for(
    trained: TrainedRouter, features: np.ndarray
) -> np.ndarray:
    """``[N, horizons, experts]`` weights from a trained router, without the pool."""
    from .model import simplex_weights

    scaled = trained.scaler.transform(np.asarray(features, dtype=np.float64))
    tensor = torch.from_numpy(np.ascontiguousarray(scaled)).to(torch.float32)
    with torch.no_grad():
        logits = trained.net.logits(tensor)
        weights = simplex_weights(logits, temperature=trained.net.architecture.temperature)
    return weights.numpy().astype(np.float64)


def fit_router(
    *,
    features: np.ndarray,
    expert_forecast_kw: np.ndarray,
    actual_kw: np.ndarray,
    horizons: tuple[int, ...],
    expert_names: tuple[str, ...],
    feature_names: tuple[str, ...],
    config: RouterTrainingConfig,
    architecture: Any,
    initial_weights: np.ndarray | None = None,
    checkpoint_path: Path | None = None,
    log: Any = None,
) -> TrainedRouter:
    """Fit the router on the routed forecast's own absolute error.

    Args:
        features: ``[N, n_features]`` origin-observable features. Rows are consumed in
            the order given; the caller is responsible for having passed only rows whose
            experts predicted out-of-sample.
        expert_forecast_kw: ``[n_experts, N, horizons]`` in kW.
        actual_kw: ``[N, horizons]`` in kW.
        horizons: Horizon steps.
        expert_names: Expert names, for the record.
        feature_names: Column names of ``features``, one per column. Stored in the
            checkpoint so a reload cannot silently reorder the inputs.
        config: Training protocol.
        architecture: Router shape.
        initial_weights: ``[horizons, experts]`` simplex weights to start from, typically
            the fixed ensemble fitted on these same rows. The pre-training state is scored
            as epoch 0 and is eligible to be selected, so the returned router is never
            worse than where it started on the selection rows.
        checkpoint_path: Where to save. Omit to skip saving.
        log: Optional progress callable.

    Returns:
        The trained router, restored to its best epoch.

    Raises:
        ValueError: If the shapes disagree or there are too few rows to split.
    """
    from ..scaling import FeatureScaler

    torch.set_num_threads(int(config.threads))
    features = np.asarray(features, dtype=np.float64)
    expert_forecast_kw = np.asarray(expert_forecast_kw, dtype=np.float64)
    actual_kw = np.asarray(actual_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)
    n_experts, n_rows, n_horizons = expert_forecast_kw.shape
    if features.ndim != 2 or features.shape[0] != n_rows:
        raise ValueError(
            f"features {features.shape} do not match {n_rows} expert-forecast rows"
        )
    if len(feature_names) != features.shape[1]:
        raise ValueError(
            f"{len(feature_names)} feature names for {features.shape[1]} columns"
        )
    if actual_kw.shape != (n_rows, n_horizons) or n_horizons != len(horizons):
        raise ValueError(
            f"shapes disagree: forecasts {expert_forecast_kw.shape}, "
            f"actual {actual_kw.shape}, horizons {horizons}"
        )
    n_train = int(round(n_rows * config.train_fraction_of_validation))
    n_train = max(1, min(n_train, n_rows - 1))
    train_rows = np.arange(n_train)
    stop_rows = np.arange(n_train, n_rows)
    if train_rows.size < 2 or stop_rows.size < 1:
        raise ValueError(
            f"need at least 2 rows to fit and 1 to select on, got {n_rows}"
        )

    scaler = FeatureScaler.fit(features[train_rows])
    scaled = scaler.transform(features).astype(np.float32)
    # Constant columns must be zeroed, not divided by their own standard deviation: the
    # scaler maps a constant to 0, and a torch input containing NaN would silently make
    # every weight NaN.
    if not np.isfinite(scaled).all():
        raise ValueError("scaled router features contain non-finite values")

    forecasts = torch.from_numpy(np.ascontiguousarray(expert_forecast_kw)).to(torch.float32)
    truth = torch.from_numpy(np.ascontiguousarray(actual_kw)).to(torch.float32)
    inputs = torch.from_numpy(np.ascontiguousarray(scaled)).to(torch.float32)

    net = build_router(
        n_features=scaled.shape[1],
        horizons=horizons,
        n_experts=n_experts,
        architecture=architecture,
        seed=config.seed,
        initial_weights=initial_weights,
    )
    optimiser = torch.optim.Adam(
        net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(int(config.seed) + 1)

    history: list[RouterHistory] = []
    started = time.perf_counter()
    stopped_early = False

    def score_stop_rows() -> tuple[float, dict[int, float]]:
        net.eval()
        with torch.no_grad():
            weights = net(inputs[stop_rows])
            mixed = torch.einsum("nhe,enh->nh", weights, forecasts[:, stop_rows, :])
            residual = (mixed - truth[stop_rows]).abs()
            by_horizon = {
                horizon: float(residual[:, column].mean())
                for column, horizon in enumerate(horizons)
            }
            return float(residual.mean()), by_horizon

    # Epoch 0 is the initialisation. Scoring it makes the router a strict improvement on
    # its own starting point on the selection rows, which is the property that lets the
    # warm start be safe rather than a gamble.
    epoch_zero_score, epoch_zero_by_horizon = score_stop_rows()
    best_score = epoch_zero_score
    best_epoch = 0
    best_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
    history.append(
        RouterHistory(
            epoch=0,
            train_loss=float("nan"),
            validation_mae_kw=epoch_zero_score,
            validation_mae_kw_by_horizon=epoch_zero_by_horizon,
            seconds=0.0,
            is_best=True,
        )
    )
    stale = 0
    if log is not None:
        log(f"    epoch   0 initial   stop {epoch_zero_score:.5f} *")

    for epoch in range(1, int(config.epochs) + 1):
        net.train()
        epoch_started = time.perf_counter()
        order = torch.randperm(train_rows.size, generator=generator).numpy()
        total_loss = 0.0
        for start in range(0, train_rows.size, int(config.batch_size)):
            batch = train_rows[order[start : start + int(config.batch_size)]]
            if batch.size == 0:
                continue
            optimiser.zero_grad(set_to_none=True)
            weights = net(inputs[batch])
            mixed = torch.einsum("nhe,enh->nh", weights, forecasts[:, batch, :])
            loss = (mixed - truth[batch]).abs().mean()
            loss.backward()
            optimiser.step()
            total_loss += float(loss.detach()) * batch.size
        train_loss = total_loss / max(1, train_rows.size)

        score, by_horizon = score_stop_rows()
        improved = score < best_score - float(config.min_delta)
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in net.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        history.append(
            RouterHistory(
                epoch=epoch,
                train_loss=train_loss,
                validation_mae_kw=score,
                validation_mae_kw_by_horizon=by_horizon,
                seconds=time.perf_counter() - epoch_started,
                is_best=improved,
            )
        )
        if log is not None:
            log(
                f"    epoch {epoch:>3d} train {train_loss:.5f} stop {score:.5f} "
                f"{'*' if improved else ''} ({history[-1].seconds:.1f}s)"
            )
        if stale >= int(config.patience):
            stopped_early = True
            break
        if time.perf_counter() - started > float(config.max_seconds):
            stopped_early = True
            if log is not None:
                log("    wall-clock cap reached")
            break

    net.load_state_dict(best_state)
    net.eval()

    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": net.state_dict(),
                "config": net.config_dict(),
                "feature_names": list(feature_names),
                "training_config": config.to_dict(),
                "expert_names": list(expert_names),
                "best_epoch": best_epoch,
                "best_validation_mae_kw": best_score,
                "scaler": scaler.to_dict(),
                "seed": config.seed,
            },
            checkpoint_path,
        )

    return TrainedRouter(
        net=net,
        scaler=scaler,
        feature_names=tuple(feature_names),
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_mae_kw=best_score,
        stopped_early=stopped_early,
        train_seconds=time.perf_counter() - started,
        train_rows=int(train_rows.size),
        validation_rows=int(stop_rows.size),
        checkpoint_path=checkpoint_path,
        architecture=net.config_dict(),
    )


def load_router(checkpoint_path: Path) -> tuple[EnergyRouterNet, Any, tuple[str, ...], dict[str, Any]]:
    """Reload a saved router: the net, its feature scaler, the column names, and metadata.

    The names come back with the net on purpose. A checkpoint whose feature order no
    longer matches its training would otherwise reload "successfully" and produce
    confident nonsense, which is the single most damaging failure mode a routing system
    has.

    Args:
        checkpoint_path: A file written by :func:`fit_router`.

    Returns:
        ``(net, scaler, feature_names, metadata)``.

    Raises:
        ValueError: If the checkpoint does not describe a router.
    """
    from ..scaling import FeatureScaler

    payload = torch.load(Path(checkpoint_path), weights_only=False)
    for key in ("state_dict", "config", "feature_names", "scaler"):
        if key not in payload:
            raise ValueError(f"{checkpoint_path} is not a router checkpoint: no {key!r}")
    config = payload["config"]
    net = build_router(
        n_features=int(config["n_features"]),
        horizons=tuple(int(h) for h in config["horizons"]),
        n_experts=int(config["n_experts"]),
        architecture=RouterArchitecture.from_dict(config["architecture"]),
        seed=int(payload.get("seed", 20260101)),
    )
    net.load_state_dict(payload["state_dict"])
    net.eval()
    metadata = {
        key: payload[key]
        for key in ("best_epoch", "best_validation_mae_kw", "training_config", "expert_names")
        if key in payload
    }
    return net, FeatureScaler.from_dict(payload["scaler"]), tuple(payload["feature_names"]), metadata
