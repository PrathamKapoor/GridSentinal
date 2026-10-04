"""The energy prediction head on frozen Qwen features, and what it can be asked.

The head is deliberately the simplest thing that can solve the task: a linear map
from a 2048-dimensional frozen activation to three per-unit deltas. That simplicity
is the point. A probe with 6k parameters either beats the established baselines or it
does not, and if it does, the credit cannot plausibly belong to a 1.7 B-parameter
backbone that never changed.

Two things are worth stating plainly about what this configuration can and cannot
show.

**It can show** that Qwen's activations are linearly informative about future demand,
and by comparing against a randomly initialised backbone of identical architecture it
can show whether that informativeness comes from *pretraining*.

**It cannot show** that Qwen is a good energy model. A frozen backbone is a fixed
feature map; the question of whether adapting the backbone itself helps is a different
experiment, and on this hardware it is measured and reported rather than assumed -
see the compute probe in the handoff.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

__all__ = [
    "FeatureBasis",
    "ProbeHead",
    "ProbeTrainingConfig",
    "TrainedProbe",
    "fit_probe",
    "probe_predict",
]


class FeatureBasis:
    """An optional PCA compression of the frozen features, fitted on train only.

    Why this exists: 2048 features and roughly 6k head parameters is enough capacity
    to memorise 10,000 training rows, which would make a weak representation look
    strong. Compressing to a few hundred components fitted **on training rows only**
    removes that failure mode.

    Why it is safe: the components are fitted on the training split and then applied
    unchanged to validation and test. Fitting on anything else would be leakage, and
    :meth:`fit` refuses to see a split argument at all so the mistake has nowhere to
    happen.
    """

    def __init__(self, width: int) -> None:
        self.width = int(width)
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "FeatureBasis":
        """Fit on training features only. ``[N, D]`` in, basis fitted in place."""
        if features.ndim != 2:
            raise ValueError(f"expected [N, D], got {features.shape}")
        width = min(self.width, features.shape[0] - 1, features.shape[1])
        if width < 1:
            raise ValueError("not enough rows to fit a basis")
        # torch's low-rank SVD is the same decomposition numpy would do, and it is
        # already a dependency; no new package for one operation.
        matrix = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        mean = matrix.mean(dim=0, keepdim=True)
        centred = matrix - mean
        _u, singular, v = torch.linalg.svd(centred, full_matrices=False)
        components = v[:width]
        variance = (singular**2) / max(1.0, float((centred**2).sum()))
        self.mean_ = mean.numpy()[0]
        self.components_ = components.numpy()
        self.explained_variance_ratio_ = variance[:width].numpy()
        self.width = width
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("basis is not fitted")
        centred = features - self.mean_
        return centred @ self.components_.T

    @property
    def cumulative_variance(self) -> float:
        if self.explained_variance_ratio_ is None:
            return float("nan")
        return float(self.explained_variance_ratio_.sum())

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "cumulative_explained_variance": self.cumulative_variance,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean_,
            components=self.components_,
            explained=self.explained_variance_ratio_,
        )

    @classmethod
    def load(cls, path: Path) -> "FeatureBasis":
        payload = np.load(path, allow_pickle=False)
        basis = cls(int(payload["components"].shape[0]))
        basis.mean_ = payload["mean"]
        basis.components_ = payload["components"]
        basis.explained_variance_ratio_ = payload["explained"]
        return basis


class ProbeHead(nn.Module):
    """A small MLP mapping frozen features to per-unit horizon deltas.

    The **per-series embedding is not decoration**. Every input is per-unit, so the
    magnitude that distinguishes a 1.4 kW household from a 210 kW shop has been
    deliberately divided out - which is correct for the metric and fatal for a probe
    with no other way to know which customer it is looking at. Phase 5's TCN carried an
    explicit ``series_embedding_dim = 8``; without the same here, Qwen would be
    handicapped against a baseline that was not, and the comparison would measure the
    handicap rather than the representation.

    Attributes:
        layers: Feature widths, e.g. ``[0]`` for a bare linear probe or ``[0, 128]``
            for one hidden layer.
        horizons: The forecast horizons, so the output width matches the task.
        series_count: Number of series, or 0 to disable the embedding.
        series_embedding_dim: Width of the per-series embedding.
        network: The constructed layers.
        output: The final linear map to ``len(horizons)`` deltas.
        series_embedding: The learned per-series embedding, if enabled.
    """

    def __init__(
        self,
        in_features: int,
        horizons: tuple[int, ...],
        *,
        layers: tuple[int, ...] = (0,),
        series_count: int = 0,
        series_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.horizons = tuple(horizons)
        self.in_features = int(in_features)
        self.series_embedding_dim = int(series_embedding_dim) if series_count else 0
        self.series_embedding = (
            nn.Embedding(int(series_count), self.series_embedding_dim)
            if series_count
            else None
        )
        total = self.in_features + self.series_embedding_dim
        modules: list[nn.Module] = []
        width = total
        for hidden in layers:
            if hidden <= 0:
                continue
            modules.append(nn.Linear(width, hidden))
            modules.append(nn.ReLU())
            width = hidden
        self.network = nn.Sequential(*modules)
        self.output = nn.Linear(width, len(self.horizons))

    def forward(
        self, features: torch.Tensor, series: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``[N, in_features]`` to ``[N, horizons]`` per-unit deltas.

        Args:
            features: The frozen (and optionally PCA-compressed) activations.
            series: Row's series index. Required when the head has an embedding.
        """
        if features.dim() != 2:
            raise ValueError(f"expected [N, features], got {tuple(features.shape)}")
        if features.shape[1] != self.in_features:
            raise ValueError(
                f"expected {self.in_features} features, got {features.shape[1]}"
            )
        if self.series_embedding is not None:
            if series is None:
                raise ValueError(
                    "this head carries a per-series embedding, so the series index is "
                    "required; omitting it would silently drop the identity signal"
                )
            embedded = self.series_embedding(series.long())
            if embedded.shape[0] != features.shape[0]:
                raise ValueError("series index and features disagree on row count")
            features = torch.cat([features, embedded], dim=1)
        return self.output(self.network(features))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass(frozen=True, slots=True)
class ProbeTrainingConfig:
    """Head training settings. All selection happens on validation.

    Attributes:
        epochs: Maximum epochs.
        batch_size: Rows per step.
        learning_rate: AdamW step size.
        weight_decay: AdamW weight decay.
        patience: Early-stopping patience in epochs.
        min_delta: Minimum validation improvement that resets patience.
        max_seconds: Compute guard; stops training and keeps the best checkpoint.
        torch_threads: Torch CPU threads.
        seed: Seed for initialisation and shuffling.
    """

    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 0.003
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 1e-6
    max_seconds: float = 1800.0
    torch_threads: int = 16
    seed: int = 20260101


@dataclass(frozen=True, slots=True)
class TrainedProbe:
    """A trained head plus everything needed to reproduce and audit it.

    Attributes:
        head: The trained module.
        best_epoch: The epoch whose validation score was best.
        best_validation_mae_per_unit: Mean per-unit MAE across horizons at that epoch.
        epochs_run: Epochs actually executed.
        history: Per-epoch records.
        train_seconds: Wall time.
        stopped_early: Whether patience or the compute guard ended training.
        parameter_count: Head parameters.
        checkpoint_path: Where the head was saved.
        basis: The fitted basis, or None when features are used raw.
    """

    head: ProbeHead
    best_epoch: int
    best_validation_mae_per_unit: float
    epochs_run: int
    history: tuple[dict[str, Any], ...]
    train_seconds: float
    stopped_early: bool
    parameter_count: int
    checkpoint_path: Path
    basis: FeatureBasis | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_validation_mae_per_unit": self.best_validation_mae_per_unit,
            "epochs_run": self.epochs_run,
            "train_seconds": round(self.train_seconds, 2),
            "stopped_early": self.stopped_early,
            "parameter_count": self.parameter_count,
            "basis": self.basis.to_dict() if self.basis is not None else None,
            "history": list(self.history),
        }


def _to_deltas(targets: np.ndarray, persistence: np.ndarray) -> np.ndarray:
    """Target as a change from the value at the origin, in per-unit terms."""
    return targets - persistence


def fit_probe(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_persistence: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    validation_persistence: np.ndarray,
    *,
    train_series_index: np.ndarray | None = None,
    validation_series_index: np.ndarray | None = None,
    horizons: tuple[int, ...],
    config: ProbeTrainingConfig,
    layers: tuple[int, ...] = (0,),
    basis_width: int | None = 256,
    series_count: int = 0,
    series_embedding_dim: int = 8,
    checkpoint_path: Path | None = None,
    log: Any = None,
) -> TrainedProbe:
    """Train the head, selecting the epoch on validation per-unit MAE.

    Args:
        train_features: ``[N, D]`` frozen features for the training rows.
        train_targets: ``[N, horizons]`` per-unit target levels.
        train_persistence: ``[N, horizons]`` per-unit ``y(t)`` at each horizon.
        validation_features: Validation features.
        validation_targets: Validation targets.
        validation_persistence: Validation persistence levels.
        horizons: The horizons, for output width and reporting order.
        config: Training settings.
        layers: Head widths; ``[0]`` is a linear probe.
        basis_width: PCA width, or None to use raw features.
        series_count: Series count for the per-series embedding; 0 disables it.
        series_embedding_dim: Width of that embedding.
        checkpoint_path: Where to save the head.
        log: Optional progress callable.

    Returns:
        The trained probe, with the best-validation weights restored.
    """
    if config.epochs < 1:
        # Without this, an epochs=0 run returns a randomly initialised head with
        # best_epoch=0 and no checkpoint state, and the caller would go on to report
        # its predictions as a measured result.
        raise ValueError(
            f"epochs must be at least 1, got {config.epochs}; a head that never trains "
            "cannot be evaluated"
        )
    if train_features.shape[0] == 0 or validation_features.shape[0] == 0:
        raise ValueError(
            "the probe needs both training and validation rows; got "
            f"{train_features.shape[0]} and {validation_features.shape[0]}"
        )

    torch.set_num_threads(config.torch_threads)
    generator = torch.Generator().manual_seed(config.seed)
    torch.manual_seed(config.seed)

    basis: FeatureBasis | None = None
    if basis_width:
        basis = FeatureBasis(basis_width).fit(train_features)
        train_features = basis.transform(train_features)
        validation_features = basis.transform(validation_features)

    head = ProbeHead(
        train_features.shape[1],
        horizons,
        layers=layers,
        series_count=series_count,
        series_embedding_dim=series_embedding_dim,
    )
    optimiser = torch.optim.AdamW(
        head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    train_x = torch.from_numpy(np.ascontiguousarray(train_features, dtype=np.float32))
    train_y = torch.from_numpy(
        np.ascontiguousarray(_to_deltas(train_targets, train_persistence), dtype=np.float32)
    )
    validation_x = torch.from_numpy(
        np.ascontiguousarray(validation_features, dtype=np.float32)
    )
    validation_deltas = _to_deltas(validation_targets, validation_persistence)
    # The head may or may not carry a series embedding; the training loop must work
    # either way, so both are always defined and None means "no embedding".
    train_series: torch.Tensor | None = None
    validation_series: torch.Tensor | None = None
    if series_count:
        if train_series_index is None or validation_series_index is None:
            raise ValueError(
                "a per-series embedding is configured, so both series index arrays "
                "are required; refusing to train an identity-blind head by accident"
            )
        train_series = torch.from_numpy(
            np.ascontiguousarray(train_series_index, dtype=np.int64)
        )
        validation_series = torch.from_numpy(
            np.ascontiguousarray(validation_series_index, dtype=np.int64)
        )

    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    since_improvement = 0
    history: list[dict[str, Any]] = []
    stopped_early = False
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        order = torch.randperm(train_x.shape[0], generator=generator)
        total_loss = 0.0
        batches = 0
        for start in range(0, order.numel(), config.batch_size):
            rows = order[start : start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            prediction = head(
                train_x[rows], None if train_series is None else train_series[rows]
            )
            loss = torch.nn.functional.l1_loss(prediction, train_y[rows])
            loss.backward()
            optimiser.step()
            total_loss += float(loss.detach())
            batches += 1

        with torch.no_grad():
            predicted = head(validation_x, validation_series).numpy()
        score = float(np.mean(np.abs(predicted - validation_deltas)))

        improved = score < best_score - config.min_delta
        if improved:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            best_epoch = epoch
            since_improvement = 0
        else:
            since_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, batches),
                "validation_mae_per_unit": score,
                "is_best": improved,
                "seconds": round(time.perf_counter() - started, 2),
            }
        )
        if log is not None:
            log(
                f"  epoch {epoch:>3d} train {total_loss / max(1, batches):.6f} "
                f"val {score:.6f}{' *' if improved else ''}"
            )

        if since_improvement >= config.patience:
            stopped_early = True
            break
        if time.perf_counter() - started > config.max_seconds:
            stopped_early = True
            break

    if best_state is not None:
        head.load_state_dict(best_state)

    resolved = checkpoint_path or Path("artifacts/phase6/probe.pt")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "in_features": head.in_features,
            "horizons": list(head.horizons),
            "layers": list(layers),
            "series_count": series_count,
            "series_embedding_dim": series_embedding_dim,
            "basis": (
                {"mean": basis.mean_, "components": basis.components_}
                if basis is not None
                else None
            ),
            "best_epoch": best_epoch,
            "best_validation_mae_per_unit": best_score,
            "seed": config.seed,
        },
        resolved,
    )

    return TrainedProbe(
        head=head,
        best_epoch=best_epoch,
        best_validation_mae_per_unit=best_score,
        epochs_run=len(history),
        history=tuple(history),
        train_seconds=time.perf_counter() - started,
        stopped_early=stopped_early,
        parameter_count=head.parameter_count(),
        checkpoint_path=resolved,
        basis=basis,
    )


@torch.no_grad()
def probe_predict(
    head: ProbeHead,
    features: np.ndarray,
    *,
    basis: FeatureBasis | None = None,
    series: np.ndarray | None = None,
) -> np.ndarray:
    """Per-unit horizon deltas from frozen features.

    Args:
        head: The trained head.
        features: ``[N, D]`` frozen activations.
        basis: The fitted PCA basis, if the head was trained on compressed features.
        series: Row's series index. Required when the head carries a per-series
            embedding, which the shipped configuration does.

    Raises:
        ValueError: The head needs a series index and none was given. Failing loudly
            matters here: dropping it would leave a trained embedding unused and
            produce confidently wrong predictions with no visible cause.
    """
    head.eval()
    matrix = features if basis is None else basis.transform(features)
    tensor = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32))
    if head.series_embedding is None:
        return head(tensor).numpy()
    if series is None:
        raise ValueError(
            "this head carries a per-series embedding, so the series index is required"
        )
    index = torch.from_numpy(np.ascontiguousarray(series, dtype=np.int64))
    return head(tensor, index).numpy()
