"""The one conventional neural baseline, and nothing more.

Scope discipline
----------------
This phase explicitly does **not** build the Energy World Model, does not touch
Qwen3, and does not create experts or routing. The only neural network here is a
standard baseline whose only job is to answer one question:

> Does a conventional neural network materially outperform the classical baselines
> on *this* data?

Two architectures, both conventional and both small:

``mlp``
    A fully connected network over the same tabular features the classical models
    use. This is the fair comparison: same inputs, different model class.

``gru``
    A single-layer GRU over the raw lookback window of the target series. This is
    the conventional sequence baseline for time-series forecasting, and it answers
    a different question: does *sequence structure* add anything over hand-chosen
    lag features?

Both are trained with a fixed seed, on training rows only, with the loss chosen as
plain L1 because MAE is the reported metric - optimising MSE and reporting MAE
would hide a systematic under-weighting of peaks.

Deliberately absent: attention, ensembles, learning-rate schedules, early stopping
on a test-visible signal, and any architecture search. A baseline that is tuned
until it wins is no longer a baseline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

__all__ = ["NeuralModel", "NEURAL_MODELS", "fit_neural", "model_defaults"]

#: Reported in this order.
NEURAL_MODELS: tuple[str, ...] = ("neural_mlp", "neural_gru")


def model_defaults(name: str) -> dict[str, Any]:
    """Hyperparameters for a named neural baseline."""
    if name == "neural_mlp":
        return {
            "hidden_sizes": [64, 32],
            "activation": "relu",
            "epochs": 40,
            "batch_size": 512,
            "learning_rate": 0.01,
            "loss": "l1",
            "weight_decay": 0.0,
            "note": (
                "two small hidden layers, L1 loss to match the reported metric, "
                "fixed epoch count so training time is comparable across runs"
            ),
        }
    if name == "neural_gru":
        return {
            "hidden_size": 32,
            "num_layers": 1,
            "epochs": 40,
            "batch_size": 512,
            "learning_rate": 0.01,
            "loss": "l1",
            "weight_decay": 0.0,
            "note": "single unidirectional GRU layer over the raw lookback window",
        }
    raise KeyError(f"unknown neural model {name!r}; available: {list(NEURAL_MODELS)}")


@dataclass(slots=True)
class NeuralModel:
    """A fitted neural baseline.

    Attributes:
        name: Model name.
        module: The torch module.
        architecture: Architecture label, e.g. ``"mlp"`` or ``"gru"``.
        params: Hyperparameters used.
        seed: Seed the initialisation used.
        fit_seconds: Wall-clock training duration.
        epochs_run: Epochs actually completed.
        final_train_loss: Loss on the final epoch's training batches.
        input_shape: Shape the model consumes.
    """

    name: str
    module: Any
    architecture: str
    params: dict[str, Any]
    seed: int
    fit_seconds: float = 0.0
    epochs_run: int = 0
    final_train_loss: float | None = None
    input_shape: tuple[int, ...] = ()
    history: list[float] = field(default_factory=list)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict in eval mode with no gradient, deterministically."""
        self.module.eval()
        tensor = torch.as_tensor(
            np.asarray(features, dtype=np.float32), dtype=torch.float32
        )
        with torch.no_grad():
            output = self.module(tensor)
        return np.asarray(output.detach().numpy(), dtype=np.float64).reshape(-1)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.module.parameters()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "params": self.params,
            "seed": self.seed,
            "fit_seconds": round(self.fit_seconds, 3),
            "epochs_run": self.epochs_run,
            "final_train_loss": self.final_train_loss,
            "parameter_count": self.parameter_count(),
            "input_shape": list(self.input_shape),
        }


def fit_neural(
    name: str,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    threads: int = 4,
) -> NeuralModel:
    """Fit one neural baseline on training rows.

    Args:
        name: One of :data:`NEURAL_MODELS`.
        features: For ``mlp``, the tabular feature matrix. For ``gru``, the
            ``(n_samples, lookback)`` lookback window matrix.
        targets: ``(n_samples,)`` target column.
        seed: Seed for weight initialisation and shuffling.
        threads: Torch CPU thread count, recorded because it changes runtime.

    Returns:
        The fitted model.

    Raises:
        ValueError: If the inputs are empty or misaligned.
    """
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.set_num_threads(max(1, threads))

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float64).reshape(-1, 1).astype(np.float32)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"features and targets disagree: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] == 0:
        raise ValueError("cannot fit a neural model on zero rows")

    params = model_defaults(name)
    started = time.perf_counter()

    if name == "neural_mlp":
        layers: list[nn.Module] = []
        width = x.shape[1]
        for hidden in params["hidden_sizes"]:
            layers.append(nn.Linear(width, hidden))
            layers.append(nn.ReLU())
            width = hidden
        layers.append(nn.Linear(width, 1))
        module = nn.Sequential(*layers)
        architecture = "mlp"
        input_shape = (x.shape[1],)
    elif name == "neural_gru":
        module = _GRUForecaster(hidden_size=params["hidden_size"], num_layers=params["num_layers"])
        architecture = "gru"
        input_shape = (x.shape[1],)
    else:
        raise KeyError(f"unknown neural model {name!r}; available: {list(NEURAL_MODELS)}")

    optimiser = torch.optim.Adam(
        module.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    loss_fn = nn.L1Loss()

    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    rows = x_tensor.shape[0]
    batch = min(int(params["batch_size"]), rows)
    history: list[float] = []

    module.train()
    for _ in range(int(params["epochs"])):
        order = torch.randperm(rows, generator=generator)
        total = 0.0
        batches = 0
        for start in range(0, rows, batch):
            index = order[start : start + batch]
            optimiser.zero_grad()
            prediction = module(x_tensor[index])
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            loss = loss_fn(prediction, y_tensor[index])
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
            batches += 1
        history.append(total / max(1, batches))

    model = NeuralModel(
        name=name,
        module=module,
        architecture=architecture,
        params=params,
        seed=seed,
        epochs_run=int(params["epochs"]),
        final_train_loss=history[-1] if history else None,
        input_shape=input_shape,
        history=history,
    )
    model.fit_seconds = time.perf_counter() - started
    return model


class _GRUForecaster(nn.Module):
    """Single-layer GRU that maps a lookback window to one value.

    Kept in this module rather than the project domain because it is a *baseline
    model*, not a domain concept: nothing in ``energy_intelligence.domain`` may
    depend on a neural network.
    """

    def __init__(self, *, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``(batch, lookback)``; treated as ``(batch, lookback, 1)``."""
        sequence = x.unsqueeze(-1)
        output, _ = self.gru(sequence)
        return self.head(output[:, -1, :])