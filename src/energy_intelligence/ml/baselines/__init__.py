"""Baseline model families for Phase 4.

Three families, in increasing order of what they assume:

``naive``       no fitting, deterministic reference
``classical``   scikit-learn tabular models, with a readable coefficient table
``neural``      one small torch network per architecture, as a control

All three implement the same two-method contract - ``predict`` and ``to_dict`` -
so the experiment runner treats them identically and adding a fourth family would
not require touching the runner.
"""

from __future__ import annotations

from .classical import CLASSICAL_MODELS, ClassicalModel, fit_classical, model_defaults
from .naive import NAIVE_MODELS, NaiveForecaster, fit_naive, naive_forecast
from .neural import NEURAL_MODELS, NeuralModel, fit_neural, model_defaults as neural_defaults

__all__ = [
    "CLASSICAL_MODELS",
    "NAIVE_MODELS",
    "NEURAL_MODELS",
    "ClassicalModel",
    "NaiveForecaster",
    "NeuralModel",
    "fit_classical",
    "fit_naive",
    "fit_neural",
    "naive_forecast",
    "classical_defaults",
    "naive_forecast_names",
    "neural_defaults",
]

#: Model families in the order the comparison table reports them.
MODEL_FAMILIES: tuple[str, ...] = ("naive", "classical", "neural")


def classical_defaults(name: str) -> dict[str, object]:
    """Hyperparameters for a classical baseline."""
    return model_defaults(name)


def naive_forecast_names() -> tuple[str, ...]:
    """Every naive rule name, including the optional drift rule."""
    return NAIVE_MODELS + ("naive_drift",)