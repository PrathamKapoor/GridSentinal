"""Read the dataset selection from ``configs/data.toml``.

A two-line module, extracted so the ML pipeline never has to know where the
dataset config lives or how it is loaded: it asks for the selection and receives a
validated object.
"""

from __future__ import annotations

from ..config.data import DataConfig, load_data_config

__all__ = ["dataset_config_from"]


def dataset_config_from(path: str | None = None) -> DataConfig:
    """Load ``configs/data.toml`` (or an explicit path)."""
    return load_data_config(path)