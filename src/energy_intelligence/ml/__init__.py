"""Phase 4: ML-ready dataset construction and baseline forecasting.

Layers, and what each one is responsible for:

```text
targets.py     which quantities SMART-DS can actually forecast
splits.py      chronological train/validation/test boundaries
features.py    feature construction, each with an availability declaration
dataset.py     windows, schema, versioning, on-disk artifact
scaling.py     train-only preprocessing statistics
metrics.py     generic and energy-specific error measures
regimes.py     whether the data contains materially different regimes
baselines/     naive, classical ML, and one conventional neural baseline
registry.py    reproducible experiment record
experiment.py  one experiment, end to end
```

Every module is dataset-agnostic above :mod:`targets`. A future dataset supplies a
new target extractor and reuses the rest unchanged, which is the same rule Phase 3
established for its adapter layer.
"""

from __future__ import annotations

from .targets import (
    TARGET_REGISTRY,
    TargetSeries,
    TargetSpec,
    TargetStatus,
    load_target,
    feeder_load_target,
    select_customer_sample,
    solar_target,
    supported_targets,
    target_spec,
)

__all__ = [
    "TARGET_REGISTRY",
    "TargetSeries",
    "TargetSpec",
    "TargetStatus",
    "feeder_load_target",
    "load_target",
    "select_customer_sample",
    "solar_target",
    "supported_targets",
    "target_spec",
]