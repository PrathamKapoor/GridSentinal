"""Connect the experiment configuration to real data and the runner.

This is the bridge between :class:`~energy_intelligence.config.ml.MlConfig` and
:func:`~energy_intelligence.ml.runner.run_phase4`. It exists as its own module so
the CLI stays a thin argument parser and the runner stays free of configuration
concerns.

Its one real responsibility is **refusing an unsupported target**. The target
registry marks battery, EV, wind and net load as UNSUPPORTED because the dataset
has no such series. Selecting one must fail here, loudly, with the reason - not
produce an empty dataset and a table of NaNs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config.ml import MlConfig
from ..data.smartds import SmartDsAdapter, SmartDsLayout
from ..paths import ProjectPaths
from .config_ml_bridge import dataset_config_from
from .dataset import DatasetBuildConfig
from .registry import REGISTRY_FILENAME
from .runner import Phase4Output, run_phase4
from .targets import (
    TargetSeries,
    TargetStatus,
    feeder_load_target,
    load_target,
    select_customer_sample,
    solar_target,
    target_spec,
)

__all__ = ["MlPipelineOutput", "run_ml_pipeline", "extract_target"]


@dataclass(frozen=True, slots=True)
class MlPipelineOutput:
    """What a ``ml dataset`` / ``ml run`` invocation produced."""

    target_id: str
    dataset_version: str
    dataset_summary: dict[str, Any]
    comparison: str
    summary: dict[str, Any]
    written: tuple[Path, ...] = ()
    phase4: Phase4Output | None = None

    def render(self) -> str:
        lines = [
            f"target: {self.target_id}",
            f"dataset version: {self.dataset_version}",
            f"samples: {self.dataset_summary.get('sample_count')}"
            f"  features: {self.dataset_summary.get('feature_count')}"
            f"  horizons: {self.dataset_summary.get('horizons')}",
            f"splits: {self.dataset_summary.get('splits')}",
        ]
        if self.summary:
            headline = self.summary.get("headline", {})
            lines.append("")
            lines.append(
                f"best at h={self.summary.get('horizons', [0])[0]}: "
                f"{headline.get('best_model_at_h')} "
                f"(MAE {headline.get('best_mae_at_h'):.4f} "
                f"{self.summary.get('target_unit')})"
                if headline.get("best_mae_at_h") is not None
                else "best: n/a"
            )
        if self.comparison:
            lines.append("")
            lines.append(self.comparison)
        if self.written:
            lines.append("")
            lines.append("written:")
            lines.extend(f"  {path}" for path in self.written)
        return "\n".join(lines)


def extract_target(config: MlConfig, layout: SmartDsLayout) -> tuple[TargetSeries, dict[str, Any], dict[str, np.ndarray] | None]:
    """Extract the configured target from the acquired dataset.

    Returns:
        The series, a provenance report about how it was selected, and weather
        arrays when the target needs them.

    Raises:
        ValueError: If the configured target is UNSUPPORTED, naming the evidence.
        FileNotFoundError: If a required file is absent from the dataset.
    """
    spec = target_spec(config.target)
    if spec.status == TargetStatus.UNSUPPORTED:
        raise ValueError(
            f"target {config.target!r} is UNSUPPORTED by SMART-DS and cannot be "
            f"forecast without fabricating data.\n"
            f"  missing: {spec.coverage}\n"
            f"  evidence: {spec.evidence}\n"
            f"  supported targets: customer_load, feeder_load, pv_generation"
        )

    adapter = SmartDsAdapter(layout)

    if config.target == "customer_load":
        names, report = select_customer_sample(
            adapter,
            count=config.customer_count,
            seed=config.seed,
            commercial_fraction=config.commercial_fraction,
        )
        series = load_target(adapter, names)
        series = _annotate(series, f"stratified sample of {len(names)} customers: {report}")
        return series, report, None

    if config.target == "feeder_load":
        series = feeder_load_target(adapter)
        return series, {"series": 1, "method": "sum over all Load objects"}, None

    if config.target == "pv_generation":
        series, weather = solar_target(layout)
        return series, {"series": 1, "method": "published column read"}, weather

    raise ValueError(f"no extractor for target {config.target!r}")  # pragma: no cover


def _annotate(series: TargetSeries, note: str) -> TargetSeries:
    """Attach a selection note to the series without changing its values."""
    from dataclasses import replace

    return replace(series, notes=series.notes + (note,))


def run_ml_pipeline(config: MlConfig, *, train: bool = True) -> MlPipelineOutput:
    """Build the dataset and, when ``train``, run every configured baseline.

    Args:
        config: The resolved experiment configuration.
        train: False builds and persists the dataset only.

    Returns:
        The pipeline output.

    Raises:
        FileNotFoundError: If the dataset has not been acquired.
        ValueError: If the target is unsupported.
    """
    data_config = dataset_config_from()
    layout = SmartDsLayout(
        root=data_config.raw_root,
        version=data_config.version,
        year=data_config.year,
        region=data_config.region,
        subregion=data_config.subregion,
        scenario=data_config.scenario,
        substation=data_config.substation,
        feeder=data_config.feeder,
    )
    if not layout.profiles_dir.is_dir():
        raise FileNotFoundError(f"dataset not acquired: {layout.profiles_dir}")

    series, _report, weather = extract_target(config, layout)
    if config.include_weather and weather is None and config.target != "pv_generation":
        # Asked for weather on a target that has none: say so rather than silently
        # proceeding without it.
        weather = None

    build_config = DatasetBuildConfig(
        lookback=config.lookback_steps,
        horizons=config.horizons,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        origin_stride=config.origin_stride,
        include_weather=config.include_weather and weather is not None,
    )
    paths = ProjectPaths.from_root()

    if not train:
        from .dataset import build_dataset, save_dataset

        dataset = build_dataset(series, build_config, weather=weather)
        npz_path, json_path = save_dataset(
            dataset, paths.artifacts / "ml" / config.label / "dataset"
        )
        return MlPipelineOutput(
            target_id=dataset.target_id,
            dataset_version=dataset.version,
            dataset_summary=dataset.summary(),
            comparison="",
            summary={},
            written=(npz_path, json_path),
        )

    output = run_phase4(
        series,
        config=build_config,
        paths=paths,
        target_label=config.label,
        weather=weather,
        models=tuple(config.models),
        seed=config.seed,
        threads=config.torch_threads,
    )
    written = tuple(
        sorted(path for path in output.report_dir.glob("*") if path.is_file())
    ) + tuple(sorted(path for path in output.dataset_dir.glob("*") if path.is_file()))
    return MlPipelineOutput(
        target_id=output.target_id,
        dataset_version=output.dataset_version,
        dataset_summary={
            "sample_count": output.summary["sample_count"],
            "feature_count": output.summary["feature_count"],
            "horizons": output.summary["horizons"],
            "splits": output.summary["splits"],
        },
        comparison=output.comparison,
        summary=output.summary,
        written=written + (output.registry_path,),
        phase4=output,
    )