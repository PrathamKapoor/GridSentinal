"""Phase 8: probabilistic forecasting, uncertainty and calibration.

Phase 7 established that a heterogeneous ensemble of persistence, a gradient-boosted
tree and a temporal convolution is the strongest point-forecast mechanism this project
has measured, and that a learned gate over it does not beat it. This phase leaves the
point forecast exactly where it is and asks a different question: **how uncertain is
it, and does saying so identify the forecasts that will be wrong?**

The layer has three parts, in the order the argument runs.

``measures``
    Disagreement between the three expert forecasts. Phase 7 established they disagree;
    this asks whether that disagreement carries information about future error, which is
    a different claim and is tested rather than assumed.

``intervals`` / ``calibration`` / ``scales`` / ``quantile``
    Four interval mechanisms in three families, and a **scale function** as the one
    concept that separates them. Every method here is ``point forecast +/- a width``, and
    the families differ only in how that width is obtained:

    ============================  ==================================================
    Family                        Width comes from
    ============================  ==================================================
    empirical residual quantile   quantiles of realised residuals, calibrated
    split conformal               the ``ceil((n+1)(1-alpha))``-th smallest score
    direct quantile regression    a pinball-loss model of the residual quantiles
    ============================  ==================================================

    Independently of family, the width can be scaled globally, by a learned function of
    origin-observable state, or by the experts' own disagreement. The phase's spine is
    whether any of those scalings beat a constant.

``artifact``
    :class:`ProbabilisticForecast`, the stable output a future optimiser, flexibility
    estimator or decision-assurance engine consumes. It carries the point forecast, the
    interval, the nominal level, a measured calibration status and provenance.

Every method here is *offline analysis plus a deployable estimator*. The production path
reads no realised target; see ``leakage_audit``.
"""

from .artifact import (
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    CALIBRATION_UNKNOWN,
    BandCutPoints,
    ProbabilisticForecast,
    ProbabilisticForecastBatch,
    UncertaintyBand,
    forecast_to_domain,
    interval_to_estimate,
    save_batch,
)
from .calibration import (
    CONFORMAL_GUARANTEE,
    ConformalCalibrator,
    conformal_quantile_index,
    fit_conformal,
    fit_conformal_quantile,
)
from .evaluation import (
    imminent_ramp_detection,
    interval_audit,
    overconfidence_analysis,
    reliability_table,
    uncertainty_ranking,
)
from .intervals import (
    DEFAULT_NOMINAL_LEVELS,
    PredictionInterval,
    empirical_quantile,
    quantile_levels,
    widths_from_scale,
)
from .measures import (
    DisagreementMeasures,
    disagreement_measures,
    spearman,
)
from .metrics import (
    IntervalMetrics,
    coverage,
    coverage_error,
    expected_calibration_error,
    interval_score,
    mean_width,
    sharpness_ratio,
    weighted_interval_score,
)
from .quantile import QuantileResidualModel, fit_quantile_model
from .scales import (
    DisagreementScale,
    GlobalScale,
    LearnedScale,
    fit_learned_scale,
    fitted_scale,
)
from .split import (
    CalibrationSplit,
    build_calibration_split,
    split_parity_check,
)

__all__ = [
    "BandCutPoints",
    "CALIBRATION_FAIL",
    "CALIBRATION_UNKNOWN",
    "CONFORMAL_GUARANTEE",
    "CALIBRATION_PASS",
    "DEFAULT_NOMINAL_LEVELS",
    "CalibrationSplit",
    "ConformalCalibrator",
    "DisagreementMeasures",
    "DisagreementScale",
    "GlobalScale",
    "IntervalMetrics",
    "LearnedScale",
    "PredictionInterval",
    "ProbabilisticForecast",
    "ProbabilisticForecastBatch",
    "QuantileResidualModel",
    "fit_conformal",
    "fit_learned_scale",
    "fit_quantile_model",
    "imminent_ramp_detection",
    "UncertaintyBand",
    "build_calibration_split",
    "conformal_quantile_index",
    "coverage",
    "coverage_error",
    "disagreement_measures",
    "empirical_quantile",
    "expected_calibration_error",
    "fit_conformal_quantile",
    "fitted_scale",
    "forecast_to_domain",
    "interval_audit",
    "interval_score",
    "interval_to_estimate",
    "mean_width",
    "overconfidence_analysis",
    "quantile_levels",
    "save_batch",
    "reliability_table",
    "sharpness_ratio",
    "spearman",
    "split_parity_check",
    "uncertainty_ranking",
    "weighted_interval_score",
    "widths_from_scale",
]