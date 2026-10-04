"""Phase 8's offline analysis. Nothing here may be reached from a forecasting path.

Same boundary as Phase 7's ``router.offline``: the oracle and every analysis that reads a
realised target live here, and no deployable module imports it. Two tests enforce that.

What is in this package
----------------------

``experiment``
    The runner. Reads Phase 7's cached expert forecasts, reuses Phase 7's fitted
    fixed-ensemble weights verbatim, fits the uncertainty methods on the calibration
    split, and evaluates once on the sealed test split.
``analysis``
    Per-method evaluation: coverage, width, sharpness, ranking, reliability,
    overconfidence, regimes, imminent-ramp detection.
``verdict``
    The phase's answer, derived from the numbers.
"""

from .experiment import METHOD_NAMES, Phase8Result, run_phase8, render_summary
from .verdict import PHASE8_ANSWERS, phase8_verdict

__all__ = [
    "METHOD_NAMES",
    "PHASE8_ANSWERS",
    "Phase8Result",
    "phase8_verdict",
    "render_summary",
    "run_phase8",
]