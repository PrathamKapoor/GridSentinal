"""Coupling Phase 8 uncertainty to how much flexibility the system will *rely* upon.

The research question
    -------------------
    Phase 8 measures how uncertain the demand forecast is. Phase 9 asks how much response
    capability to associate with the system. The natural bridge is: *when the forecast is
    unusually uncertain, promise less flexibility.* That is the intuition, and it is not
    automatically true - so this module exists to test it rather than to assume it.

Two distinct objects, deliberately not conflated
    ----------------------------------------------
    The **behavioural envelope** says how far demand has historically moved from its own
    expected profile. The **forecast interval** says how well the current model predicts.
    On a dataset where every load is uncommandable, these may be substantially the same
    number, in which case conditioning one on the other adds no information and the honest
    answer is that the coupling is unjustified.

    :func:`justification` measures exactly that overlap, and the phase reports it whichever
    way it comes out. A coupling that cannot be justified is not shipped as a discount.

The reliance factor
    ------------------
    For a row whose Phase 8 normalised interval width is ``w`` and whose trailing typical
    width is ``w_ref``:

    .. code-block:: text

        reliance = clip(w_ref / max(w, eps), floor, 1.0)

    ``1.0`` when the current moment is no harder to predict than usual, falling towards
    ``floor`` as it becomes harder. This is a **reliance policy**, not a statistical claim:
    it says how much of the behavioural envelope the system is willing to promise, which is
    a governance choice about conservatism, not an inference about the world.

    Three properties make it defensible rather than arbitrary:

    * it is **monotone decreasing** in predictive uncertainty, which is the stated intent;
    * it is **anchored** - ``w_ref`` is a causal trailing median, so a typical moment is
      never discounted at all and the phase's headline envelopes match the raw ones;
    * it is **bounded below** by a configured floor, so a pathological interval collapses the
      claim toward the floor rather than to zero, and the floor is reported.

Why not simply shrink the envelope by the interval width
    Because that would make the promised capability a rescaled forecast error, i.e. it
    would silently define flexibility as "how wrong we usually are" and then discount it.
    That is a different claim from the one being made, and it would collapse the two
    quantities this phase is careful to keep apart.

Causality of the reference
    ``w_ref`` at row ``t`` is a median over rows strictly **before** ``t`` in the panel's
    own chronological order. Phase 8's interval at a forecast origin is available at that
    origin, so every row contributing to the reference was already published when the row
    being scored was issued. No future row enters any reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "RelianceFactor",
    "apply_reliance",
    "causal_trailing_reference",
    "justification",
    "reliance_from_widths",
]


@dataclass(frozen=True, slots=True)
class RelianceFactor:
    """Per-row reliance factors and the reference they were computed against.

    Attributes:
        factor: ``[rows]`` reliance in ``[floor, 1.0]``.
        reference: ``[rows]`` the causal trailing typical width used for each row.
        normalised_width: ``[rows]`` the row's own Phase 8 width, per unit of rated kW.
        floor: The configured lower bound.
        window_rows: How many preceding rows the trailing reference covered.
    """

    factor: np.ndarray
    reference: np.ndarray
    normalised_width: np.ndarray
    floor: float
    window_rows: int

    def describe(self) -> dict[str, Any]:
        return {
            "floor": float(self.floor),
            "window_rows": int(self.window_rows),
            "factor": {
                "mean": float(self.factor.mean()),
                "min": float(self.factor.min()),
                "p10": float(np.quantile(self.factor, 0.10)),
                "median": float(np.median(self.factor)),
                "p90": float(np.quantile(self.factor, 0.90)),
                "max": float(self.factor.max()),
                "share_at_full": float(np.mean(self.factor >= 1.0 - 1e-12)),
                "share_at_floor": float(np.mean(self.factor <= self.floor + 1e-12)),
            },
        }


def causal_trailing_reference(
    normalised: np.ndarray, *, window: int, min_periods: int
) -> np.ndarray:
    """A rolling median over the rows strictly **before** each row.

    Args:
        normalised: ``[rows]`` per-unit widths in panel chronological order.
        window: How many preceding rows to consider.
        min_periods: Rows needed before a reference is produced. Below this the reference is
            the expanding median of everything available so far.

    Returns:
        ``[rows]`` trailing reference. The first entry uses only the rows before it, so it
        is ``NaN`` when the window is empty - which the caller must handle rather than
        silently substituting the row's own width, since that would make the reliance factor
        exactly 1.0 everywhere at the start of the panel.

    Raises:
        ValueError: If ``window`` or ``min_periods`` is below 1, or the input is not 1-D.
    """
    values = np.asarray(normalised, dtype=np.float64).ravel()
    if int(window) < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    if int(min_periods) < 1:
        raise ValueError(f"min_periods must be at least 1, got {min_periods}")
    if not np.isfinite(values).all():
        raise ValueError(
            f"normalised widths contain {int((~np.isfinite(values)).sum())} non-finite "
            f"entries; a trailing reference over them would be undefined"
        )
    n = values.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    # Expanding prefix medians are cheap and correct; only the later rows need the window.
    for i in range(n):
        start = max(0, i - int(window))
        segment = values[start:i]
        if segment.size >= int(min_periods):
            out[i] = float(np.median(segment))
        elif i > 0:
            out[i] = float(np.median(values[:i]))
    return out


def reliance_from_widths(
    *,
    interval_width_kw: np.ndarray,
    rated_kw: np.ndarray,
    window: int = 3840,
    min_periods: int = 96,
    floor: float = 0.25,
) -> RelianceFactor:
    """Turn Phase 8 interval widths into reliance factors.

    Args:
        interval_width_kw: ``[rows]`` Phase 8 interval width in kW, panel chronological order.
        rated_kw: ``[rows]`` the row's rated kW. Widths are divided by it so a 4 kW interval
            on a 5 kW household and on a 3000 kW shop are comparable; without this the
            factor would mostly measure customer size.
        window: Preceding rows forming the trailing reference.
        min_periods: Rows needed before the window is used.
        floor: Lowest reliance factor permitted.

    Returns:
        The reliance factors.

    Raises:
        ValueError: If shapes disagree, a rated kW is non-positive, or ``floor`` is outside
            ``[0, 1]``.
    """
    width = np.asarray(interval_width_kw, dtype=np.float64).ravel()
    scale = np.asarray(rated_kw, dtype=np.float64).ravel()
    if width.size != scale.size:
        raise ValueError(
            f"interval_width_kw {width.shape} and rated_kw {scale.shape} must be the same length"
        )
    if (scale <= 0).any():
        raise ValueError(
            f"rated kW must be positive; got a minimum of {float(scale.min())}"
        )
    if not 0.0 <= float(floor) <= 1.0:
        raise ValueError(f"floor must lie in [0, 1], got {floor}")
    normalised = width / scale
    reference = causal_trailing_reference(
        normalised, window=int(window), min_periods=int(min_periods)
    )
    # Rows with no reference yet are not discounted: with nothing to compare against, the
    # honest statement is "typical", not "uncertain". They are counted so the report can say
    # how many rows that covered.
    usable = np.isfinite(reference) & (reference > 0)
    factor = np.ones(width.size, dtype=np.float64)
    factor[usable] = np.clip(
        reference[usable] / np.maximum(normalised[usable], 1e-12), float(floor), 1.0
    )
    return RelianceFactor(
        factor=factor,
        reference=np.where(usable, reference, np.nan),
        normalised_width=normalised,
        floor=float(floor),
        window_rows=int(window),
    )


def apply_reliance(magnitudes_kw: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Scale magnitudes by the reliance factor.

    Args:
        magnitudes_kw: ``[rows]`` or ``[rows, horizons]`` magnitudes, kW.
        factor: ``[rows]`` reliance factors.

    Returns:
        The discounted magnitudes, same shape.

    Raises:
        ValueError: If the row counts disagree, or a factor is outside ``[0, 1]``.
    """
    magnitudes = np.asarray(magnitudes_kw, dtype=np.float64)
    factors = np.asarray(factor, dtype=np.float64).ravel()
    if magnitudes.shape[0] != factors.size:
        raise ValueError(
            f"magnitudes have {magnitudes.shape[0]} rows but {factors.size} factors were supplied"
        )
    if not np.isfinite(factors).all() or (factors < 0).any() or (factors > 1).any():
        raise ValueError(
            f"reliance factors must lie in [0, 1]; got [{np.nanmin(factors)}, {np.nanmax(factors)}]"
        )
    if magnitudes.ndim == 1:
        return magnitudes * factors
    return magnitudes * factors[:, None]


def justification(
    *,
    normalised_width: np.ndarray,
    absolute_deviation_kw: np.ndarray,
    factor: np.ndarray,
    spearman,
) -> dict[str, Any]:
    """Test whether predictive uncertainty actually predicts deviation magnitude.

    This is the measurement that decides whether the reliance discount means anything. If
    the correlation is near zero then a wide interval says nothing about how far the load
    will move, and discounting on it would reduce the promised flexibility for no evidential
    reason - which would be conservatism without cause.

    Args:
        normalised_width: ``[rows]`` per-unit Phase 8 width.
        absolute_deviation_kw: ``[rows]`` absolute deviation of realised demand from the
            expected profile.
        factor: ``[rows]`` the reliance factor that was applied.
        spearman: The rank-correlation callable, injected so this module does not depend on
            which statistics implementation is in use.

    Returns:
        The correlations, the dispersion of the factor, and an explicit verdict string that
        says whether the coupling is supported.
    """
    width = np.asarray(normalised_width, dtype=np.float64).ravel()
    deviation = np.asarray(absolute_deviation_kw, dtype=np.float64).ravel()
    factors = np.asarray(factor, dtype=np.float64).ravel()
    if not (width.size == deviation.size == factors.size):
        raise ValueError(
            f"width {width.size}, deviation {deviation.size} and factor {factors.size} "
            f"must be the same length"
        )
    usable = np.isfinite(width) & np.isfinite(deviation) & np.isfinite(factors)
    rho = float("nan")
    if int(usable.sum()) > 2:
        rho = float(spearman(width[usable], deviation[usable]))
    spread = float(np.std(factors[usable])) if usable.any() else 0.0
    # A threshold, stated rather than fitted: below 0.05 the rank correlation is not
    # distinguishable from noise at this sample size, and a discount driven by it would be
    # conservatism without evidence.
    supported = bool(rho == rho and rho >= 0.05)
    return {
        "spearman_width_vs_absolute_deviation": rho,
        "n_rows": int(usable.sum()),
        "factor_mean": float(factors[usable].mean()) if usable.any() else None,
        "factor_std": spread,
        "factor_distinct_values": int(np.unique(np.round(factors[usable], 6)).size)
        if usable.any()
        else 0,
        "threshold": 0.05,
        "coupling_supported": supported,
        "verdict": (
            "predictive uncertainty carries information about realised deviation magnitude, "
            "so conditioning the promised flexibility on it is evidence-based"
            if supported
            else "predictive uncertainty does NOT predict realised deviation magnitude on "
            "this data, so discounting flexibility by it would be conservatism without "
            "evidence; the factor is reported but not relied upon"
        ),
    }