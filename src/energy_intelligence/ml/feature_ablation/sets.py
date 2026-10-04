"""Phase 10's feature sets, built only from the repository's existing catalogue.

Phase 10 is an **ablation study, not feature engineering**. Every name below is an existing
``FeatureSpec`` from :mod:`energy_intelligence.ml.features`; nothing is invented here. In
particular the four weather-derived features are excluded from every set, because the phase
brief forbids adding weather and because including them would confound "does the feature family
help" with "does exogenous information help".

The five sets form a nested ladder, which is what makes the incremental-effect arithmetic in
:mod:`selection` well defined:

===========  ==================================================  ======
set          members                                              count
===========  ==================================================  ======
``A``        calendar                                              4
``B``        autoregressive lags                                   5
``C``        A + B                                                 9
``D``        C + rolling statistics                               11
``E``        D + ramp and same-hour rolling                       13
===========  ==================================================  ======

``B`` deliberately includes ``value_at_origin``. It reads the target series at or before the
origin, exactly as the lag terms do, so leaving it out would make ``A`` -> ``B`` a comparison
between "calendar only" and "calendar plus history *except the present*", which is not the
question the increment is meant to ask.

``E`` is every non-weather feature in the catalogue, so the ladder terminates at the full
available set rather than at an arbitrary stopping point.
"""

from __future__ import annotations

from ..features import FEATURE_CATALOGUE

__all__ = [
    "FEATURE_SETS",
    "SET_ORDER",
    "FEATURE_SET_IDS",
    "WEATHER_FEATURES",
    "members",
    "checksum",
    "describe",
]

#: Features whose source is the solar file's weather columns. Excluded from every set.
WEATHER_FEATURES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_CATALOGUE if spec.requires_weather
)

_CALENDAR: tuple[str, ...] = (
    "hour_of_day",
    "day_of_week",
    "day_of_year",
    "is_weekend",
)

_LAGS: tuple[str, ...] = (
    "value_at_origin",
    "lag_1",
    "lag_4",
    "lag_96",
    "lag_672",
)

_ROLLING: tuple[str, ...] = ("roll_mean_96", "roll_std_96")

_RAMP: tuple[str, ...] = ("ramp_1", "roll_mean_same_hour_7d")

#: The ladder, in nesting order. ``C`` is defined as the union rather than spelled out, so a
#: change to ``A`` or ``B`` cannot silently desynchronise ``C``.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "A": _CALENDAR,
    "B": _LAGS,
    "C": _CALENDAR + _LAGS,
    "D": _CALENDAR + _LAGS + _ROLLING,
    "E": _CALENDAR + _LAGS + _ROLLING + _RAMP,
}

#: Report and iteration order.
SET_ORDER: tuple[str, ...] = ("A", "B", "C", "D", "E")

#: The IDs a caller may name, which is exactly :data:`SET_ORDER`.
FEATURE_SET_IDS: frozenset[str] = frozenset(SET_ORDER)


def members(set_id: str) -> tuple[str, ...]:
    """The feature names in one set.

    Args:
        set_id: ``"A"`` through ``"E"``.

    Returns:
        The member feature names, in the order they are handed to the matrix builder.

    Raises:
        KeyError: If the set is not one of the five.
    """
    return FEATURE_SETS[set_id]


def checksum(set_id: str) -> str:
    """A stable 12-hex-character fingerprint of one set's membership.

    Short and stable rather than cryptographic: its job is to make "the same feature set" a
    checkable claim in the freeze files, not to resist tampering. The full member list is
    written into the freeze alongside it, so a hash collision could not hide a difference.

    Args:
        set_id: ``"A"`` through ``"E"``.

    Returns:
        The fingerprint.
    """
    import hashlib

    payload = "|".join(members(set_id)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def describe() -> dict[str, dict[str, object]]:
    """Every set with its members, count, checksum and nesting parent.

    Returns:
        ``{set_id: {...}}``, in :data:`SET_ORDER`.
    """
    parents = {"A": None, "B": None, "C": "A", "D": "C", "E": "D"}
    return {
        set_id: {
            "features": list(members(set_id)),
            "count": len(members(set_id)),
            "checksum": checksum(set_id),
            "parent": parents[set_id],
            "adds_vs_parent": (
                []
                if parents[set_id] is None
                else [name for name in members(set_id) if name not in members(parents[set_id])]
            ),
            "excludes_weather": sorted(WEATHER_FEATURES),
        }
        for set_id in SET_ORDER
    }