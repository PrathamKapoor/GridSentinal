"""Phase 7: heterogeneous energy expert routing.

The pool is persistence, Phase 4's classical gradient boosting and Phase 5's temporal
convolutional network - heterogeneous models, deliberately **not** made alike. A small
learned router weights them per row and per horizon from origin-observable features.

The layout is a boundary, not an organisational preference:

``router.experts`` / ``router.features`` / ``router.model`` / ``router.routing`` /
``router.training`` / ``router.ensembles`` / ``router.cache``
    Deployable. Given a panel position and the origin-observable state, these produce a
    forecast and the routing decision behind it. None of them can see a realised target,
    and none of them imports anything from ``router.offline``.

``router.offline``
    Analysis that legitimately reads the target: the oracle upper bound, expert
    diversity, routing-quality metrics, ablations and the experiment runner. A test
    asserts the deployable side never imports it, so "the oracle is only for offline
    analysis" is enforced rather than promised.
"""

from .cache import ExpertCache, load_expert_cache, save_expert_cache
from .ensembles import (
    FixedEnsemble,
    apply_best_single,
    best_single_forecast,
    ensemble_similarity,
    fit_fixed_weights,
    fixed_forecast,
    select_best_single,
    uniform_weights,
)
from .experts import (
    PHASE4_FEATURE_NAMES,
    PHASE7_EXPERTS,
    EvaluationPanel,
    ExpertForecast,
    ExpertPool,
    ForecastExpert,
    GbmExpert,
    PersistenceExpert,
    TcnExpert,
    build_experts,
    build_panel,
)
from .features import (
    FEATURE_GROUPS,
    RouterFeatureSpec,
    RouterFeatures,
    append_lagged_error_features,
    assert_router_features_are_causal,
    build_router_features,
    router_feature_version,
)
from .model import EnergyRouterNet, RouterArchitecture, build_router, simplex_weights
from .routing import (
    ROUTING_STRATEGIES,
    ROUTING_STRATEGY_NAMES,
    EnergyRouter,
    RoutingDecision,
    hard_selection,
    mix,
)
from .training import (
    RouterTrainingConfig,
    TrainedRouter,
    fit_router,
    load_router,
    router_weights_for,
)

__all__ = [
    "PHASE4_FEATURE_NAMES",
    "PHASE7_EXPERTS",
    "FEATURE_GROUPS",
    "ROUTING_STRATEGIES",
    "ROUTING_STRATEGY_NAMES",
    "EnergyRouter",
    "EnergyRouterNet",
    "EvaluationPanel",
    "ExpertCache",
    "ExpertForecast",
    "ExpertPool",
    "FixedEnsemble",
    "ForecastExpert",
    "GbmExpert",
    "PersistenceExpert",
    "RouterArchitecture",
    "RouterFeatureSpec",
    "RouterFeatures",
    "RouterTrainingConfig",
    "RoutingDecision",
    "TcnExpert",
    "TrainedRouter",
    "append_lagged_error_features",
    "apply_best_single",
    "assert_router_features_are_causal",
    "best_single_forecast",
    "build_experts",
    "build_panel",
    "build_router",
    "build_router_features",
    "ensemble_similarity",
    "fit_fixed_weights",
    "fit_router",
    "fixed_forecast",
    "hard_selection",
    "load_expert_cache",
    "load_router",
    "mix",
    "router_feature_version",
    "router_weights_for",
    "save_expert_cache",
    "select_best_single",
    "simplex_weights",
    "uniform_weights",
]