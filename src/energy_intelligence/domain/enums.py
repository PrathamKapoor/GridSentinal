"""Controlled vocabulary of the energy domain.

Every enumeration the domain uses is declared here and nowhere else, so the
contract the rest of the project depends on can be audited in one place. This
matters because later phases (4-19) will import these names; adding a member to
an enum is then an explicit, reviewable act rather than a string appearing in a
distant module.

Deliberately absent: nothing energy-domain specific is invented here beyond the
categories the Phase 2 brief enumerates. In particular there is **no** optimizer,
no forecast model, no agent and no digital-twin type. Those belong to their
phases.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "AssetType",
    "AuthorityLevel",
    "ConstraintCategory",
    "FlexibilityBasis",
    "FlexibilityDirection",
    "NetworkElementKind",
    "ObjectiveCategory",
    "ObjectiveDirection",
    "QualityFlag",
    "StateOrigin",
    "UncertaintyKind",
    "Unit",
    "VariableRole",
    "ActionType",
    "CONTROLLABLE_AUTHORITY_LEVELS",
]


class Unit(StrEnum):
    """Physical units.

    Enforced explicitly rather than left to convention so a value cannot be
    silently compared across incompatible units. Every numeric field in the
    domain carries one of these.
    """

    KILOWATT = "kW"
    KILOWATT_HOURS = "kWh"
    MEGAWATT = "MW"
    MEGAWATT_HOURS = "MWh"
    VOLT_AMPERE_REACTIVE = "var"
    VOLT = "V"
    KILOVOLT = "kV"
    AMPERE = "A"
    HERTZ = "Hz"
    FRACTION = "fraction"
    PERCENT = "percent"
    DEGREE_CELSIUS = "degC"
    IRRADIANCE = "W/m2"
    WIND_SPEED = "m/s"
    MINUTES = "min"
    COUNT = "count"
    DIMENSIONLESS_RATIO = "ratio"


class VariableRole(StrEnum):
    """What a variable *is* in the control loop.

    This is the mechanism that enforces the action-versus-observation boundary
    required by the Phase 2 brief. A variable has exactly one role, and the role
    determines which container may hold it:

    ==================  ===========================================================
    OBSERVATION         Measured or inferred system state. Read-only to control.
    ACTION              A decision variable the system may set.
    DERIVED             Computable from observations (e.g. available flexibility).
    CONSTRAINT          A bound or relation the solution must respect.
    OBJECTIVE           A quantity to be optimised.
    ==================  ===========================================================
    """

    OBSERVATION = "observation"
    ACTION = "action"
    DERIVED = "derived"
    CONSTRAINT = "constraint"
    OBJECTIVE = "objective"


class QualityFlag(StrEnum):
    """Data-quality conditions attachable to any measurement.

    Modelled as flags rather than a single status because conditions co-occur:
    a value can simultaneously be stale and estimated. See ``decisions.md``
    (D-029).
    """

    OK = "ok"
    MISSING = "missing"
    STALE = "stale"
    OUTLIER = "outlier"
    INVALID_RANGE = "invalid_range"
    SENSOR_ERROR = "sensor_error"
    ESTIMATED = "estimated"
    INTERPOLATED = "interpolated"


class StateOrigin(StrEnum):
    """Where a state came from.

    The Phase 2 brief requires observed, simulated and predicted states to be
    distinguishable and non-interchangeable, so that the future digital twin can
    be compared against the real system without conflating them.
    """

    OBSERVED = "observed"
    SIMULATED = "simulated"
    PREDICTED = "predicted"
    ASSIMILATED = "assimilated"


class UncertaintyKind(StrEnum):
    """Which source of uncertainty a figure describes.

    ``UNKNOWN`` is a first-class member so a model can honestly say "not yet
    characterised" rather than omitting the field or defaulting to zero.
    """

    MEASUREMENT = "measurement"
    MODEL = "model"
    PARAMETRIC = "parametric"
    COMBINED = "combined"
    UNKNOWN = "unknown"


class AssetType(StrEnum):
    """Asset classes.

    Each maps to a concrete class in ``assets.py``. A generic ``Asset`` exists
    for infrastructure that carries no energy-specific parameters, so that
    ``Asset`` is never forced to hold every possible field.
    """

    GRID_CONNECTION = "grid_connection"
    SUBSTATION = "substation"
    LOAD = "load"
    FLEXIBLE_LOAD = "flexible_load"
    HVAC = "hvac"
    SOLAR = "solar"
    WIND = "wind"
    BATTERY = "battery"
    EV_CHARGER = "ev_charger"
    EV_FLEET = "ev_fleet"


class AuthorityLevel(StrEnum):
    """How much control the system has over an asset.

    This is the mechanism behind the chosen control-authority model: an action
    is only valid against an asset whose authority permits it. ``ADVISORY_ONLY``
    exists so a system can *reason* about an uncontrollable asset without ever
    dispatching it. See ``decisions.md`` (D-028).
    """

    DISPATCHABLE = "dispatchable"
    CURTAILABLE = "curtailable"
    SCHEDULEABLE = "scheduleable"
    ADVISORY_ONLY = "advisory_only"
    NOT_CONTROLLABLE = "not_controllable"


#: Authority levels under which an action may actually be executed.
#: ``ADVISORY_ONLY`` and ``NOT_CONTROLLABLE`` are excluded by construction.
CONTROLLABLE_AUTHORITY_LEVELS: frozenset[AuthorityLevel] = frozenset(
    {
        AuthorityLevel.DISPATCHABLE,
        AuthorityLevel.CURTAILABLE,
        AuthorityLevel.SCHEDULEABLE,
    }
)


class NetworkElementKind(StrEnum):
    """Network elements the topology can represent.

    Phase 2 carries these as descriptive metadata with declared physical
    parameters; it does **not** solve power flow. Voltage, current and thermal
    loading are therefore *observations* and *constraint categories*, not
    computed quantities. See ``decisions.md`` (D-026).
    """

    LINE = "line"
    TRANSFORMER = "transformer"
    SWITCH = "switch"
    CAPACITOR = "capacitor"
    REGULATOR = "regulator"
    FEEDER_HEAD = "feeder_head"


class ActionType(StrEnum):
    """Candidate control actions.

    Each action type declares which :class:`AuthorityLevel` is sufficient to
    perform it, enforced by ``actions.py``. Actions present here but absent from
    a given system are simply not instantiable there.
    """

    BATTERY_CHARGE = "battery_charge"
    BATTERY_DISCHARGE = "battery_discharge"
    EV_CHARGING = "ev_charging"
    EV_DEFER_CHARGING = "ev_defer_charging"
    FLEXIBLE_LOAD_SHIFT = "flexible_load_shift"
    HVAC_SETPOINT_ADJUSTMENT = "hvac_setpoint_adjustment"
    RENEWABLE_CURTAILMENT = "renewable_curtailment"
    DEMAND_RESPONSE = "demand_response"
    GRID_IMPORT_LIMIT = "grid_import_limit"


class ConstraintCategory(StrEnum):
    """Constraint categories.

    Phase 2 declares these; it does not enforce them. No algebraic formulation
    is committed, because the optimizer and its formulation belong to Phase 10.
    See ``decisions.md`` (D-030).
    """

    POWER_BALANCE = "power_balance"
    STATE_OF_CHARGE_LIMITS = "state_of_charge_limits"
    CHARGE_DISCHARGE_POWER_LIMITS = "charge_discharge_power_limits"
    RENEWABLE_AVAILABILITY = "renewable_availability"
    FLEXIBLE_LOAD_BOUNDS = "flexible_load_bounds"
    GRID_CONNECTION_CAPACITY = "grid_connection_capacity"
    VOLTAGE_LIMITS = "voltage_limits"
    THERMAL_LOADING_LIMITS = "thermal_loading_limits"
    MINIMUM_UP_TIME = "minimum_up_time"
    RAMP_RATE_LIMITS = "ramp_rate_limits"
    EV_DEPARTURE_DEADLINE = "ev_departure_deadline"
    DEMAND_FLEXIBILITY_LIMIT = "demand_flexibility_limit"


class ObjectiveCategory(StrEnum):
    """Energy objectives.

    Phase 2 defines these *separately* and records their conflicts. It does not
    combine them into a weighted sum: choosing weights is a policy decision
    belonging to Phase 10, and premature weighting would hide the trade-offs.
    See ``decisions.md`` (D-031).
    """

    ENERGY_COST = "energy_cost"
    PEAK_DEMAND = "peak_demand"
    RENEWABLE_CURTAILMENT = "renewable_curtailment"
    RENEWABLE_UTILIZATION = "renewable_utilization"
    BATTERY_DEGRADATION = "battery_degradation"
    CONSTRAINT_VIOLATION = "constraint_violation"
    GRID_IMPORT_PEAK = "grid_import_peak"
    LOAD_SHED_AMOUNT = "load_shed_amount"
    UNMET_DEMAND = "unmet_demand"
    EMISSIONS = "emissions"


class ObjectiveDirection(StrEnum):
    """Whether an objective is minimised or maximised."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class FlexibilityDirection(StrEnum):
    """Which way a controllable quantity would have to move.

    **Defined on the controlled quantity's own sign convention, not on which
    direction is desirable.** The convention here is *net demand (load), positive
    when consuming*, because that is the quantity a demand-response product acts on
    and the one :class:`ConstraintCategory`.DEMAND_FLEXIBILITY_LIMIT is denominated
    in (``Unit.KILOWATT``).

    ===================  ==========================================================
    DOWNWARD             net demand decreases. For a load, power is shed. This
                         is the direction a peak-shaving system wants and the one
                         most demand-response programmes pay for.
    UPWARD               net demand increases. For a load, power is added, e.g.
                         to absorb surplus generation.
    ===================  ==========================================================

    Two other sign conventions exist in this project and they are **opposite** to
    each other, which is why the mapping is stated rather than assumed:

    * :class:`~energy_intelligence.domain.state.NodePower` takes *load as negative*
      ("generation and grid import are positive, load and grid export are
      negative"). Under that convention a DOWNWARD flexibility is a movement of net
      power **upward**, toward zero.
    * :class:`ActionType`.FLEXIBLE_LOAD_SHIFT carries a **signed load** setpoint,
      so DOWNWARD is a **negative** setpoint.

    A magnitude is always non-negative in ``Unit.KILOWATT``; only the mapping to
    those two conventions carries a sign.
    """

    UPWARD = "upward"
    DOWNWARD = "downward"


class FlexibilityBasis(StrEnum):
    """What kind of evidence supports a flexibility figure.

    This is the single most important distinction in Phase 9 and the reason the
    enumeration exists rather than a boolean "estimated" flag. The failure it exists
    to prevent is a statistical quantity being read as a physical capability -
    "the load has moved by 4 kW before" being reported as "4 kW of dispatchable
    flexibility".

    ====================  ========================================================
    PHYSICAL              Supported by authoritative control or capability metadata:
                          a rated power limit, a usable-energy window, a stated
                          setpoint range. Requires the asset to be one the system
                          may actually command.
    STATISTICAL_PROXY     Derived from observed behaviour, e.g. how far a load has
                          historically departed from its own expected profile. It
                          describes **behaviour**, not capability: it says the load
                          *has moved* by this much, never that an operator *can make
                          it* move by this much.
    ASSUMED               Declared by a scenario, for demonstration or planning. Not
                          discovered from data and not evidence about any real asset.
    UNKNOWN               Not supported by the available data. Carries **no
                          magnitude**; an unknown cannot have a number.
    ====================  ========================================================

    The status axes this project already owns are deliberately *not* duplicated
    here. ``VariableRole.DERIVED`` says what role the variable plays (its own
    docstring names available flexibility as its example), ``AuthorityLevel`` says
    how much control the system holds, ``QualityFlag`` describes data condition,
    and ``StateOrigin`` describes where a state came from. This enum answers only
    the question none of those answer: *what kind of evidence is behind this
    number*.
    """

    PHYSICAL = "physical"
    STATISTICAL_PROXY = "statistical_proxy"
    ASSUMED = "assumed"
    UNKNOWN = "unknown"
