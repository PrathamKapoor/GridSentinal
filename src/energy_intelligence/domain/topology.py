"""Network topology.

Chosen model: **topology-aware, node-granular, physics deferred.** See
``decisions.md`` (D-026).

What this means concretely, and why:

* Real :class:`Node` entities exist and carry identity, nominal voltage and a
  parent reference, so the graph is a first-class part of the model rather than
  an afterthought. This is what the Phase 11 digital twin will need.
* Assets attach to nodes, so power can be attributed per node.
* Network elements carry declared ratings and, where available, physical
  parameters. They are **descriptive metadata in Phase 2**.
* Phase 2 does **not** solve power flow and does **not** compute voltage,
  current or thermal loading. Those appear as *observation* variables and as
  *constraint categories*, never as derived quantities. Claiming otherwise
  without a solver would be a false capability.

The model degrades cleanly: a system with a single node and no elements is a
valid lumped representation, so the same types serve both a detailed feeder and
an aggregate portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import NetworkElementKind
from .errors import DomainValidationError
from .identifiers import NetworkElementId, NodeId
from .quantities import Quantity

__all__ = ["Node", "NetworkElement", "NetworkTopology"]


@dataclass(frozen=True, slots=True)
class Node:
    """A point in the distribution network.

    Attributes:
        node_id: Unique identifier.
        name: Human-readable label.
        voltage_kv: Nominal voltage level, if known. ``None`` for an aggregate
            node, which is legitimate.
        parent_node_id: Upstream node, or ``None`` for the point of common
            coupling with the utility.
        is_grid_connection: True for the node(s) where the utility grid attaches.
        metadata: Free-form, non-authoritative extras.
    """

    node_id: NodeId
    name: str
    voltage_kv: Quantity | None = None
    parent_node_id: NodeId | None = None
    is_grid_connection: bool = False
    metadata: dict[str, str] | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.node_id, NodeId):
            errors.append(f"node_id must be a NodeId, got {type(self.node_id).__name__}")
        if not isinstance(self.name, str) or not self.name.strip():
            errors.append(f"name must be a non-empty string, got {self.name!r}")
        if self.voltage_kv is not None:
            if not isinstance(self.voltage_kv, Quantity):
                errors.append(
                    f"voltage_kv must be a Quantity or None, got {type(self.voltage_kv).__name__}"
                )
            elif self.voltage_kv.value <= 0:
                errors.append(f"voltage_kv must be positive, got {self.voltage_kv.value}")
        if self.parent_node_id is not None and not isinstance(self.parent_node_id, NodeId):
            errors.append(
                f"parent_node_id must be a NodeId or None, got {type(self.parent_node_id).__name__}"
            )
        if self.parent_node_id == self.node_id:
            errors.append(f"node {self.node_id} cannot be its own parent")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            errors.append(f"metadata must be a dict or None, got {type(self.metadata).__name__}")
        if errors:
            raise DomainValidationError(errors, subject="Node")


@dataclass(frozen=True, slots=True)
class NetworkElement:
    """A physical connection or device between two nodes.

    Attributes:
        element_id: Unique identifier.
        kind: What sort of element this is.
        from_node_id: Upstream node.
        to_node_id: Downstream node.
        rating: Thermal or power rating, if declared.
        is_in_service: Whether the element is currently energised. An
            out-of-service element is how an outage or a switching state enters
            the model, which the Phase 12 Red Team will need to perturb.
        parameters: Physical parameters (impedance, length, tap position, ...).
            Descriptive in Phase 2.
        metadata: Free-form, non-authoritative extras.
    """

    element_id: NetworkElementId
    kind: NetworkElementKind
    from_node_id: NodeId
    to_node_id: NodeId
    rating: Quantity | None = None
    is_in_service: bool = True
    parameters: dict[str, float] | None = None
    metadata: dict[str, str] | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.element_id, NetworkElementId):
            errors.append(
                f"element_id must be a NetworkElementId, got {type(self.element_id).__name__}"
            )
        if not isinstance(self.kind, NetworkElementKind):
            errors.append(f"kind must be a NetworkElementKind, got {type(self.kind).__name__}")
        for field_name in ("from_node_id", "to_node_id"):
            value = getattr(self, field_name)
            if not isinstance(value, NodeId):
                errors.append(f"{field_name} must be a NodeId, got {type(value).__name__}")
        if isinstance(self.from_node_id, NodeId) and self.from_node_id == self.to_node_id:
            errors.append(
                f"element {self.element_id} connects {self.from_node_id} to itself"
            )
        if self.rating is not None:
            if not isinstance(self.rating, Quantity):
                errors.append(
                    f"rating must be a Quantity or None, got {type(self.rating).__name__}"
                )
            elif self.rating.value < 0:
                errors.append(f"rating must be non-negative, got {self.rating.value}")
        if self.parameters is not None:
            if not isinstance(self.parameters, dict):
                errors.append(
                    f"parameters must be a dict or None, got {type(self.parameters).__name__}"
                )
            else:
                for key, value in self.parameters.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        errors.append(
                            f"parameter {key!r} must be a real number, got {value!r}"
                        )
        if self.metadata is not None and not isinstance(self.metadata, dict):
            errors.append(f"metadata must be a dict or None, got {type(self.metadata).__name__}")
        if errors:
            raise DomainValidationError(errors, subject="NetworkElement")


@dataclass(frozen=True, slots=True)
class NetworkTopology:
    """The set of nodes and the elements connecting them.

    Attributes:
        nodes: Every node. Must be non-empty; an empty grid has nothing to model.
        elements: Connections between nodes.
        description: Human-readable description of the boundary being modelled.
        metadata: Free-form, non-authoritative extras.
    """

    nodes: tuple[Node, ...]
    elements: tuple[NetworkElement, ...] = ()
    description: str = ""
    metadata: dict[str, str] | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []

        if not isinstance(self.nodes, tuple):
            errors.append(f"nodes must be a tuple, got {type(self.nodes).__name__}")
            nodes = ()
        else:
            nodes = self.nodes
            if not nodes:
                errors.append("topology must declare at least one node")
            if not all(isinstance(node, Node) for node in nodes):
                errors.append("every entry in nodes must be a Node")

        if not isinstance(self.elements, tuple):
            errors.append(f"elements must be a tuple, got {type(self.elements).__name__}")
            elements = ()
        else:
            elements = self.elements
            if not all(isinstance(element, NetworkElement) for element in elements):
                errors.append("every entry in elements must be a NetworkElement")

        if not errors:
            node_ids = [node.node_id for node in nodes]
            duplicates = _duplicates(node_ids)
            if duplicates:
                errors.append(f"duplicate node_id(s): {sorted(str(d) for d in duplicates)}")

            known = set(node_ids)
            for node in nodes:
                if node.parent_node_id is not None and node.parent_node_id not in known:
                    errors.append(
                        f"node {node.node_id} references unknown parent {node.parent_node_id}"
                    )
                if node.is_grid_connection and node.parent_node_id is not None:
                    errors.append(
                        f"node {node.node_id} is a grid connection and must not have "
                        f"a parent, but references {node.parent_node_id}"
                    )

            element_ids = [element.element_id for element in elements]
            duplicates = _duplicates(element_ids)
            if duplicates:
                errors.append(
                    f"duplicate element_id(s): {sorted(str(d) for d in duplicates)}"
                )

            for element in elements:
                for label, node_ref in (
                    ("from_node_id", element.from_node_id),
                    ("to_node_id", element.to_node_id),
                ):
                    if node_ref not in known:
                        errors.append(
                            f"element {element.element_id} references unknown "
                            f"{label} {node_ref}"
                        )

            if nodes and not any(node.is_grid_connection for node in nodes):
                errors.append(
                    "no node is marked is_grid_connection; the system boundary "
                    "must state where it connects to the utility grid"
                )

        if errors:
            raise DomainValidationError(errors, subject="NetworkTopology")

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    @property
    def node_ids(self) -> frozenset[NodeId]:
        return frozenset(node.node_id for node in self.nodes)

    @property
    def grid_connection_nodes(self) -> tuple[Node, ...]:
        return tuple(node for node in self.nodes if node.is_grid_connection)

    def node(self, node_id: NodeId) -> Node | None:
        for candidate in self.nodes:
            if candidate.node_id == node_id:
                return candidate
        return None

    def has_node(self, node_id: NodeId) -> bool:
        return self.node(node_id) is not None

    @property
    def is_aggregate(self) -> bool:
        """Whether this topology is a single lumped node.

        A single node with no elements is the aggregate representation. Systems
        report this rather than hiding it, because a result obtained on an
        aggregate topology cannot support per-location claims.
        """
        return len(self.nodes) == 1 and not self.elements


def _duplicates(values: list) -> list:
    seen: set = set()
    repeated: set = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return list(repeated)
