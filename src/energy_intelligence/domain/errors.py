"""Domain validation errors.

Every domain object validates itself on construction. Failing at construction
rather than at consumption means an invalid energy state cannot exist, so no
later phase has to defend against one.

Following the principle borrowed from SAT-SA: violations are collected and
reported together, so a malformed object can be fixed in one pass. See
``decisions.md`` (D-025).
"""

from __future__ import annotations

__all__ = ["DomainError", "DomainValidationError"]


class DomainError(Exception):
    """Base class for energy-domain failures."""


class DomainValidationError(DomainError):
    """Raised when a domain object violates an energy-system invariant.

    Attributes:
        errors: Every violation found, not just the first.
        subject: Optional name of the offending object, for context.
    """

    def __init__(self, errors: list[str] | tuple[str, ...], subject: str | None = None) -> None:
        self.errors = tuple(errors)
        self.subject = subject
        detail = "\n".join(f"  - {error}" for error in self.errors)
        prefix = f"{subject}: " if subject else ""
        super().__init__(
            f"Invalid energy domain object ({len(self.errors)} error(s)):\n{prefix}{detail}"
        )
