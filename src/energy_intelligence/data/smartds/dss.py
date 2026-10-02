"""Generic OpenDSS ``.dss`` directive parser.

OpenDSS is the network model format SMART-DS distributes. The parser is
deliberately **dataset-agnostic**: it knows the grammar of a ``.dss`` file and
nothing about SMART-DS. All SMART-DS-specific knowledge lives in the adapter
layer above this module, so a future dataset using OpenDSS would reuse this
parser unchanged.

Grammar actually observed in SMART-DS files
-------------------------------------------

Directives are whitespace-insensitive and take the form
``<Verb> <Class>.<name> key=value key=value ...``. Elements are separated by
blank lines or by a new ``New`` directive. Values may contain spaces and ``=``
signs (for example a loadshape's ``mult = (file=../../x.csv)``), so splitting is
done by scanning for ``key=`` boundaries rather than splitting on whitespace.

The parser is strict: a malformed directive raises rather than being silently
skipped, because a silently skipped ``New Storage.*`` line would mean a battery
vanished from the model without anyone noticing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "DssObject",
    "DssParseError",
    "parse_dss",
    "split_directives",
    "KEY_VALUE_RE",
]

#: A ``key=value`` boundary. The key is a bare word, optionally prefixed with
#: ``%`` -- OpenDSS uses percent-prefixed parameter names such as
#: ``%Cutout``, ``%Cutin``, ``%EffCharge`` and ``%EffDischarge``, and omitting the
#: ``%`` silently swallows the rest of the directive into the previous value.
#:
#: A negative number, or a value containing ``=`` (for example a loadshape's
#: ``mult = (file=../../x.csv)``), must not be mistaken for a key, so a key must be
#: preceded by whitespace or start of input, and must be followed by ``=``.
_KEY_VALUE_RE = re.compile(r"(?:^|\s)(?P<key>%?[A-Za-z_][A-Za-z0-9_.%]*)\s*=\s*")

class DssParseError(ValueError):
    """Raised when a ``.dss`` file cannot be parsed faithfully."""


@dataclass(frozen=True, slots=True)
class DssObject:
    """One instantiated element.

    Attributes:
        element_class: The OpenDSS class, e.g. ``"Load"``.
        name: The element name after the dot, e.g. ``"load_p1ulv279_1"``.
        params: Parameter name to raw string value. Keys preserve the source
            casing; lookups are therefore case-sensitive by default and
            :meth:`get` performs a case-insensitive fallback.
        order: Parameters in source order, so a directive can be reproduced
            byte-for-byte.
    """

    element_class: str
    name: str
    params: dict[str, str] = field(default_factory=dict)
    order: tuple[str, ...] = ()

    def get(self, key: str, default: str | None = None) -> str | None:
        """Return a parameter, case-insensitively, or ``default``."""
        if key in self.params:
            return self.params[key]
        lowered = key.lower()
        for name, value in self.params.items():
            if name.lower() == lowered:
                return value
        return default

    def require(self, key: str) -> str:
        """Return a parameter or raise, rather than returning a silent ``None``."""
        value = self.get(key)
        if value is None:
            raise DssParseError(
                f"{self.element_class}.{self.name} is missing required parameter {key!r}; "
                f"present: {sorted(self.params)}"
            )
        return value

    def get_float(self, key: str, default: float | None = None) -> float | None:
        """Parse a parameter as float, or return ``default``.

        Raises:
            DssParseError: If the value is present but not a real number. A
                silently defaulted numeric value would corrupt every
                downstream energy total.
        """
        raw = self.get(key)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError as exc:
            raise DssParseError(
                f"{self.element_class}.{self.name} parameter {key!r} is not a real "
                f"number: {raw!r}"
            ) from exc

    def get_int(self, key: str, default: int | None = None) -> int | None:
        """Parse a parameter as int, or return ``default``."""
        raw = self.get(key)
        if raw is None:
            return default
        try:
            return int(float(raw))
        except ValueError as exc:
            raise DssParseError(
                f"{self.element_class}.{self.name} parameter {key!r} is not an "
                f"integer: {raw!r}"
            ) from exc

    @property
    def qualified_name(self) -> str:
        return f"{self.element_class}.{self.name}"


def split_directives(text: str) -> list[tuple[str, str]]:
    """Split file text into ``(verb, remainder)`` pairs.

    A new element directive implicitly terminates the previous one, because a
    ``.dss`` element ends where the next one begins even without a blank line.

    Args:
        text: Full file contents.

    Returns:
        Ordered list of ``(verb, remainder)``.

    Raises:
        DssParseError: If the first non-comment directive has no verb.
    """
    directives: list[tuple[str, str]] = []
    current_verb: str | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_verb is not None and current_verb == "new":
                directives.append((current_verb, " ".join(current_lines)))
                current_verb, current_lines = None, []
            continue
        if line.startswith("//") or line.startswith("!"):
            continue

        verb_match = re.match(r"^(?P<verb>[A-Za-z_][A-Za-z0-9_.]*)\b\s*(?P<rest>.*)$", line)
        if verb_match is None:
            if current_verb == "new":
                current_lines.append(line)
                continue
            raise DssParseError(f"cannot parse directive line: {line!r}")

        verb = verb_match.group("verb")
        rest = verb_match.group("rest")

        if current_verb == "new" and verb != "new":
            directives.append((current_verb, " ".join(current_lines)))
            current_verb, current_lines = None, []

        if verb == "new":
            if current_verb == "new":
                directives.append((current_verb, " ".join(current_lines)))
            current_verb, current_lines = "new", [rest]
        else:
            directives.append((verb.lower(), rest))

    if current_verb == "new" and current_lines:
        directives.append((current_verb, " ".join(current_lines)))
    return directives


def _parse_params(remainder: str) -> tuple[str, dict[str, str], tuple[str, ...]]:
    """Extract the element name and key/value pairs from a directive remainder."""
    head_match = re.match(r"^(?P<qualified>\S+)\s*(?P<rest>.*)$", remainder)
    if head_match is None:
        raise DssParseError(f"element directive has no name: {remainder!r}")

    qualified = head_match.group("qualified")
    if "." not in qualified:
        raise DssParseError(
            f"element name {qualified!r} is not of the form Class.name"
        )
    element_class, _, name = qualified.partition(".")

    rest = head_match.group("rest")
    matches = list(_KEY_VALUE_RE.finditer(rest))
    if not matches:
        return element_class, name, {}, ()

    params: dict[str, str] = {}
    order: list[str] = []
    for index, match in enumerate(matches):
        key = match.group("key")
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(rest)
        value = rest[start:end].strip()
        # A trailing comma is OpenDSS's field separator in some exports.
        if value.endswith(","):
            value = value[:-1].strip()
        if key not in params:
            order.append(key)
        params[key] = value

    return element_class, name, params, tuple(order)


def parse_dss(text: str) -> list[DssObject]:
    """Parse a ``.dss`` file into element objects.

    Non-element directives (``Redirect``, ``Clear``, ``Set``, ``Solve``, ...) are
    skipped: they describe execution rather than data.

    Args:
        text: Full file contents.

    Returns:
        Elements in source order.

    Raises:
        DssParseError: On any malformed directive or missing element name.
    """
    objects: list[DssObject] = []
    for verb, remainder in split_directives(text):
        if verb != "new":
            continue
        if not remainder.strip():
            raise DssParseError("empty element directive")
        element_class, name, params, order = _parse_params(remainder)
        objects.append(
            DssObject(
                element_class=element_class,
                name=name,
                params=params,
                order=order,
            )
        )
    return objects
