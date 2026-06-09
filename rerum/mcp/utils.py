"""Shared MCP-layer utilities.

``json_safe`` is the single value-sanitizer for everything that crosses the
MCP transport: every tool response and error payload is ``json.dumps``-ed,
but engine values are not all JSON-native. A ``fractions.Fraction`` (exact
rational arithmetic) has broken serialization repeatedly, and a non-finite
float would serialize as the non-spec ``Infinity``/``NaN`` tokens that
strict JSON parsers reject.
"""

import math
from fractions import Fraction
from typing import Any

from rerum.expr import format_sexpr


def json_safe(value: Any) -> Any:
    """Recursively make a value JSON-serializable for an MCP payload.

    - ``bool`` is preserved (checked before any numeric branch, since bool
      subclasses int).
    - ``Fraction`` renders to its exact s-expr string (``"(/ 1 2)"``).
    - A non-finite ``float`` (inf/-inf/nan) renders to its ``str`` form;
      raw it would emit non-spec JSON tokens.
    - dicts and lists are recursed; every other value passes through
      unchanged (json.dumps validates the rest).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, Fraction):
        return format_sexpr(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value
