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
    - ``Fraction`` renders to its exact rational literal (``"1/2"``).
    - A non-finite ``float`` (inf/-inf/nan) renders to its ``str`` form;
      raw it would emit non-spec JSON tokens.
    - dicts, lists, and tuples are recursed (a tuple becomes a list, the
      JSON-native shape); a non-string dict KEY is coerced to ``str`` (JSON
      object keys are strings, and a Fraction/int key would otherwise crash
      ``json.dumps``); a ``set``/``frozenset`` becomes a sorted-where-
      possible list (JSON has no set type); every other value passes through
      unchanged (json.dumps validates the rest).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, Fraction):
        return format_sexpr(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = k if isinstance(k, str) else _safe_key(k)
            out[key] = json_safe(v)
        return out
    if isinstance(value, (set, frozenset)):
        items = [json_safe(v) for v in value]
        try:
            return sorted(items, key=lambda x: (str(type(x)), x))
        except TypeError:
            return items
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def _safe_key(key: Any) -> str:
    """Coerce a non-string dict key to a JSON object key (a string).

    A bool key keeps Python's ``True``/``False`` spelling; a Fraction key
    uses its rational literal; everything else uses ``str``.
    """
    if isinstance(key, bool):
        return str(key)
    if isinstance(key, Fraction):
        return format_sexpr(key)
    return str(key)
