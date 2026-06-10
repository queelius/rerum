"""Tool registry: the single source of truth for the MCP tool surface.

Everything is DERIVED FROM STRUCTURE (no hand-maintained parallel tables):

- DISCOVERY: every callable named ``tool_*`` in ``rerum.mcp.tools`` is a
  tool; the name is the prefix-stripped function name.
- DEPENDENCIES: a handler's positional (pre-``*``) parameters are
  server-injected dependencies (``engine``, ``store``, ``sampler``),
  recorded by name for the dispatcher's provider map.
- SCHEMA: the keyword-only parameters are the caller's inputs. Their type
  annotations generate the JSON ``inputSchema`` (``Literal`` becomes an
  enum, ``Optional[T]`` unwraps to T, defaults mark a parameter optional),
  and their descriptions come from the Google-style ``Args:`` section of
  the handler docstring.
- VALIDATION: ``validate_and_coerce`` enforces the derived schema at
  dispatch (unknown/missing parameters are a clear ``parse_error``; string
  arguments are coerced to the annotated numeric/bool types as
  defense-in-depth -- a schema is advisory to the client, not enforced by
  the transport).

Adding a tool is therefore: write one annotated, docstringed ``tool_*``
function. Registry, schema, dispatch, and client-facing docs all follow.
"""

import inspect
import typing
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from rerum.mcp.errors import MCPToolError

_MISSING = object()


# =====================================================================
# Annotation -> JSON schema
# =====================================================================

_SCALARS = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}


def _permit_null(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Widen a schema to also accept JSON ``null`` (for ``Optional[T]``).

    The server's ``validate_and_coerce`` accepts ``None`` for every
    optional parameter, and the SDK validates calls against the EMITTED
    schema before the handler runs -- so the schema must permit null too,
    or a client passing ``{"metric": null}`` (a common LLM habit) is
    rejected over the wire but accepted in-process. ``oneOf``/``anyOf``
    gain a null member; an ``enum`` gains ``None``; a ``type`` gains
    ``"null"``; a typeless schema already accepts anything.
    """
    if "oneOf" in schema:
        return {**schema, "oneOf": schema["oneOf"] + [{"type": "null"}]}
    if "anyOf" in schema:
        return {**schema, "anyOf": schema["anyOf"] + [{"type": "null"}]}
    out = dict(schema)
    if "enum" in out and None not in out["enum"]:
        out["enum"] = list(out["enum"]) + [None]
    t = out.get("type")
    if isinstance(t, str):
        out["type"] = [t, "null"]
    elif isinstance(t, list) and "null" not in t:
        out["type"] = t + ["null"]
    return out


def _annotation_to_schema(annotation: Any) -> Dict[str, Any]:
    """Translate a Python type annotation into a JSON-schema fragment.

    Covers the closed set the tool handlers use: scalars, ``Literal``
    (enum), ``Optional[T]`` (unwraps T but PERMITS null, matching
    validate_and_coerce), ``List[T]``, ``Dict[str, T]``, and ``Union[...]``
    (oneOf). An unrecognized annotation yields the permissive ``{}`` with a
    build-time warning rather than failing registration.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    if annotation in _SCALARS:
        return dict(_SCALARS[annotation])

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Literal:
        enum = list(args)
        schema: Dict[str, Any] = {"enum": enum}
        if all(isinstance(v, str) for v in enum):
            schema["type"] = "string"
        return schema

    if origin is typing.Union:
        has_none = type(None) in args
        members = [a for a in args if a is not type(None)]
        if len(members) == 1:  # Optional[T]
            base = _annotation_to_schema(members[0])
            return _permit_null(base) if has_none else base
        schemas = [_annotation_to_schema(m) for m in members]
        if has_none:
            schemas.append({"type": "null"})
        return {"oneOf": schemas}

    if origin in (list, List):
        items = _annotation_to_schema(args[0]) if args else {}
        out: Dict[str, Any] = {"type": "array"}
        if items:
            out["items"] = items
        return out

    if origin in (dict, Dict):
        out = {"type": "object"}
        if len(args) == 2 and args[1] is not Any:
            value_schema = _annotation_to_schema(args[1])
            if value_schema:
                out["additionalProperties"] = value_schema
        return out

    warnings.warn(
        f"registry: no JSON-schema mapping for annotation {annotation!r}; "
        "emitting a permissive schema",
        stacklevel=2,
    )
    return {}


# =====================================================================
# Docstring parsing (summary + Args: descriptions)
# =====================================================================

def _parse_docstring(fn: Callable) -> Tuple[str, Dict[str, str]]:
    """Extract ``(summary, {param: description})`` from a handler docstring.

    The summary is the text up to the first blank line. Parameter
    descriptions come from a Google-style ``Args:`` section: a parameter
    line is ``name: text`` at one indent level; deeper-indented lines are
    continuations. The parser is deliberately minimal -- it covers the
    convention the handlers follow, nothing more.
    """
    doc = inspect.getdoc(fn) or ""
    lines = doc.splitlines()

    summary_lines: List[str] = []
    for line in lines:
        if not line.strip():
            break
        summary_lines.append(line.strip())
    summary = " ".join(summary_lines)

    arg_docs: Dict[str, str] = {}
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == "Args:")
    except StopIteration:
        return summary, arg_docs

    current: Optional[str] = None
    for line in lines[start + 1:]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            break  # next section ("Returns:", prose, ...)
        stripped = line.strip()
        head, sep, rest = stripped.partition(": ")
        is_param_line = (
            sep and indent <= 4 and head.isidentifier()
        )
        if is_param_line:
            current = head
            arg_docs[current] = rest.strip()
        elif current is not None:
            arg_docs[current] += " " + stripped
    return summary, arg_docs


# =====================================================================
# Specs
# =====================================================================

@dataclass(frozen=True)
class ParamSpec:
    """One keyword-only (caller-facing) parameter of a tool."""

    name: str
    annotation: Any
    required: bool
    default: Any
    description: str
    json_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    """One tool: handler, injected dependencies, params, and schema."""

    name: str
    handler: Callable
    deps: Tuple[str, ...]
    params: Tuple[ParamSpec, ...]
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)

    def param(self, name: str) -> Optional[ParamSpec]:
        for p in self.params:
            if p.name == name:
                return p
        return None


def _build_tool_spec(name: str, fn: Callable) -> ToolSpec:
    sig = inspect.signature(fn)
    summary, arg_docs = _parse_docstring(fn)

    deps: List[str] = []
    params: List[ParamSpec] = []
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            deps.append(p.name)
        elif p.kind is inspect.Parameter.KEYWORD_ONLY:
            required = p.default is inspect.Parameter.empty
            schema = _annotation_to_schema(p.annotation)
            description = arg_docs.get(p.name, "")
            if description:
                schema = {**schema, "description": description}
            if not required and p.default is not None and isinstance(
                    p.default, (str, int, float, bool)):
                schema = {**schema, "default": p.default}
            params.append(ParamSpec(
                name=p.name,
                annotation=p.annotation,
                required=required,
                default=(_MISSING if required else p.default),
                description=description,
                json_schema=schema,
            ))

    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {p.name: p.json_schema for p in params},
        "additionalProperties": False,
    }
    required_names = [p.name for p in params if p.required]
    if required_names:
        input_schema["required"] = required_names

    return ToolSpec(
        name=name,
        handler=fn,
        deps=tuple(deps),
        params=tuple(params),
        description=summary,
        input_schema=input_schema,
    )


def build_registry(tools_module) -> Dict[str, ToolSpec]:
    """Scan ``tools_module`` for ``tool_*`` callables and build their specs."""
    registry: Dict[str, ToolSpec] = {}
    for attr_name, fn in vars(tools_module).items():
        if not attr_name.startswith("tool_") or not callable(fn):
            continue
        name = attr_name[len("tool_"):]
        registry[name] = _build_tool_spec(name, fn)
    return registry


_REGISTRY: Optional[Dict[str, ToolSpec]] = None


def get_registry() -> Dict[str, ToolSpec]:
    """The lazily-built, cached registry over ``rerum.mcp.tools``."""
    global _REGISTRY
    if _REGISTRY is None:
        from rerum.mcp import tools as _tools
        _REGISTRY = build_registry(_tools)
    return _REGISTRY


# =====================================================================
# Dispatch-time validation + coercion
# =====================================================================

def _coerce_string(value: str, annotation: Any) -> Any:
    """Best-effort coercion of a string arg to its annotated scalar type.

    A schema is advisory to the client; a sloppy client may send "6" for an
    integer or "true" for a boolean. Coerce when the annotation says so; an
    uncoercible string is returned unchanged for the handler to reject.
    ``Optional[T]`` unwraps to T; non-scalar annotations pass through.
    """
    target = annotation
    origin = typing.get_origin(target)
    if origin is typing.Union:
        members = [a for a in typing.get_args(target)
                   if a is not type(None)]
        if len(members) == 1:
            target = members[0]
            origin = typing.get_origin(target)

    if target is bool:
        low = value.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        return value
    if target in (int, float):
        try:
            return target(value)
        except (TypeError, ValueError):
            return value
    return value


def validate_and_coerce(spec: ToolSpec,
                        args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ``args`` against the spec; coerce; raise parse_error early.

    - An UNKNOWN argument is rejected (a misspelled parameter silently
      running with defaults is the worst failure mode for an LLM client).
    - A MISSING required argument is rejected with the full list.
    - A string argument whose annotation is numeric/bool is coerced.
    - A ``Literal``-annotated argument must be one of the enum values.
    """
    accepted = {p.name for p in spec.params}
    unknown = sorted(set(args) - accepted)
    if unknown:
        raise MCPToolError(
            "parse_error",
            f"unknown parameter(s) {unknown} for tool {spec.name!r}; "
            f"accepted: {sorted(accepted)}",
            details={"unknown": unknown, "accepted": sorted(accepted)},
        )

    missing = [p.name for p in spec.params
               if p.required and p.name not in args]
    if missing:
        raise MCPToolError(
            "parse_error",
            f"missing required parameter(s) {missing} for tool {spec.name!r}",
            details={"missing": missing,
                     "required": [p.name for p in spec.params if p.required]},
        )

    out: Dict[str, Any] = {}
    for key, value in args.items():
        p = spec.param(key)
        if isinstance(value, str):
            value = _coerce_string(value, p.annotation)
        enum = p.json_schema.get("enum")
        # None is allowed for an Optional[Literal[...]] (it means "use the
        # engine default"), so only non-None values are enum-checked.
        if enum is not None and value is not None and value not in enum:
            raise MCPToolError(
                "parse_error",
                f"invalid value {value!r} for parameter {key!r} of tool "
                f"{spec.name!r}; allowed: {enum}",
                details={"parameter": key, "allowed": enum},
            )
        out[key] = value
    return out
