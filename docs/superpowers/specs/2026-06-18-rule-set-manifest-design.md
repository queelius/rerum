# Rule-set manifest: self-describing rule files

**Status:** approved design (user sign-off 2026-06-12), ready to implement
**Date:** 2026-06-18

## Motivation

Every example domain's loading contract -- which prelude bundles it needs,
which custom fold ops, which theory, which metadata sidecar, how it is meant
to be driven (simplify vs solve) -- currently lives only in `.rules` file
COMMENTS and is re-implemented by hand in each test loader
(`make_diff_engine`, `_integration_engine`, `_limits_engine`, ...). Nothing
can auto-assemble a domain or fail loudly when a required fold op is missing.
Two concrete pains the review verified:

1. A skeleton `(! op ...)` whose `op` is absent from the prelude emits silent
   JUNK (`(subst 1 a b)` survives as a literal compound) rather than erroring.
   A guard `(! op ...)` with a missing op errors only at first apply, not load.
2. The MCP server cannot load the bundled example rule FILES; an agent must
   paste rule text or pre-seed the `.rerum` store.

A machine-readable manifest -- riding the existing `:`-directive namespace
that already hosts `:include` -- lets the engine assemble a domain from a
single file and audit it.

## The six directives

All live in the DSL `:`-directive namespace, usable in any `.rules`/`.manifest`
file (a manifest is just a DSL file that carries directives, typically with an
`:include` of the rule body). Each is optional; repeats accumulate for the
list-valued ones.

- `:requires <bundle> [<bundle> ...]` -- named prelude bundles from
  `PRELUDE_BUNDLES` (none/minimal/arithmetic/math/predicate/full), combined
  left-to-right.
- `:requires-ops <op> [<op> ...]` -- fold-op names the rules need. A
  DECLARATION the assembler verifies are present after preludes are combined
  (so a manifest can assert "I need op X" and fail loudly if no bundle
  provides it). Custom ops that live only in an example `.py` module are not
  installable by a path-restricted manifest -- a manifest declaring such an
  op fails honestly, and that domain is loaded the manual way.
- `:theory <path>` -- a theory JSON file (relative to the manifest), parsed
  via `Theory.from_json` and installed as the session theory.
- `:metadata <path>` -- a metadata sidecar JSON (relative), merged via
  `load_metadata_json` after rules load.
- `:driver simplify|solve` -- a HINT (data a caller may read); the engine
  does not act on it.
- `:goal <json>` -- a goal-description HINT (data); e.g. `{"op_free": ["int"]}`.

## API

- `RuleSetManifest` (frozen dataclass in a new `rerum/manifest.py`): fields
  `requires: tuple[str,...]`, `requires_ops: tuple[str,...]`,
  `theory: str|None`, `metadata: str|None`, `driver: str|None`,
  `goal: dict|None`. Plus `is_empty` (no directive present).
- `parse_manifest(text) -> RuleSetManifest` (pure, in `manifest.py`): scans
  directive lines, ignores rule/group/`:include` lines, raises `ValueError`
  on a malformed directive (unknown `:driver` value, non-JSON `:goal`,
  unknown bundle name in `:requires`).
- `RuleEngine.from_manifest(path) -> RuleEngine` (classmethod): the full
  assembly --
  1. parse the file's manifest;
  2. install the combined `:requires` preludes via `with_prelude`;
  3. install `:theory` via the new `with_theory`;
  4. load the file's rules (its `:include`d body and any inline rules) with
     `validate_examples=False`;
  5. merge the `:metadata` sidecar with `validate_examples=True` (so examples
     validate against the now-assembled prelude);
  6. FAIL-LOUD AUDIT: collect every `(! op ...)` head across all rule
     skeletons AND guard conditions, plus every `:requires-ops` name; raise a
     `ValueError` naming every op absent from the assembled prelude.
  7. store the manifest (driver/goal hints) on the engine.
- `RuleEngine.with_theory(theory) -> RuleEngine` (new public setter; sets
  `_theory`, invalidates the cached simplifier; the review flagged its
  absence -- the MCP `load_theory` tool currently pokes `_theory` directly).
- `RuleEngine.missing_fold_ops() -> list[str]` (new): the audit as a public
  method over the currently-loaded rules + installed prelude -- usable
  outside manifests to catch the silent-junk footgun.
- `engine.manifest: RuleSetManifest|None` -- set by `from_manifest`, and by
  plain `load_file` when the loaded file carried directives.

## Boundaries (the approved decisions)

- **Plain `load_file` applies NOTHING from the manifest.** It parses and
  stores `engine.manifest` (so a caller can inspect a file's declared
  contract) but installs no prelude/theory and merges no sidecar. Loading a
  file must not silently mutate the engine's prelude. Assembly happens only
  via the explicit `from_manifest`. This keeps `load_file` backward
  compatible.
- **The audit FAILS LOUD, not warns.** `from_manifest` raises naming every
  missing op. Files without directives are untouched (empty manifest, no
  audit beyond what already happens).
- **Directive parsing must not break existing files.** Files that use only
  `:include` and `[group]` parse to an empty manifest. The loader must skip
  the new `:`-directives instead of mis-parsing them as rules.

## Tests (`rerum/tests/test_manifest.py`)

- `parse_manifest`: each directive parsed; repeats accumulate; unknown
  `:driver`/bad `:goal` JSON/unknown bundle -> `ValueError`; a directive-free
  file -> empty manifest.
- `from_manifest` end-to-end on a real example: a new
  `examples/differentiation.manifest` (`:requires math predicate`,
  `:theory arithmetic.theory.json`, `:metadata differentiation.metadata.json`,
  `:driver simplify`, `:include differentiation.rules`) assembles an engine
  that differentiates correctly -- replacing `make_diff_engine`'s hand wiring.
- A `boolean.manifest` (`:requires none`, theory, metadata, `:include
  boolean.rules`) -- the no-prelude path.
- Missing-op audit: a manifest whose rules use `(! frobnicate ...)` with no
  bundle providing it raises `ValueError` naming `frobnicate`; a guard-only
  `(! ...)` missing op is also caught (load-time, not first-apply).
- `:requires-ops` declaring an absent op fails even if no rule uses it yet.
- `load_file` of a manifest installs NO prelude (BC) but sets
  `engine.manifest`.
- `with_theory` / `missing_fold_ops` unit tests.

## Out of scope (follow-ons)

- CLI flag to run a manifest; MCP tool to load a manifest by name from a
  restricted directory (solves the "MCP can't load example files" finding --
  path-restricted, so safe).
- A `:driver`/`:goal`-aware run helper. v1 stores them as data only.
