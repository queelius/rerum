"""Verify the MCP server contains no domain logic (Section 0 swap test).

No domain operator symbol (dd, int, lim, and, or as calculus/boolean
literals) may appear as a special case in any rerum/mcp/ source file.

This is the lock-in guardrail for the general-engine principle: the MCP
layer is a thin orchestration surface over a domain-agnostic rewriting
engine. Operators arrive as DATA (rules, theories, caller goals); the
server special-cases none of them. If this test fails it has caught an
executable special-case leak -- fix the offending module, do not weaken
the test.
"""

import pathlib
import re

import pytest

MCP_DIR = pathlib.Path(__file__).resolve().parent.parent / "mcp"

# Quoted operator literals that would betray a hardcoded domain. These are
# the canonical calculus/boolean operators used throughout the test corpus
# and docs as CALLER data; none of them may drive control flow in the MCP
# source. We scan for the quoted-string form (both quote styles) because a
# domain special-case necessarily writes the operator as a string literal
# (``if op == "int":``, ``"dd" in expr``, a dispatch table keyed on
# ``"lim"``, etc.). Bare mentions in prose/comments are stripped before the
# scan so a docstring example like ``{"op_free": ["int", "lim"]}`` does not
# trip the guard.
DOMAIN_TOKENS = [
    r'"dd"', r"'dd'",
    r'"int"', r"'int'",
    r'"lim"', r"'lim'",
    r'"diff"', r"'diff'",
    r'"and"', r"'and'",
    r'"or"', r"'or'",
    r'"not"', r"'not'",
]


def _strip_comments_and_docstrings(text):
    """Best-effort removal of comments and triple-quoted strings.

    A domain leak that actually drives behavior is an executable string
    literal in real code, not a comment or a docstring example. We strip
    line comments and triple-quoted blocks so caller-data examples in
    docstrings (e.g. ``{"op_free": ["int", "lim"]}``) do not produce false
    positives, while a genuine ``if op == "int":`` in code is still caught.
    """
    # Remove triple-quoted strings (docstrings and multi-line examples).
    text = re.sub(r'"""(?:.|\n)*?"""', "", text)
    text = re.sub(r"'''(?:.|\n)*?'''", "", text)
    # Remove line comments.
    text = re.sub(r"#.*", "", text)
    return text


@pytest.mark.parametrize("path", sorted(MCP_DIR.glob("*.py")))
def test_no_domain_operator_literals(path):
    """No mcp/*.py file may use a domain operator as a code literal."""
    text = _strip_comments_and_docstrings(path.read_text(encoding="utf-8"))
    for token in DOMAIN_TOKENS:
        assert not re.search(token, text), (
            f"{path.name} contains domain operator literal {token} in code; "
            f"the MCP server must be domain-agnostic (rules, theories, and "
            f"caller goals are DATA -- the server special-cases no operator)."
        )


def test_op_free_goal_operator_names_come_from_caller():
    """``_compile_goal`` reads operator names from the caller's goal dict.

    The goal-directed search tool holds no operator literal; the names that
    define the goal come entirely from the caller. A predicate built for
    ``op_free`` over ``["int", "lim"]`` rejects an expression that still
    contains one of those operators and accepts one that does not.
    """
    from rerum.mcp.tools import _compile_goal

    pred = _compile_goal({"op_free": ["int", "lim"]})
    # Predicate fired on an expression containing those ops returns False.
    assert pred(["int", ["sin", "x"], "x"]) is False
    assert pred(["lim", "x", 0, ["/", 1, "x"]]) is False
    assert pred(["+", "x", 1]) is True


def test_guard_catches_a_planted_domain_literal(tmp_path):
    """Sanity-check the guard: a planted literal MUST trip it.

    This proves the scan is real -- if someone reintroduced a domain
    special-case such as ``if head == "int":`` the parametrized test above
    would catch it. We synthesize an offending source string and assert the
    same scan logic flags it, so the guardrail cannot silently rot into a
    no-op.
    """
    planted = 'def handle(expr):\n    if expr[0] == "int":\n        return 1\n'
    stripped = _strip_comments_and_docstrings(planted)
    assert any(re.search(tok, stripped) for tok in DOMAIN_TOKENS)


def test_docstring_example_does_not_trip_the_guard():
    """A caller-data example inside a docstring is NOT a leak.

    The strip step removes docstrings before scanning, so the canonical
    ``{"op_free": ["int", "lim"]}`` usage example does not produce a false
    positive. This pins the false-positive boundary so the guard stays
    precise (catches code, ignores prose).
    """
    docstring_src = (
        'def tool(goal):\n'
        '    """Goal is DATA, e.g. {"op_free": ["int", "lim"]}."""\n'
        '    return goal\n'
    )
    stripped = _strip_comments_and_docstrings(docstring_src)
    assert not any(re.search(tok, stripped) for tok in DOMAIN_TOKENS)
