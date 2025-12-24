"""Tests for CLI module."""

import subprocess
import sys
from pathlib import Path
import pytest

from rerum.cli import RerumREPL, ScriptRunner, load_custom_prelude, BUILTIN_PRELUDES


class TestBuiltinPreludes:
    """Tests for built-in prelude names."""

    def test_builtin_prelude_names(self):
        """All expected built-in preludes exist."""
        expected = {"none", "minimal", "arithmetic", "math", "predicate", "full"}
        assert set(BUILTIN_PRELUDES.keys()) == expected


class TestREPLCommands:
    """Tests for REPL command handling."""

    def test_help_command(self):
        """Help command returns help text."""
        repl = RerumREPL()
        result = repl.handle_command(":help")
        assert "help" in result.lower()
        assert "load" in result.lower()

    def test_prelude_command(self):
        """Prelude command sets prelude."""
        repl = RerumREPL()
        result = repl.handle_command(":prelude arithmetic")
        assert "arithmetic" in result.lower()
        assert repl.prelude is not None

    def test_unknown_prelude(self):
        """Unknown prelude returns error."""
        repl = RerumREPL()
        result = repl.handle_command(":prelude nonexistent")
        assert "Unknown" in result

    def test_trace_command(self):
        """Trace command toggles tracing."""
        repl = RerumREPL()
        assert repl.trace == False

        result = repl.handle_command(":trace on")
        assert repl.trace == True
        assert "enabled" in result.lower()

        result = repl.handle_command(":trace off")
        assert repl.trace == False
        assert "disabled" in result.lower()

    def test_trace_toggle(self):
        """Trace command without arg toggles."""
        repl = RerumREPL()
        repl.handle_command(":trace")
        assert repl.trace == True
        repl.handle_command(":trace")
        assert repl.trace == False

    def test_strategy_command(self):
        """Strategy command sets strategy."""
        repl = RerumREPL()
        result = repl.handle_command(":strategy bottomup")
        assert repl.strategy == "bottomup"
        assert "bottomup" in result.lower()

    def test_unknown_strategy(self):
        """Unknown strategy returns error."""
        repl = RerumREPL()
        result = repl.handle_command(":strategy invalid")
        assert "Unknown" in result

    def test_clear_command(self):
        """Clear command removes all rules."""
        repl = RerumREPL()
        repl.process_line("@test: (f ?x) => :x")
        assert len(repl.engine) == 1

        repl.handle_command(":clear")
        assert len(repl.engine) == 0

    def test_rules_command_empty(self):
        """Rules command with no rules."""
        repl = RerumREPL()
        result = repl.handle_command(":rules")
        assert "No rules" in result

    def test_rules_command_with_rules(self):
        """Rules command lists rules."""
        repl = RerumREPL()
        repl.process_line("@test: (f ?x) => :x")
        result = repl.handle_command(":rules")
        assert "test" in result

    def test_quit_command(self):
        """Quit command sets running to False."""
        repl = RerumREPL()
        assert repl.running == True
        repl.handle_command(":quit")
        assert repl.running == False

    def test_groups_command_empty(self):
        """Groups command with no groups."""
        repl = RerumREPL()
        result = repl.handle_command(":groups")
        assert "No groups" in result

    def test_enable_disable_group(self):
        """Enable/disable group commands work."""
        repl = RerumREPL()
        repl.handle_command(":disable testgroup")
        assert "testgroup" in repl.engine._disabled_groups

        repl.handle_command(":enable testgroup")
        assert "testgroup" not in repl.engine._disabled_groups


class TestREPLProcessLine:
    """Tests for REPL line processing."""

    def test_empty_line(self):
        """Empty line returns None."""
        repl = RerumREPL()
        assert repl.process_line("") is None
        assert repl.process_line("   ") is None

    def test_comment_line(self):
        """Comment line returns None."""
        repl = RerumREPL()
        assert repl.process_line("# comment") is None

    def test_rule_definition(self):
        """Rule definition adds rule."""
        repl = RerumREPL()
        result = repl.process_line("@add-zero: (+ ?x 0) => :x")
        assert "Added" in result
        assert len(repl.engine) == 1

    def test_expression_evaluation(self):
        """Expression is evaluated."""
        repl = RerumREPL()
        repl.process_line("@add-zero: (+ ?x 0) => :x")
        result = repl.process_line("(+ y 0)")
        assert result == "y"

    def test_expression_unchanged(self):
        """Expression that doesn't match returns unchanged."""
        repl = RerumREPL()
        result = repl.process_line("(f x)")
        assert result == "(f x)"


class TestScriptRunner:
    """Tests for script execution."""

    def test_run_expression(self):
        """Run single expression."""
        runner = ScriptRunner()
        runner.repl.process_line("@add-zero: (+ ?x 0) => :x")

        # Capture output by checking return code
        # (expression output goes to stdout)
        code = runner.run_expression("(+ y 0)")
        assert code == 0


class TestCLIIntegration:
    """Integration tests using subprocess."""

    def test_help_flag(self):
        """--help flag works."""
        result = subprocess.run(
            [sys.executable, "-m", "rerum.cli", "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "RERUM" in result.stdout

    def test_version_flag(self):
        """--version flag works."""
        result = subprocess.run(
            [sys.executable, "-m", "rerum.cli", "--version"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_expression_mode(self):
        """Expression mode evaluates expression."""
        result = subprocess.run(
            [sys.executable, "-m", "rerum.cli", "-e", "(+ 1 2)"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        # Without rules, expression is unchanged
        assert "(+ 1 2)" in result.stdout

    def test_expression_with_prelude(self):
        """Expression with prelude evaluates."""
        result = subprocess.run(
            [sys.executable, "-m", "rerum.cli", "-p", "full", "-e",
             "@fold: (+ ?a ?b) => (! + :a :b) when (! and (! const? :a) (! const? :b))"],
            capture_output=True, text=True
        )
        # Just defining a rule should work
        assert result.returncode == 0

    def test_pipe_mode(self):
        """Pipe mode processes stdin."""
        result = subprocess.run(
            [sys.executable, "-m", "rerum.cli", "-q"],
            input="(f x)\n(g y)\n",
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "(f x)" in result.stdout
        assert "(g y)" in result.stdout


class TestCustomPreludeLoading:
    """Tests for custom prelude loading."""

    def test_load_nonexistent_prelude(self):
        """Loading nonexistent prelude returns None."""
        result = load_custom_prelude("/nonexistent/path.py")
        assert result is None

    def test_load_builtin_prelude_names(self):
        """Built-in names should not trigger file search."""
        repl = RerumREPL()
        assert repl.set_prelude("full") == True
        assert repl.set_prelude("arithmetic") == True
        assert repl.set_prelude("none") == True


class TestMultiLineInput:
    """Tests for multi-line input parsing."""

    def test_count_parens_balanced(self):
        """Balanced parens return 0."""
        from rerum.cli import count_parens
        assert count_parens("(+ x 1)") == 0
        assert count_parens("(+ (+ x 1) 2)") == 0
        assert count_parens("x") == 0

    def test_count_parens_unbalanced_open(self):
        """More open parens return positive count."""
        from rerum.cli import count_parens
        assert count_parens("(+ x") == 1
        assert count_parens("(+ (+ x") == 2
        assert count_parens("(") == 1

    def test_count_parens_unbalanced_close(self):
        """More close parens return negative count."""
        from rerum.cli import count_parens
        assert count_parens("(+ x))") == -1
        assert count_parens(")") == -1

    def test_count_parens_ignores_strings(self):
        """Parens inside strings are ignored."""
        from rerum.cli import count_parens
        # Quotes inside expressions would be in string literals
        assert count_parens('(f ")" x)') == 0
        assert count_parens('(f "(" x)') == 0

    def test_repl_multi_line_buffer(self):
        """REPL has multi-line buffer initialized."""
        repl = RerumREPL()
        assert repl.multi_line_buffer == ""


class TestTabCompletion:
    """Tests for tab completion."""

    def test_completer_commands(self):
        """Completer suggests commands."""
        from rerum.cli import RerumCompleter
        repl = RerumREPL()
        completer = RerumCompleter(repl)

        matches = completer._get_matches(":", ":")
        assert ":help" in matches
        assert ":quit" in matches
        assert ":load" in matches

    def test_completer_partial_command(self):
        """Completer handles partial command."""
        from rerum.cli import RerumCompleter
        repl = RerumREPL()
        completer = RerumCompleter(repl)

        matches = completer._get_matches(":h", ":h")
        assert ":help" in matches
        assert ":quit" not in matches

    def test_completer_prelude_names(self):
        """Completer suggests prelude names after :prelude."""
        from rerum.cli import RerumCompleter
        repl = RerumREPL()
        completer = RerumCompleter(repl)

        matches = completer._get_matches("", ":prelude ")
        assert "full" in matches
        assert "arithmetic" in matches
        assert "none" in matches

    def test_completer_strategy_names(self):
        """Completer suggests strategies after :strategy."""
        from rerum.cli import RerumCompleter
        repl = RerumREPL()
        completer = RerumCompleter(repl)

        matches = completer._get_matches("", ":strategy ")
        assert "exhaustive" in matches
        assert "once" in matches
        assert "bottomup" in matches
        assert "topdown" in matches

    def test_completer_groups(self):
        """Completer suggests groups after :enable/:disable."""
        from rerum.cli import RerumCompleter
        repl = RerumREPL()

        # Add some rules with groups
        repl.engine.load_dsl('''
            [algebra]
            @add-zero: (+ ?x 0) => :x

            [calculus]
            @dd-const: (dd ?c:const ?v) => 0
        ''')

        completer = RerumCompleter(repl)

        matches = completer._get_matches("", ":enable ")
        assert "algebra" in matches
        assert "calculus" in matches

    def test_completer_rule_names(self):
        """Completer suggests rule names starting with @."""
        from rerum.cli import RerumCompleter
        repl = RerumREPL()

        repl.engine.load_dsl('''
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
        ''')

        completer = RerumCompleter(repl)

        matches = completer._get_matches("@", "@")
        assert "@add-zero" in matches
        assert "@mul-one" in matches

    def test_completer_trace_options(self):
        """Completer suggests on/off after :trace."""
        from rerum.cli import RerumCompleter
        repl = RerumREPL()
        completer = RerumCompleter(repl)

        matches = completer._get_matches("", ":trace ")
        assert "on" in matches
        assert "off" in matches

