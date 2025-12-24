#!/usr/bin/env python3
"""
RERUM Command-Line Interface

Provides interactive REPL, script execution, and pipe/filter modes.

Usage:
    rerum                           # Start REPL
    rerum script.rerum              # Run script
    rerum -e "(+ x 0)"              # Evaluate expression
    rerum -r rules.rules            # REPL with rules preloaded
    rerum -r rules.rules -e "(+ x 0)"  # One-shot with rules
    echo "(+ x 0)" | rerum -r rules.rules  # Filter mode

Script Format (.rerum files):
    #!/usr/bin/env rerum
    :prelude full
    :load algebra.rules

    @add-zero: (+ ?x 0) => :x

    (+ x 0)
    (+ 1 2)

REPL Commands:
    :help              Show help
    :load FILE         Load rules from file
    :rules             List loaded rules
    :clear             Clear all rules
    :prelude NAME      Set prelude (arithmetic, math, full, none, or path)
    :trace on|off      Toggle tracing
    :strategy NAME     Set strategy (exhaustive, once, bottomup, topdown)
    :groups            Show groups
    :enable GROUP      Enable group
    :disable GROUP     Disable group
    :quit              Exit
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from .engine import RuleEngine, E, format_sexpr, parse_sexpr, load_rules_from_dsl
from .rewriter import (
    ARITHMETIC_PRELUDE, MATH_PRELUDE, FULL_PRELUDE,
    MINIMAL_PRELUDE, PREDICATE_PRELUDE, NO_PRELUDE,
    FoldFuncsType,
)

# Try to import readline for better REPL experience
try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False

# Built-in preludes
BUILTIN_PRELUDES: Dict[str, FoldFuncsType] = {
    "none": NO_PRELUDE,
    "minimal": MINIMAL_PRELUDE,
    "arithmetic": ARITHMETIC_PRELUDE,
    "math": MATH_PRELUDE,
    "predicate": PREDICATE_PRELUDE,
    "full": FULL_PRELUDE,
}

# Standard prelude search paths
PRELUDE_SEARCH_PATHS = [
    Path("./preludes"),
    Path.home() / ".config" / "rerum" / "preludes",
]


def load_custom_prelude(name_or_path: str) -> Optional[FoldFuncsType]:
    """
    Load a custom prelude from a Python file.

    The file should define a PRELUDE dict.

    Args:
        name_or_path: Either a path to a .py file, or a name to search for

    Returns:
        The PRELUDE dict from the file, or None if not found
    """
    path = Path(name_or_path)

    # If it's an explicit path
    if path.suffix == ".py" or "/" in name_or_path or "\\" in name_or_path:
        if not path.exists():
            return None
        search_paths = [path]
    else:
        # Search for name.py in standard locations
        search_paths = []
        for search_dir in PRELUDE_SEARCH_PATHS:
            candidate = search_dir / f"{name_or_path}.py"
            if candidate.exists():
                search_paths.append(candidate)

    for prelude_path in search_paths:
        if prelude_path.exists():
            try:
                spec = importlib.util.spec_from_file_location("custom_prelude", prelude_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "PRELUDE"):
                        return module.PRELUDE
            except Exception as e:
                print(f"Error loading prelude from {prelude_path}: {e}", file=sys.stderr)

    return None


class RerumCompleter:
    """Tab completer for RERUM REPL."""

    # Commands that can be completed
    COMMANDS = [
        ":help", ":quit", ":exit", ":q",
        ":load", ":rules", ":clear",
        ":prelude", ":trace", ":strategy",
        ":groups", ":enable", ":disable",
    ]

    STRATEGIES = ["exhaustive", "once", "bottomup", "topdown"]
    TRACE_OPTIONS = ["on", "off"]

    def __init__(self, repl: 'RerumREPL'):
        self.repl = repl

    def complete(self, text: str, state: int) -> Optional[str]:
        """Return the next possible completion for 'text'."""
        if state == 0:
            # Build completion list on first call
            line = readline.get_line_buffer() if HAS_READLINE else ""
            self.matches = self._get_matches(text, line)

        try:
            return self.matches[state]
        except IndexError:
            return None

    def _get_matches(self, text: str, line: str) -> list:
        """Get list of matches for the current input."""
        line = line.lstrip()

        # After :prelude, complete prelude names (check before general command completion)
        if line.startswith(":prelude "):
            prelude_names = list(BUILTIN_PRELUDES.keys())
            return [p for p in prelude_names if p.startswith(text)]

        # After :strategy, complete strategy names
        if line.startswith(":strategy "):
            return [s for s in self.STRATEGIES if s.startswith(text)]

        # After :trace, complete on/off
        if line.startswith(":trace "):
            return [t for t in self.TRACE_OPTIONS if t.startswith(text)]

        # After :enable or :disable, complete group names
        if line.startswith(":enable ") or line.startswith(":disable "):
            groups = list(self.repl.engine.groups())
            return [g for g in groups if g.startswith(text)]

        # After :load, complete file paths
        if line.startswith(":load "):
            return self._complete_path(text)

        # Command completion (only if text starts with : or line is just starting a command)
        if text.startswith(":") or (line.startswith(":") and " " not in line):
            return [c for c in self.COMMANDS if c.startswith(text)]

        # In expression context, complete rule names for reference
        if text.startswith("@"):
            rule_names = ["@" + name for name in self.repl.engine._rule_names.keys()]
            return [r for r in rule_names if r.startswith(text)]

        return []

    def _complete_path(self, text: str) -> list:
        """Complete file paths."""
        import glob

        # Handle empty text
        if not text:
            text = "./"

        # Add wildcard for glob matching
        pattern = text + "*"

        matches = []
        for path in glob.glob(pattern):
            if Path(path).is_dir():
                matches.append(path + "/")
            else:
                matches.append(path)

        return matches


def count_parens(text: str) -> int:
    """Count unbalanced parentheses. Returns >0 if more open than close."""
    depth = 0
    in_string = False
    escape = False

    for c in text:
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1

    return depth


class RerumREPL:
    """Interactive REPL for rerum."""

    def __init__(self):
        self.engine = RuleEngine()
        self.prelude: Optional[FoldFuncsType] = None
        self.trace = False
        self.strategy = "exhaustive"
        self.running = True
        self.multi_line_buffer = ""

        # Set up readline history and completion
        if HAS_READLINE:
            self.history_file = Path.home() / ".rerum_history"
            try:
                readline.read_history_file(self.history_file)
            except FileNotFoundError:
                pass
            readline.set_history_length(1000)

            # Set up tab completion
            self.completer = RerumCompleter(self)
            readline.set_completer(self.completer.complete)
            readline.parse_and_bind("tab: complete")

            # Configure completion delimiters (don't break on colons for commands)
            readline.set_completer_delims(" \t\n")

    def save_history(self):
        """Save readline history."""
        if HAS_READLINE:
            try:
                readline.write_history_file(self.history_file)
            except Exception:
                pass

    def set_prelude(self, name: str) -> bool:
        """Set the prelude by name or path."""
        name_lower = name.lower()

        if name_lower in BUILTIN_PRELUDES:
            self.prelude = BUILTIN_PRELUDES[name_lower]
            self.engine = self.engine.with_prelude(self.prelude)
            return True

        # Try to load custom prelude
        custom = load_custom_prelude(name)
        if custom is not None:
            self.prelude = custom
            self.engine = self.engine.with_prelude(self.prelude)
            return True

        return False

    def handle_command(self, line: str) -> Optional[str]:
        """
        Handle a REPL command (starts with :).

        Returns a message to print, or None.
        """
        parts = line[1:].split(None, 1)
        if not parts:
            return "Unknown command. Type :help for help."

        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "help":
            return self.help_text()

        elif cmd == "quit" or cmd == "exit" or cmd == "q":
            self.running = False
            return None

        elif cmd == "load":
            if not arg:
                return "Usage: :load FILENAME"
            try:
                path = Path(arg)
                self.engine.load_file(path)
                return f"Loaded {len(self.engine)} rules from {path}"
            except Exception as e:
                return f"Error loading {arg}: {e}"

        elif cmd == "rules":
            rules = self.engine.list_rules()
            if not rules:
                return "No rules loaded"
            return "\n".join(rules)

        elif cmd == "clear":
            self.engine.clear()
            return "Cleared all rules"

        elif cmd == "prelude":
            if not arg:
                available = ", ".join(BUILTIN_PRELUDES.keys())
                return f"Usage: :prelude NAME\nAvailable: {available}\nOr provide a path to a .py file"
            if self.set_prelude(arg):
                return f"Prelude set to: {arg}"
            else:
                return f"Unknown prelude: {arg}"

        elif cmd == "trace":
            if arg.lower() in ("on", "true", "1"):
                self.trace = True
                return "Tracing enabled"
            elif arg.lower() in ("off", "false", "0"):
                self.trace = False
                return "Tracing disabled"
            else:
                self.trace = not self.trace
                return f"Tracing {'enabled' if self.trace else 'disabled'}"

        elif cmd == "strategy":
            if arg.lower() in ("exhaustive", "once", "bottomup", "topdown"):
                self.strategy = arg.lower()
                return f"Strategy set to: {self.strategy}"
            else:
                return "Unknown strategy. Options: exhaustive, once, bottomup, topdown"

        elif cmd == "groups":
            groups = self.engine.groups()
            if not groups:
                return "No groups defined"
            return "Groups: " + ", ".join(sorted(groups))

        elif cmd == "enable":
            if not arg:
                return "Usage: :enable GROUP"
            self.engine.enable_group(arg)
            return f"Enabled group: {arg}"

        elif cmd == "disable":
            if not arg:
                return "Usage: :disable GROUP"
            self.engine.disable_group(arg)
            return f"Disabled group: {arg}"

        else:
            return f"Unknown command: {cmd}. Type :help for help."

    def help_text(self) -> str:
        """Return help text."""
        return """RERUM REPL Commands:
  :help              Show this help
  :load FILE         Load rules from file (.rules or .json)
  :rules             List all loaded rules
  :clear             Clear all rules
  :prelude NAME      Set prelude (arithmetic, math, full, none, or path.py)
  :trace on|off      Toggle tracing
  :strategy NAME     Set strategy (exhaustive, once, bottomup, topdown)
  :groups            Show all groups
  :enable GROUP      Enable a group
  :disable GROUP     Disable a group
  :quit              Exit

Syntax:
  @name: (pattern) => (skeleton)           Define a rule
  @name[priority]: (pattern) => (skeleton) Rule with priority
  @name: (pat) => (skel) when (cond)       Rule with guard
  [groupname]                              Start a rule group
  (expression)                             Evaluate an expression
"""

    def process_line(self, line: str) -> Optional[str]:
        """
        Process a single line of input.

        Returns the result to print, or None.
        """
        line = line.strip()

        # Empty line or comment
        if not line or line.startswith("#"):
            return None

        # Command
        if line.startswith(":"):
            return self.handle_command(line)

        # Rule definition (has =>)
        if "=>" in line:
            # Parse and add the rule
            parsed = load_rules_from_dsl(line)
            if parsed:
                for metadata, rule in parsed:
                    self.engine._rules.append(rule)
                    self.engine._metadata.append(metadata)
                    if metadata.name:
                        self.engine._rule_names[metadata.name] = len(self.engine._rules) - 1
                self.engine._sort_by_priority()
                self.engine._simplifier = None
                return f"Added {len(parsed)} rule(s)"
            else:
                return "Failed to parse rule"

        # Group declaration
        if line.startswith("[") and line.endswith("]"):
            # Just acknowledge - groups are handled when rules are loaded
            return f"Group: {line[1:-1]}"

        # Expression to evaluate
        try:
            expr = parse_sexpr(line)
            if expr is None:
                return None

            if self.trace:
                result, trace = self.engine(expr, trace=True, strategy=self.strategy)
                output = format_sexpr(result)
                if trace.steps:
                    return f"{output}\n{trace.format('rules')}"
                return output
            else:
                result = self.engine(expr, strategy=self.strategy)
                return format_sexpr(result)

        except Exception as e:
            return f"Error: {e}"

    def run(self):
        """Run the REPL loop."""
        print("RERUM - Rewriting Expressions via Rules Using Morphisms")
        print("Type :help for help, :quit to exit")
        print("Multi-line input: expressions with unbalanced parens continue on next line")
        print()

        while self.running:
            try:
                # Determine prompt based on whether we're in multi-line mode
                if self.multi_line_buffer:
                    prompt = "...... "
                else:
                    prompt = "rerum> "

                line = input(prompt)

                # Handle multi-line input
                if self.multi_line_buffer:
                    self.multi_line_buffer += "\n" + line
                else:
                    self.multi_line_buffer = line

                # Check if we have balanced parentheses
                paren_count = count_parens(self.multi_line_buffer)

                if paren_count > 0:
                    # More open parens than close - continue reading
                    continue
                elif paren_count < 0:
                    # More close parens than open - syntax error
                    print("Error: Unbalanced parentheses (too many closing)")
                    self.multi_line_buffer = ""
                    continue

                # Process the complete input
                complete_input = self.multi_line_buffer
                self.multi_line_buffer = ""

                result = self.process_line(complete_input)
                if result:
                    print(result)

            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                # Cancel multi-line input on Ctrl+C
                if self.multi_line_buffer:
                    print("\nInput cancelled")
                    self.multi_line_buffer = ""
                else:
                    print()
                continue

        self.save_history()


class ScriptRunner:
    """Runs rerum scripts."""

    def __init__(self):
        self.repl = RerumREPL()

    def run_script(self, path: Path, quiet: bool = False) -> int:
        """
        Run a script file.

        Args:
            path: Path to the script
            quiet: If True, don't print expression results

        Returns:
            Exit code (0 for success)
        """
        try:
            lines = path.read_text().splitlines()
        except Exception as e:
            print(f"Error reading {path}: {e}", file=sys.stderr)
            return 1

        # Track current group for rule loading
        current_group = None

        for lineno, line in enumerate(lines, 1):
            line = line.strip()

            # Skip empty lines, comments, and shebang
            if not line or line.startswith("#") or line.startswith("#!"):
                continue

            # Handle commands
            if line.startswith(":"):
                result = self.repl.handle_command(line)
                if result and not quiet:
                    # Don't print command confirmations in script mode
                    # unless it's an error
                    if "Error" in result or "Unknown" in result:
                        print(f"{path}:{lineno}: {result}", file=sys.stderr)
                        return 1
                continue

            # Handle group declarations
            if line.startswith("[") and line.endswith("]"):
                current_group = line[1:-1].strip()
                continue

            # Handle rules
            if "=>" in line:
                # Add group context if any
                if current_group:
                    line = f"[{current_group}]\n{line}"
                result = self.repl.process_line(line)
                if result and "Error" in result:
                    print(f"{path}:{lineno}: {result}", file=sys.stderr)
                    return 1
                continue

            # Handle expressions
            try:
                result = self.repl.process_line(line)
                if result:
                    print(result)
            except Exception as e:
                print(f"{path}:{lineno}: Error: {e}", file=sys.stderr)
                return 1

        return 0

    def run_expression(self, expr_str: str) -> int:
        """
        Evaluate a single expression.

        Returns:
            Exit code (0 for success)
        """
        result = self.repl.process_line(expr_str)
        if result:
            print(result)
            if "Error" in result:
                return 1
        return 0

    def run_stdin(self) -> int:
        """
        Read expressions from stdin and evaluate them.

        Returns:
            Exit code (0 for success)
        """
        for line in sys.stdin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            result = self.repl.process_line(line)
            if result:
                print(result)
                if "Error" in result:
                    return 1

        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="rerum",
        description="RERUM - Rewriting Expressions via Rules Using Morphisms",
        epilog="Examples:\n"
               "  rerum                          Start REPL\n"
               "  rerum script.rerum             Run script\n"
               "  rerum -e '(+ x 0)'             Evaluate expression\n"
               "  rerum -r rules.rules           REPL with rules\n"
               "  echo '(+ x 0)' | rerum -r rules.rules  Filter mode\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "script",
        nargs="?",
        help="Script file to run (.rerum)"
    )

    parser.add_argument(
        "-r", "--rules",
        action="append",
        default=[],
        help="Load rules from file (can be specified multiple times)"
    )

    parser.add_argument(
        "-e", "--expr",
        help="Evaluate a single expression"
    )

    parser.add_argument(
        "-p", "--prelude",
        default="none",
        help="Set prelude (arithmetic, math, full, none, or path.py)"
    )

    parser.add_argument(
        "-t", "--trace",
        action="store_true",
        help="Enable tracing"
    )

    parser.add_argument(
        "-s", "--strategy",
        default="exhaustive",
        choices=["exhaustive", "once", "bottomup", "topdown"],
        help="Rewriting strategy"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (suppress non-essential output)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )

    args = parser.parse_args()

    # Create runner
    runner = ScriptRunner()

    # Set prelude
    if not runner.repl.set_prelude(args.prelude):
        print(f"Unknown prelude: {args.prelude}", file=sys.stderr)
        sys.exit(1)

    # Set trace and strategy
    runner.repl.trace = args.trace
    runner.repl.strategy = args.strategy

    # Load rules files
    for rules_file in args.rules:
        try:
            runner.repl.engine.load_file(Path(rules_file))
            if not args.quiet:
                print(f"Loaded rules from {rules_file}", file=sys.stderr)
        except Exception as e:
            print(f"Error loading {rules_file}: {e}", file=sys.stderr)
            sys.exit(1)

    # Determine mode
    if args.script:
        # Script mode
        sys.exit(runner.run_script(Path(args.script), quiet=args.quiet))

    elif args.expr:
        # Expression mode
        sys.exit(runner.run_expression(args.expr))

    elif not sys.stdin.isatty():
        # Pipe/filter mode (stdin is not a terminal)
        sys.exit(runner.run_stdin())

    else:
        # REPL mode
        runner.repl.run()


if __name__ == "__main__":
    main()
