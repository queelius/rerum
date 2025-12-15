# Command-Line Interface

RERUM includes a CLI for interactive use and scripting.

## Modes

### REPL Mode

Start an interactive session:

```bash
$ rerum
rerum> @add-zero: (+ ?x 0) => :x
Added 1 rule(s)
rerum> (+ y 0)
y
rerum> :quit
```

### Script Mode

Run a `.rerum` script:

```bash
$ rerum script.rerum
```

Scripts support shebang:

```bash
#!/usr/bin/env rerum
:prelude full
@add-zero: (+ ?x 0) => :x
(+ x 0)
```

```bash
$ chmod +x script.rerum
$ ./script.rerum
x
```

### Expression Mode

Evaluate a single expression:

```bash
$ rerum -r rules.rules -p full -e "(+ x 0)"
x
```

### Pipe Mode

Process stdin:

```bash
$ echo "(+ x 0)" | rerum -r rules.rules -p full -q
x

$ cat expressions.txt | rerum -r rules.rules -q
```

## Command-Line Options

```
rerum [script]              Run script or start REPL

Options:
  -r, --rules FILE          Load rules from file (repeatable)
  -e, --expr EXPR           Evaluate single expression
  -p, --prelude NAME        Set prelude
  -t, --trace               Enable tracing
  -s, --strategy NAME       Set strategy
  -q, --quiet               Suppress non-essential output
  --version                 Show version
  -h, --help                Show help
```

### Prelude Options

| Name | Description |
|------|-------------|
| `none` | No computation (default) |
| `arithmetic` | `+`, `-`, `*`, `/`, `^` |
| `math` | Arithmetic + trig/exp/log |
| `full` | Arithmetic + predicates |
| `path.py` | Load custom prelude |

### Strategy Options

| Name | Description |
|------|-------------|
| `exhaustive` | Repeat until fixpoint (default) |
| `once` | Apply at most one rule |
| `bottomup` | Children before parent |
| `topdown` | Parent before children |

## REPL Commands

| Command | Description |
|---------|-------------|
| `:help` | Show help |
| `:load FILE` | Load rules from file |
| `:rules` | List loaded rules |
| `:clear` | Clear all rules |
| `:prelude NAME` | Set prelude |
| `:trace on/off` | Toggle tracing |
| `:strategy NAME` | Set strategy |
| `:groups` | Show all groups |
| `:enable GROUP` | Enable a group |
| `:disable GROUP` | Disable a group |
| `:quit` | Exit |

## Script Format

Scripts can contain:

- **Comments**: Lines starting with `#`
- **Directives**: Lines starting with `:`
- **Rules**: Lines containing `=>`
- **Groups**: Lines like `[groupname]`
- **Expressions**: Everything else (printed to stdout)

```bash
#!/usr/bin/env rerum
# My script

:prelude full
:load base.rules

[local]
@custom: (f ?x) => (g :x)

# Evaluate these expressions
(+ 1 2)
(f a)
```

## Custom Preludes

Create a Python file with a `PRELUDE` dict:

```python
# my_prelude.py
from rerum import binary_only, unary_only
import math

PRELUDE = {
    "gcd": binary_only(math.gcd),
    "factorial": unary_only(math.factorial),
    "even?": unary_only(lambda x: x % 2 == 0),
}
```

Use it:

```bash
$ rerum -p my_prelude.py -r rules.rules
```

Or in scripts:

```
:prelude my_prelude.py
```

## Examples

### Interactive Algebra

```bash
$ rerum -r algebra.rules -p full
rerum> (+ (* 2 3) (* 4 5))
26
rerum> (+ x 0)
x
```

### Batch Processing

```bash
$ cat << EOF | rerum -r algebra.rules -p full -q
(+ 1 2)
(* 3 4)
(+ x 0)
EOF
3
12
x
```

### Traced Simplification

```bash
$ rerum -r algebra.rules -p full -t -e "(+ (* x 1) 0)"
x
add-zero -> mul-one
```
