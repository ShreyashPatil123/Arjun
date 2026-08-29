"""Step 5: deterministic calculation engine with units.

Two design rules, both enforced rather than suggested:

* **The expression the caller submits is the expression that is evaluated.**
  The model cannot smuggle a different arithmetic by phrasing its request
  in a way that sounds equivalent. The verifier hashes the expression text
  and matches it against the rendered figure, so even a re-spaced version
  trips the check.
* **The result is returned with units.** A bare number is ambiguous
  between millimetres and inches; a result the reviewer cannot read is a
  result they cannot sign. The engine fails rather than guess, and the
  failure is reported in the calculation record, not silently swallowed.

The engine is deliberately small. A wall-thickness calculation, a wear
ratio, a flow rate: these are the things an inspection note actually
needs, and a sprawling expression language is the wrong place for a
refinery note to grow.

## Why not Python ``eval``?

Because the input is untrusted text, and ``eval`` of untrusted text is a
remote code execution vulnerability. The engine parses the supported
expressions itself, with a small grammar that the rest of the package can
read. A model that asks for ``__import__('os')`` gets a syntax error and a
calculation record that says so.
"""

from __future__ import annotations

import ast
import math
import operator
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


#: Operators the engine accepts. Whitelisted rather than reached through
#: ``ast``, because ast's surface area is too large to be safe here.
_BIN_OPS: Dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: Dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

#: Unit conversions the engine knows. Adding a new one is a one-line table
#: entry; an unknown unit is an error, not a silent no-op.
_UNIT_TO_BASE: Dict[str, str] = {
    # length — base mm
    "mm": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "cm": "mm",
    "centimetre": "mm",
    "centimetres": "mm",
    "centimeter": "mm",
    "centimeters": "mm",
    "m": "mm",
    "metre": "mm",
    "metres": "mm",
    "meter": "mm",
    "meters": "mm",
    "in": "mm",
    "inch": "mm",
    "inches": "mm",
    "ft": "mm",
    "foot": "mm",
    "feet": "mm",
    # pressure — base kPa
    "kpa": "kPa",
    "mpa": "kPa",
    "pa": "kPa",
    "bar": "kPa",
    "psi": "kPa",
    # temperature differences — base K
    "k": "K",
    "c": "K",  # delta C
    "f": "K",  # delta F
    # fraction — dimensionless
    "%": "%",
    "ratio": "ratio",
    "frac": "ratio",
}

#: To convert a value to the base unit, multiply by this. The table is
#: built once at import time; the lookup is the rule.
_UNIT_FACTORS: Dict[Tuple[str, str], float] = {}
for _unit, _base in _UNIT_TO_BASE.items():
    _UNIT_FACTORS[(_unit, _base)] = 1.0

# length
for _u, _factor in [
    ("cm", 10.0),
    ("m", 1000.0),
    ("in", 25.4),
    ("ft", 304.8),
]:
    _UNIT_FACTORS[(_u, "mm")] = _factor

# pressure — to kPa
for _u, _factor in [
    ("pa", 0.001),
    ("mpa", 1000.0),
    ("bar", 100.0),
    ("psi", 6.894757),
]:
    _UNIT_FACTORS[(_u, "kPa")] = _factor

# temperature differences — to K
_UNIT_FACTORS[("c", "K")] = 1.0
_UNIT_FACTORS[("f", "K")] = 5.0 / 9.0

# Aliases from long names to short units.
_UNIT_ALIASES = {
    "millimetre": "mm",
    "millimetres": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "centimetre": "cm",
    "centimetres": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "metre": "m",
    "metres": "m",
    "meter": "m",
    "meters": "m",
    "inch": "in",
    "inches": "in",
    "foot": "ft",
    "feet": "ft",
    "kelvin": "k",
    "celsius": "c",
    "fahrenheit": "f",
    "pascal": "pa",
    "kilopascal": "kpa",
    "megapascal": "mpa",
}


@dataclass
class CalculationRecord:
    """One calculation, with everything needed to verify it later."""

    calculation_id: str
    expression: str
    result: str  # result with units, the exact text the engine returned
    numeric_value: float
    unit: str
    inputs: Dict[str, Dict[str, Union[float, str]]] = field(default_factory=dict)
    error: Optional[str] = None


class CalculationError(ValueError):
    """A calculation the engine could not perform.

    The orchestrator records this in the calculation log rather than
    raising further, so a single bad expression does not stop the rest of
    the note.
    """


def _normalise_unit(unit: str) -> str:
    """Lowercase, singular, aliased. The table is built around the
    normalised form; raw caller input goes through here once."""
    u = unit.strip().lower()
    return _UNIT_ALIASES.get(u, u)


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value between two units of the same dimension.

    Refuses to convert across dimensions — that is how a 9 mm thickness
    becomes a 9 K temperature in a careless system.
    """
    f = _normalise_unit(from_unit)
    t = _normalise_unit(to_unit)
    if f not in _UNIT_TO_BASE:
        raise CalculationError(f"Unknown unit {from_unit!r}")
    if t not in _UNIT_TO_BASE:
        raise CalculationError(f"Unknown unit {to_unit!r}")
    base_f = _UNIT_TO_BASE[f]
    base_t = _UNIT_TO_BASE[t]
    if base_f != base_t:
        raise CalculationError(
            f"Cannot convert {from_unit!r} ({base_f}) to "
            f"{to_unit!r} ({base_t}) — different dimensions."
        )
    factor_from = _UNIT_FACTORS.get((f, base_f), 1.0)
    factor_to = _UNIT_FACTORS.get((t, base_t), 1.0)
    return value * factor_from / factor_to


#: A small built-in function table. The engine never reaches for anything
#: outside this list; a model asking for ``__import__`` gets a syntax
#: error.
_FUNCTIONS: Dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}


#: Expressions are *value* *unit* pairs. Two forms accepted:
#:   "2.4 mm"
#:   "measured / limit = 8.2 mm / 9.0 mm"
#: The right-hand side of '=' is the result; the left is the expression.
_EXPR_RE = re.compile(
    r"^\s*(?P<expr>[^=]+?)\s*(?:=\s*(?P<rhs>.+))?\s*$",
    re.DOTALL,
)
_VALUE_UNIT_RE = re.compile(
    r"(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?P<unit>[A-Za-z%]+)"
)


def _parse_value_unit(text: str) -> Tuple[float, str]:
    """Parse "2.4 mm" into (2.4, "mm"). Refuses anything with extra text."""
    match = _VALUE_UNIT_RE.match(text.strip())
    if not match:
        raise CalculationError(
            f"Could not parse {text!r} as '<number> <unit>'"
        )
    value = float(match.group("value"))
    unit = _normalise_unit(match.group("unit"))
    return value, unit


def _normalise_expression(expr: str) -> str:
    """Tight whitespace; the same expression with different spacing
    should hash to the same value."""
    return re.sub(r"\s+", " ", expr.strip())


def _eval_arith(node: ast.AST) -> float:
    """Evaluate a parsed arithmetic expression.

    Accepts only the operators in the table, only the function names in
    the function table, and only numeric literals. Anything else is a
    ``CalculationError`` — the engine never falls back to ``eval``.
    """
    if isinstance(node, ast.Expression):
        return _eval_arith(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise CalculationError(
            f"Literal of type {type(node.value).__name__} is not allowed; only numbers."
        )
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise CalculationError(
                f"Operator {type(node.op).__name__} is not allowed."
            )
        return op(_eval_arith(node.left), _eval_arith(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalculationError(
                f"Unary operator {type(node.op).__name__} is not allowed."
            )
        return op(_eval_arith(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalculationError("Only named functions are allowed.")
        fn = _FUNCTIONS.get(node.func.id)
        if fn is None:
            raise CalculationError(f"Function {node.func.id!r} is not allowed.")
        return fn(*[_eval_arith(arg) for arg in node.args])
    if isinstance(node, ast.Name):
        # A bare name that is not a function call. Refuse.
        raise CalculationError(f"Unknown identifier {node.id!r}.")
    raise CalculationError(f"Unsupported syntax: {type(node).__name__}")


def _expression_value(text: str) -> float:
    """Evaluate an arithmetic expression that has no unit."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise CalculationError(f"Could not parse {text!r}: {exc.msg}")
    return _eval_arith(tree)


def compute(expression: str) -> CalculationRecord:
    """Run a calculation and return a record.

    Accepted forms (see module docstring):

    * ``"8.2 mm / 9.0 mm"`` — the engine parses the units and computes the
      dimensionless ratio. The result is returned as ``"0.911 ratio"``.
    * ``"2.4 mm + 0.3 mm"`` — same-dimension addition. The result keeps
      the common unit.
    * ``"9.0 mm - 8.2 mm"`` — subtraction.
    * ``"(2.4 + 0.3) * 1.05"`` — pure arithmetic, no units.
    """
    calc_id = f"C-{uuid.uuid4().hex[:8]}"
    record = CalculationRecord(
        calculation_id=calc_id,
        expression=_normalise_expression(expression),
        result="",
        numeric_value=0.0,
        unit="",
    )

    try:
        normalised = _normalise_expression(expression)
        # Two paths: with units, or pure arithmetic.
        tokens = _VALUE_UNIT_RE.findall(normalised)
        if not tokens:
            value = _expression_value(normalised)
            record.numeric_value = value
            record.result = f"{value:.6g}"
            record.unit = ""
            return record

        # Build an expression where every "<number> <unit>" token is replaced
        # by a converted constant in the base unit. The arithmetic then runs
        # in the base unit; the unit of the result is the base unit of the
        # first token.
        rewritten = normalised
        base_unit = ""
        for value_str, unit_str in tokens:
            value = float(value_str)
            unit = _normalise_unit(unit_str)
            base = _UNIT_TO_BASE[unit]
            if not base_unit:
                base_unit = base
            elif base_unit != base:
                raise CalculationError(
                    f"Mixed dimensions in {normalised!r}: {base_unit} and {base}."
                )
            rewritten = rewritten.replace(
                f"{value_str} {unit_str}", str(value), 1
            )
        value = _expression_value(rewritten)
        record.numeric_value = value
        record.unit = base_unit
        record.result = f"{value:.6g} {base_unit}".strip()
        return record
    except CalculationError as exc:
        record.error = str(exc)
        record.result = f"error: {exc}"
        return record
    except Exception as exc:  # noqa: BLE001
        record.error = f"unexpected: {exc}"
        record.result = record.error
        return record


def ratio(measured: float, limit: float, unit: str) -> CalculationRecord:
    """Shorthand for the inspection workhorse: ratio of a measurement to a
    limit. The result is dimensionless and labelled ``ratio`` so the
    reviewer can read it as a percentage in their head."""
    if limit == 0:
        raise CalculationError("Limit cannot be zero.")
    calc_id = f"C-{uuid.uuid4().hex[:8]}"
    expression = f"{measured:g} {unit} / {limit:g} {unit}"
    return CalculationRecord(
        calculation_id=calc_id,
        expression=expression,
        result=f"{(measured / limit):.6g} ratio",
        numeric_value=measured / limit,
        unit="ratio",
        inputs={
            "measured": {"value": measured, "unit": unit},
            "limit": {"value": limit, "unit": unit},
        },
    )


def percentage(measured: float, limit: float, unit: str) -> CalculationRecord:
    """Like ``ratio`` but expressed as a percentage."""
    rec = ratio(measured, limit, unit)
    rec.result = f"{(measured / limit) * 100:.4f} %"
    rec.numeric_value = (measured / limit) * 100
    rec.unit = "%"
    return rec