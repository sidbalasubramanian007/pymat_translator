"""
Loads compiler/rules.yaml (NumPy call -> MATLAB translation rules) and applies it:
binding a call's arguments to a rule's params, picking the matching case,
computing the result shape, and rendering the MATLAB template.

The format is documented at the top of rules.yaml.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

from compiler.ir import Const, Expr, Name, Shape, Tuple

RULES_PATH = Path(__file__).with_name("rules.yaml")

_RULE_KEYS = {"params", "kwargs", "method", "method_pack", "attribute", "via", "cases"}
_CASE_KEYS = {"when", "matlab", "shape"}
_NAMED_CONDITIONS = {"absent", "present", "int", "size", "positive", "name", "nonconst", "scalar", "1d", "2d"}
_PLACEHOLDER = re.compile(r"\{(\()?(\w+)(?:\.(\d+))?(?:([+-])(\d+))?(\))?\}")


class RuleError(Exception):
    """A call doesn't fit a rule (e.g. an unknown keyword), or rules.yaml is malformed."""


@dataclass(frozen=True)
class Case:
    when: dict[str, tuple]
    matlab: str
    shape: Any


@dataclass(frozen=True)
class Rule:
    name: str
    params: tuple[str, ...]
    kwargs: tuple[str, ...] = ()
    cases: tuple[Case, ...] = ()
    method: str | None = None
    method_pack: str | None = None
    attribute: str | None = None
    via: str | None = None


@lru_cache(maxsize=None)
def load(path: Path = RULES_PATH) -> dict[str, Rule]:
    raw = yaml.safe_load(Path(path).read_text())
    return {name: _parse_rule(name, spec) for name, spec in raw.items()}


def get(name: str) -> Rule | None:
    return load().get(name)


def by_method(method: str) -> Rule | None:
    return next((r for r in load().values() if r.method == method), None)


def by_attribute(attribute: str) -> Rule | None:
    return next((r for r in load().values() if r.attribute == attribute), None)


def _parse_rule(name: str, spec: dict) -> Rule:
    unknown = set(spec) - _RULE_KEYS
    if unknown:
        raise RuleError(f"{name}: unknown keys {sorted(unknown)}")
    params = tuple(spec["params"])
    kwargs = tuple(spec.get("kwargs", ()))
    if not set(kwargs) <= set(params):
        raise RuleError(f"{name}: kwargs must be params")
    if bool(spec.get("via")) == bool(spec.get("cases")):
        raise RuleError(f"{name}: needs exactly one of `via` or `cases`")

    cases = []
    for case in spec.get("cases", ()):
        unknown = set(case) - _CASE_KEYS
        if unknown or "matlab" not in case:
            raise RuleError(f"{name}: bad case keys {sorted(case)}")
        when = {}
        for param, conds in (case.get("when") or {}).items():
            if param not in params:
                raise RuleError(f"{name}: condition on unknown param {param!r}")
            conds = tuple(conds) if isinstance(conds, list) else (conds,)
            for cond in conds:
                if not (isinstance(cond, int) or cond in _NAMED_CONDITIONS or re.fullmatch(r"sizes:\d+", str(cond))):
                    raise RuleError(f"{name}: unknown condition {cond!r}")
            when[param] = conds
        cases.append(Case(when, case["matlab"], case.get("shape")))

    return Rule(
        name=name,
        params=params,
        kwargs=kwargs,
        cases=tuple(cases),
        method=spec.get("method"),
        method_pack=spec.get("method_pack"),
        attribute=spec.get("attribute"),
        via=spec.get("via"),
    )


# --- Applying a rule ---

def bind(rule: Rule, args: list[Expr], kwargs: dict[str, Expr]) -> dict[str, Expr]:
    """Maps a call's arguments to the rule's param names; raises RuleError if they don't fit."""
    if len(args) > len(rule.params):
        raise RuleError(f"{rule.name} takes at most {len(rule.params)} positional arguments")
    bound = dict(zip(rule.params, args))
    for key, value in kwargs.items():
        if key not in rule.kwargs:
            raise RuleError(f"{rule.name} does not accept keyword {key!r}")
        if key in bound:
            raise RuleError(f"{rule.name} got {key!r} twice")
        bound[key] = value
    return bound


def match(rule: Rule, bound: dict[str, Expr]) -> Case | None:
    for case in rule.cases:
        if all(_holds(cond, bound.get(param)) for param, conds in case.when.items() for cond in conds):
            return case
    return None


def _holds(cond, arg: Expr | None) -> bool:
    if cond == "absent":
        return arg is None
    if arg is None:
        return False
    if cond == "present":
        return True
    value = _int_value(arg)
    if isinstance(cond, int):
        return value == cond
    if cond == "int":
        return value is not None
    if cond == "size":
        return value is not None and value >= 0
    if cond == "positive":
        return value is not None and value > 0
    if cond == "name":
        return isinstance(arg, Name)
    if cond == "nonconst":
        return not isinstance(arg, Const)
    if cond == "scalar":
        return arg.shape == "scalar"
    if cond in ("1d", "2d"):
        return isinstance(arg.shape, tuple) and len(arg.shape) == int(cond[0])
    if cond.startswith("sizes:"):
        return (
            isinstance(arg, Tuple)
            and len(arg.elements) == int(cond.split(":")[1])
            and all(_holds("size", e) for e in arg.elements)
        )
    raise RuleError(f"unknown condition {cond!r}")


def case_shape(case: Case, bound: dict[str, Expr]) -> Shape | None:
    spec = case.shape
    if spec is None:
        return None
    if spec == "scalar":
        return "scalar"
    if isinstance(spec, dict) and "same" in spec:
        return bound[spec["same"]].shape
    if isinstance(spec, dict) and "range" in spec:
        start, stop, step = (v if isinstance(v, int) else _int_value(bound[v]) for v in spec["range"])
        return (len(range(start, stop, step)),)
    if isinstance(spec, list):
        return tuple(_dim(d, bound) for d in spec)
    raise RuleError(f"bad shape spec {spec!r}")


def _dim(spec, bound: dict[str, Expr]) -> int | None:
    if spec is None or isinstance(spec, int):
        return spec
    param, _, k = str(spec).partition(".")
    arg = bound[param]
    if not k:
        return _int_value(arg)
    if isinstance(arg, Tuple):
        return _int_value(arg.elements[int(k)])
    return arg.shape[int(k)] if isinstance(arg.shape, tuple) else None


def render(case: Case, bound: dict[str, Expr], text_of: Callable[[Expr, bool], str]) -> str:
    """
    Fills the case's MATLAB template. text_of(expr, parenthesize) returns an
    argument's MATLAB code, parenthesized when asked unless it's a single term.
    """
    def fill(m: re.Match) -> str:
        open_paren, param, k, sign, offset, close_paren = m.groups()
        arg = bound[param]
        if k is not None:
            arg = arg.elements[int(k)]
        if sign:
            value = _int_value(arg)
            return str(value + int(offset) if sign == "+" else value - int(offset))
        return text_of(arg, bool(open_paren and close_paren))

    return _PLACEHOLDER.sub(fill, case.matlab)


def identity_param(case: Case) -> str | None:
    """The param name if the template is just "{p}" (the call is a no-op on that argument)."""
    m = re.fullmatch(r"\{(\w+)\}", case.matlab)
    return m.group(1) if m else None


def _int_value(arg: Expr) -> int | None:
    if isinstance(arg, Const) and isinstance(arg.value, int) and not isinstance(arg.value, bool):
        return arg.value
    return None
