"""
Language-neutral intermediate representation (IR) for the translator.

Source code (e.g. Python/NumPy) is lowered into these nodes, analysis passes
annotate them (such as filling in each expression's `shape`), and a backend
emits target code (e.g. MATLAB/Octave) from them. Keeping the IR independent of
both languages lets semantics like operator meaning and array shapes be decided
once, instead of being pattern-matched on source text.

Indices in `Index`/`Slice` follow the source language's convention (0-based,
half-open slices for Python); backends are responsible for converting them.

Constructs the frontend can't lower become `Unsupported` records rather than
being silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

# A dimension of None means the size is unknown
Shape = tuple[int | None, ...]

BinOpKind = Literal["add", "sub", "mul", "div", "pow", "matmul"]
UnaryOpKind = Literal["neg", "pos", "not"]


# --- Base classes ---

@dataclass
class Expr:
    # Filled in by shape inference; None means not yet known
    shape: Shape | None = field(default=None, kw_only=True)


@dataclass
class Stmt:
    pass


# --- Expressions ---

@dataclass
class Name(Expr):
    id: str


@dataclass
class Const(Expr):
    value: int | float | complex | bool | str | None


# Nested lists of expressions, one nesting level per array dimension
ArrayElements = list[Union[Expr, "ArrayElements"]]


@dataclass
class ArrayLiteral(Expr):
    elements: ArrayElements


@dataclass
class BinOp(Expr):
    op: BinOpKind
    left: Expr
    right: Expr


@dataclass
class UnaryOp(Expr):
    op: UnaryOpKind
    operand: Expr


@dataclass
class Call(Expr):
    func: Expr
    args: list[Expr] = field(default_factory=list)
    kwargs: dict[str, Expr] = field(default_factory=dict)


@dataclass
class Slice(Expr):
    start: Expr | None = None
    stop: Expr | None = None
    step: Expr | None = None


@dataclass
class Index(Expr):
    value: Expr
    # One entry per indexed dimension; entries may be Slice nodes
    indices: list[Expr]


@dataclass
class Attribute(Expr):
    value: Expr
    attr: str


# --- Statements ---

@dataclass
class Assign(Stmt):
    # More than one target means multiple outputs, e.g. `a, b = f(x)`
    targets: list[Expr]
    value: Expr


@dataclass
class Unsupported:
    """A source construct the frontend could not lower into the IR."""
    lineno: int
    source: str
    reason: str


@dataclass
class Module:
    body: list[Stmt | Unsupported] = field(default_factory=list)
