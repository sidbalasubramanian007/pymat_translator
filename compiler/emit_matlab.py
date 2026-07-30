"""
MATLAB backend: emits MATLAB source from shape-annotated IR (run
compiler.shapes.infer first).

Operators default to MATLAB's elementwise (dotted) forms, which match NumPy
semantics for any shape; plain forms are only used when the annotated shapes
prove they're equivalent. 1-D NumPy arrays become MATLAB row vectors.
"""

from __future__ import annotations

from compiler.ir import ArrayLiteral, Assign, BinOp, Const, Expr, Module, Name, UnaryOp, Unsupported

# MATLAB precedence, loosest to tightest. MATLAB's ^ is left-associative
# (Python's ** is right), so every binary op is emitted as left-associative.
_PREC_ADD = 1
_PREC_MUL = 2
_PREC_UNARY = 3
_PREC_POW = 4
_PREC_ATOM = 5

_UNARY_OPS = {"neg": "-", "pos": "+", "not": "~"}

_MATMUL_TODO = "TODO: verify matmul orientation"


class _UnsupportedError(Exception):
    """Raised for IR the emitter can't handle yet; the statement becomes a placeholder."""


def emit(module: Module) -> str:
    lines = []
    for stmt in module.body:
        if isinstance(stmt, Unsupported):
            lines.append(f"% UNSUPPORTED (line {stmt.lineno}): {stmt.reason}")
        elif isinstance(stmt, Assign):
            lines.append(_emit_assign(stmt))
        else:
            lines.append(f"% UNSUPPORTED: {type(stmt).__name__}")
    return "".join(line + "\n" for line in lines)


def _emit_assign(stmt: Assign) -> str:
    target = stmt.targets[0]
    target_desc = target.id if isinstance(target, Name) else type(target).__name__
    emitter = _ExprEmitter()
    try:
        if len(stmt.targets) != 1 or not isinstance(target, Name):
            raise _UnsupportedError("assignment target")
        value, _ = emitter.emit(stmt.value)
    except _UnsupportedError as e:
        return f"% UNSUPPORTED: {e} in assignment to {target_desc}"

    line = f"{target.id} = {value};"
    if emitter.todos:
        line += "  % " + "; ".join(emitter.todos)
    return line


class _ExprEmitter:
    def __init__(self):
        self.todos: list[str] = []

    def emit(self, node: Expr) -> tuple[str, int]:
        """Returns (text, precedence of its outermost operator)."""
        if isinstance(node, Const):
            text = _format_const(node.value)
            # A negative literal behaves like unary minus: (-1)^2 must keep its parens
            return text, _PREC_UNARY if text.startswith("-") else _PREC_ATOM

        if isinstance(node, Name):
            return node.id, _PREC_ATOM

        if isinstance(node, ArrayLiteral):
            return self._array(node), _PREC_ATOM

        if isinstance(node, UnaryOp):
            # Parenthesize anything looser than ^, so -(a + b) and -(-x) stay explicit
            return f"{_UNARY_OPS[node.op]}{self._wrap(node.operand, _PREC_POW)}", _PREC_UNARY

        if isinstance(node, BinOp):
            if node.op == "matmul":
                return self._matmul(node)
            return self._elementwise(node)

        raise _UnsupportedError(type(node).__name__)

    def _wrap(self, node: Expr, min_prec: int) -> str:
        text, prec = self.emit(node)
        return f"({text})" if prec < min_prec else text

    def _binary(self, left: Expr, op: str, right: Expr, prec: int) -> tuple[str, int]:
        # Left-associative: an equal-precedence right operand needs parens (a - (b - c))
        return f"{self._wrap(left, prec)} {op} {self._wrap(right, prec + 1)}", prec

    def _elementwise(self, node: BinOp) -> tuple[str, int]:
        left_scalar = node.left.shape == "scalar"
        right_scalar = node.right.shape == "scalar"
        if node.op in ("add", "sub"):
            return self._binary(node.left, "+" if node.op == "add" else "-", node.right, _PREC_ADD)
        if node.op == "mul":
            return self._binary(node.left, "*" if left_scalar or right_scalar else ".*", node.right, _PREC_MUL)
        if node.op == "div":
            return self._binary(node.left, "/" if right_scalar else "./", node.right, _PREC_MUL)
        if node.op == "pow":
            return self._binary(node.left, "^" if left_scalar and right_scalar else ".^", node.right, _PREC_POW)
        raise _UnsupportedError(f"BinOp {node.op}")

    def _matmul(self, node: BinOp) -> tuple[str, int]:
        a, b = node.left.shape, node.right.shape
        ndims = (len(a), len(b)) if isinstance(a, tuple) and isinstance(b, tuple) else None

        if ndims == (2, 1):
            # Matrix times 1-D: multiply by the column, then back to a row vector
            left = self._wrap(node.left, _PREC_MUL)
            right = self._wrap(node.right, _PREC_ATOM)
            return f"({left} * {right}.').'", _PREC_ATOM
        if ndims == (1, 1):
            # Dot product of two row vectors
            left = self._wrap(node.left, _PREC_MUL)
            right = self._wrap(node.right, _PREC_ATOM)
            return f"{left} * {right}.'", _PREC_MUL
        if ndims not in ((2, 2), (1, 2)):
            self.todos.append(_MATMUL_TODO)
        return self._binary(node.left, "*", node.right, _PREC_MUL)

    def _array(self, node: ArrayLiteral) -> str:
        elements = node.elements
        if not any(isinstance(e, list) for e in elements):
            return "[" + ", ".join(self.emit(e)[0] for e in elements) + "]"

        rows_are_flat = all(
            isinstance(row, list) and not any(isinstance(e, list) for e in row) for row in elements
        )
        if rows_are_flat and len({len(row) for row in elements}) == 1:
            return "[" + "; ".join(", ".join(self.emit(e)[0] for e in row) for row in elements) + "]"

        raise _UnsupportedError("ArrayLiteral that is ragged or has more than 2 dimensions")


def _format_const(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)  # Shortest round-tripping form, so 0.1 stays 0.1
    raise _UnsupportedError(f"Const {value!r}")
