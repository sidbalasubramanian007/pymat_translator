"""
MATLAB backend: emits MATLAB source from shape-annotated IR (run
compiler.shapes.infer first).

Operators default to MATLAB's elementwise (dotted) forms, which match NumPy
semantics for any shape; plain forms are only used when the annotated shapes
prove they're equivalent. 1-D NumPy arrays become MATLAB row vectors.
"""

from __future__ import annotations

from compiler import rules
from compiler.ir import ArrayLiteral, Assign, BinOp, Call, Const, Expr, Index, Module, Name, Slice, UnaryOp, Unsupported

# MATLAB precedence, loosest to tightest. MATLAB's ^ is left-associative
# (Python's ** is right), so every binary op is emitted as left-associative.
_PREC_COLON = 0  # a:b ranges, e.g. from np.arange; 0:4 + 1 would mean 0:5
_PREC_ADD = 1
_PREC_MUL = 2
_PREC_UNARY = 3
_PREC_POW = 4
_PREC_ATOM = 5

_UNARY_OPS = {"neg": "-", "pos": "+", "not": "~"}

_MATMUL_TODO = "TODO: verify matmul orientation"
_INDEX_TODO = "TODO: verify indexing dimensionality"


class _UnsupportedError(Exception):
    """Raised for IR the emitter can't handle; the statement becomes an unsupported marker."""


def emit(module: Module) -> str:
    return emit_with_unsupported(module)[0]


def emit_with_unsupported(module: Module) -> tuple[str, list[Unsupported]]:
    """
    Emits MATLAB code plus every statement that couldn't be translated, whether the
    frontend already gave up on it or the emitter can't handle it. Each one gets a
    stable id (U1, U2, ... in source order) and exactly one marker line at its position:
        % UNSUPPORTED[U1]: <original Python source>
    """
    lines: list[str] = []
    unsupported: list[Unsupported] = []
    for stmt in module.body:
        if isinstance(stmt, Assign):
            try:
                lines.append(_emit_assign(stmt))
                continue
            except _UnsupportedError as e:
                entry = Unsupported(stmt.lineno, stmt.source or "", str(e))
        elif isinstance(stmt, Unsupported):
            entry = stmt
        else:
            entry = Unsupported(
                getattr(stmt, "lineno", None), getattr(stmt, "source", None) or "",
                f"unsupported statement: {type(stmt).__name__}",
            )
        entry.id = f"U{len(unsupported) + 1}"
        unsupported.append(entry)
        lines.append(unsupported_marker(entry))
    return "".join(line + "\n" for line in lines), unsupported


def unsupported_marker(entry: Unsupported) -> str:
    """The single marker line for an unsupported statement. Multi-line source is kept on
    one line with its newlines written as \\n; the full source is on the entry itself."""
    source = entry.source.replace("\n", "\\n") if entry.source else f"<{entry.reason}>"
    return f"% UNSUPPORTED[{entry.id}]: {source}"


def _emit_assign(stmt: Assign) -> str:
    """MATLAB line for an assignment; raises _UnsupportedError if any part can't be translated."""
    target = stmt.targets[0]
    if len(stmt.targets) != 1 or not isinstance(target, Name):
        raise _UnsupportedError("assignment target")
    emitter = _ExprEmitter()
    value, _ = emitter.emit(stmt.value)

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

        if isinstance(node, Index):
            return self._index(node), _PREC_ATOM

        if isinstance(node, Call):
            return self._call(node)

        raise _UnsupportedError(type(node).__name__)

    def _todo(self, message: str):
        if message not in self.todos:
            self.todos.append(message)

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
            self._todo(_MATMUL_TODO)
        return self._binary(node.left, "*", node.right, _PREC_MUL)

    def _call(self, node: Call) -> tuple[str, int]:
        """np.* calls are translated from rules.yaml; no rule or no matching case is unsupported."""
        rule = rules.get(node.func)
        if rule is None:
            raise _UnsupportedError(f"no translation rule for {node.func}")
        try:
            bound = rules.bind(rule, node.args, node.kwargs)
        except rules.RuleError as e:
            raise _UnsupportedError(str(e)) from None
        case = rules.match(rule, bound)
        if case is None:
            raise _UnsupportedError(f"{node.func} with these argument values or shapes")

        # A no-op template ("{x}") keeps the argument's own precedence
        identity = rules.identity_param(case)
        if identity is not None:
            return self.emit(bound[identity])

        text = rules.render(case, bound, lambda arg, parens: self._wrap(arg, _PREC_ATOM) if parens else self.emit(arg)[0])
        return text, _PREC_COLON if _has_top_level_colon(text) else _PREC_ATOM

    # --- Indexing: Python is 0-based with exclusive stops, MATLAB 1-based with inclusive ends ---

    def _index(self, node: Index) -> str:
        if not isinstance(node.value, Name):
            raise _UnsupportedError("Index into an expression")  # MATLAB can't index (a + b)(1)

        indices = node.indices
        base = node.value.shape
        if isinstance(base, tuple):
            if len(indices) > len(base):
                raise _UnsupportedError("Index with more indices than dimensions")
            if len(base) > 2:
                # MATLAB keeps singleton dims (t(2, :, :) stays 1x3x4), so N-D needs more than a rewrite
                raise _UnsupportedError("Index into an array with more than 2 dimensions")
            dims = list(base)
        elif base is None:
            dims = [None] * len(indices)
        else:
            raise _UnsupportedError("Index into a scalar")

        # a[:] (or A[:, :]) is the whole array; MATLAB's v(:) would turn a row vector into a column
        if all(isinstance(i, Slice) and i.start is None and i.stop is None and i.step is None for i in indices):
            return node.value.id

        parts = [
            self._slice(i, dim) if isinstance(i, Slice) else self._position(i)
            for i, dim in zip(indices, dims)
        ]

        if base is None and len(parts) == 1:
            self._todo(_INDEX_TODO)
        # A single index on a 2-D array selects a row (A[1] -> A(2, :)), never a linear index.
        # 1-D arrays are row vectors, where a(i) already means element i.
        if isinstance(base, tuple) and len(base) == 2 and len(parts) == 1:
            parts.append(":")

        text = f"{node.value.id}({', '.join(parts)})"
        # A[:, j] is 1-D in NumPy (a row vector here), but A(:, j) is a MATLAB column
        if len(indices) == 2 and isinstance(indices[0], Slice) and not isinstance(indices[1], Slice):
            if base is None:
                self._todo(_INDEX_TODO)
            text += ".'"
        return text

    def _position(self, node: Expr) -> str:
        """A 0-based Python position (negatives count from the end) as a 1-based MATLAB one."""
        if isinstance(node, Const):
            if not _is_int(node.value):
                raise _UnsupportedError("non-integer index")
            return str(node.value + 1) if node.value >= 0 else _end_offset(node.value + 1)
        if isinstance(node.shape, tuple):
            raise _UnsupportedError("array (fancy) index")
        return f"{self._wrap(node, _PREC_ADD)} + 1"

    def _slice(self, node: Slice, dim: int | None) -> str:
        start, stop, step = node.start, node.stop, node.step
        if step is not None and not (isinstance(step, Const) and _is_int(step.value) and step.value != 0):
            raise _UnsupportedError("slice with a zero or non-constant step")

        clamped = _clamped_slice(node, dim)
        if clamped is not None:
            return clamped

        if step is None or step.value > 0:
            if start is None and stop is None and step is None:
                return ":"
            first = "1" if start is None else self._position(start)
            # Python's exclusive stop t is MATLAB's inclusive end t, so no +1
            if stop is None:
                last = "end"
            elif isinstance(stop, Const):
                if not _is_int(stop.value):
                    raise _UnsupportedError("non-integer slice bound")
                last = str(stop.value) if stop.value >= 0 else _end_offset(stop.value)
            else:
                last = self._wrap(stop, _PREC_ADD)
            middle = [] if step is None else [str(step.value)]
            return ":".join([first, *middle, last])

        # Negative step: counting down, the exclusive stop t means the last element is t+1
        # (0-based), i.e. t+2 in MATLAB. Only constant bounds are converted.
        for bound in (start, stop):
            if bound is not None and not (isinstance(bound, Const) and _is_int(bound.value)):
                raise _UnsupportedError("negative-step slice with non-constant bounds")
        first = "end" if start is None else self._position(start)
        if stop is None:
            last = "1"
        elif stop.value >= 0:
            last = str(stop.value + 2)
        else:
            last = _end_offset(stop.value + 2)  # Stop -m is len-m, so end-m+2
        return f"{first}:{step.value}:{last}"

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


def _has_top_level_colon(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == ":" and depth == 0:
            return True
    return False


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _clamped_slice(node: Slice, dim: int | None) -> str | None:
    """
    Python clamps out-of-range slice bounds (v[:9] on 8 elements is all of v) where
    MATLAB raises. For an all-constant slice with such a bound on a known-size dim,
    returns the exact MATLAB range; otherwise None (the bounds map directly).
    """
    if dim is None:
        return None
    parts = (node.start, node.stop, node.step)
    if not all(p is None or (isinstance(p, Const) and _is_int(p.value)) for p in parts):
        return None
    start, stop, step = (None if p is None else p.value for p in parts)

    # Ranges where the direct start/stop mapping is exact (a negative step starts at most at dim-1)
    max_start = dim if step is None or step > 0 else dim - 1
    if (start is None or -dim <= start <= max_start) and (stop is None or -dim <= stop <= dim):
        return None

    positions = range(*slice(start, stop, step).indices(dim))
    if not positions:
        return "1:0"  # Empty range; v([]) would be 0x0 rather than 1x0
    first, last = positions[0] + 1, positions[-1] + 1
    return f"{first}:{last}" if positions.step == 1 else f"{first}:{positions.step}:{last}"


def _end_offset(offset: int) -> str:
    """MATLAB's `end` shifted by offset: end, end-1, end+1."""
    return "end" if offset == 0 else f"end{offset:+d}"


def _format_const(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)  # Shortest round-tripping form, so 0.1 stays 0.1
    raise _UnsupportedError(f"Const {value!r}")
