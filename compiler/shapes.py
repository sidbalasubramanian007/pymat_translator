"""
Shape inference: annotates every IR expression's `shape` in place and tracks
the shape of each variable as assignments run in order.

A shape is "scalar", a tuple of dims (int, or None for an unknown size), or
None when nothing is known.
"""

from __future__ import annotations

from dataclasses import fields

from compiler.ir import (
    ArrayLiteral, Assign, Attribute, BinOp, Call, Const, Expr, Index, Module, Name, Shape, Slice, Tuple, UnaryOp,
)

_ELEMENTWISE_OPS = {"add", "sub", "mul", "div", "pow"}


def infer(module: Module) -> dict[str, Shape | None]:
    symbols: dict[str, Shape | None] = {}
    for stmt in module.body:
        # Unsupported entries carry no IR to annotate
        if isinstance(stmt, Assign):
            shape = _infer_expr(stmt.value, symbols)
            target = stmt.targets[0]
            if isinstance(target, Name):
                target.shape = shape
                symbols[target.id] = shape
    return symbols


def _infer_expr(node: Expr, symbols: dict[str, Shape | None]) -> Shape | None:
    if isinstance(node, Const):
        is_number = isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool)
        shape = "scalar" if is_number else None
    elif isinstance(node, Name):
        shape = symbols.get(node.id)
    elif isinstance(node, ArrayLiteral):
        shape = _literal_shape(node.elements, symbols)
    else:
        # Annotate children first; the rules below read their shapes
        for child in _children(node):
            _infer_expr(child, symbols)
        shape = _derived_shape(node)
    node.shape = shape
    return shape


def _derived_shape(node: Expr) -> Shape | None:
    """Shape of a node whose children are already annotated. None whenever unsure."""
    if isinstance(node, BinOp):
        if node.op == "matmul":
            return _matmul_shape(node.left.shape, node.right.shape)
        if node.op in _ELEMENTWISE_OPS:
            return _broadcast_shape(node.left.shape, node.right.shape)
        return None

    if isinstance(node, UnaryOp):
        return node.operand.shape if node.op in ("neg", "pos") else None

    if isinstance(node, Index):
        return _index_shape(node)

    if isinstance(node, Attribute):
        if node.attr == "T":
            base = node.value.shape
            return tuple(reversed(base)) if isinstance(base, tuple) else base
        return None  # .shape and others: no rule yet

    if isinstance(node, Call):
        return _call_shape(node)

    # Tuple, Slice: not arrays
    return None


def _broadcast_shape(a: Shape | None, b: Shape | None) -> Shape | None:
    if a is None or b is None:
        return None
    if a == "scalar":
        return b
    if b == "scalar":
        return a

    # NumPy broadcasting: align from the right, padding the shorter shape with 1s
    ndim = max(len(a), len(b))
    a = (1,) * (ndim - len(a)) + a
    b = (1,) * (ndim - len(b)) + b
    dims = []
    for x, y in zip(a, b):
        if x is None or y is None:
            dims.append(None)
        elif x == y or y == 1:
            dims.append(x)
        elif x == 1:
            dims.append(y)
        else:
            return None  # Incompatible
    return tuple(dims)


def _matmul_shape(a: Shape | None, b: Shape | None) -> Shape | None:
    # Only 1-D and 2-D operands; scalars aren't allowed with @
    if not (isinstance(a, tuple) and isinstance(b, tuple)):
        return None
    if len(a) not in (1, 2) or len(b) not in (1, 2):
        return None
    inner_a, inner_b = a[-1], b[0]
    if inner_a is None or inner_b is None or inner_a != inner_b:
        return None
    # A 1-D operand contributes no output dim: (n,)@(n,) -> scalar, (m,n)@(n,) -> (m,), ...
    return (a[:-1] + b[1:]) or "scalar"


def _index_shape(node: Index) -> Shape | None:
    base = node.value.shape
    if not isinstance(base, tuple) or len(node.indices) > len(base):
        return None

    dims = []
    for idx, dim in zip(node.indices, base):
        if isinstance(idx, Slice):
            if isinstance(idx.step, Const) and idx.step.value == 0:
                return None  # Python raises on a zero step
            dims.append(_slice_length(idx, dim))
        elif idx.shape == "scalar":
            if isinstance(idx, Const):
                if not _is_int(idx.value):
                    return None  # Float index raises
                if dim is not None and not -dim <= idx.value < dim:
                    return None  # Out of bounds
            continue  # Integer index drops the dimension
        else:
            return None  # Array/unknown index (fancy indexing): no rule

    # Unindexed trailing dims are kept, as in NumPy
    dims.extend(base[len(node.indices):])
    return tuple(dims) or "scalar"


def _slice_length(s: Slice, dim: int | None) -> int | None:
    if dim is None:
        return None
    bounds = []
    for part in (s.start, s.stop, s.step):
        if part is None:
            bounds.append(None)
        elif isinstance(part, Const) and _is_int(part.value):
            bounds.append(part.value)
        else:
            return None  # Non-constant bound
    # Let Python's own slice resolve negative bounds and steps
    return len(range(*slice(*bounds).indices(dim)))


def _call_shape(node: Call) -> Shape | None:
    if node.func in ("np.zeros", "np.ones") and node.args:
        shape_arg = node.args[0]
        if isinstance(shape_arg, Tuple):
            dims = [_size_const(e) for e in shape_arg.elements]
            if not dims or None in dims:
                return None
            return tuple(dims)
        n = _size_const(shape_arg)
        return (n,) if n is not None else None

    # np.eye(n, M) is (n, M); only the square form has a rule
    if node.func == "np.eye" and len(node.args) == 1 and "M" not in node.kwargs:
        n = _size_const(node.args[0])
        return (n, n) if n is not None else None

    return None


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _size_const(node: Expr) -> int | None:
    """The value of a non-negative int Const, else None."""
    if isinstance(node, Const) and _is_int(node.value) and node.value >= 0:
        return node.value
    return None


def _literal_shape(elements: list, symbols: dict[str, Shape | None]) -> Shape | None:
    """Shape of a (possibly nested) list literal; None if ragged or any leaf isn't a scalar."""
    # Annotate everything first so leaves get shapes even when the literal is ragged
    sub_shapes = [_literal_shape(e, symbols) if isinstance(e, list) else _infer_expr(e, symbols) for e in elements]

    if any(isinstance(e, list) for e in elements):
        # Rows must all be lists with the same known shape
        if not all(isinstance(e, list) for e in elements):
            return None
        if sub_shapes[0] is None or any(s != sub_shapes[0] for s in sub_shapes):
            return None
        return (len(elements), *sub_shapes[0])

    # Flat: every element must be a scalar (an array element like
    # np.array([a, b]) with a = [1, 2] would add a dimension)
    if any(s != "scalar" for s in sub_shapes):
        return None
    return (len(elements),)


def _children(node: Expr):
    for f in fields(node):
        value = getattr(node, f.name)
        values = value.values() if isinstance(value, dict) else value if isinstance(value, list) else [value]
        for v in values:
            if isinstance(v, Expr):
                yield v
