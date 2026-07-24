"""
Shape inference: annotates every IR expression's `shape` in place and tracks
the shape of each variable as assignments run in order.

A shape is "scalar", a tuple of dims (int, or None for an unknown size), or
None when nothing is known.
"""

from __future__ import annotations

from dataclasses import fields

from compiler.ir import ArrayLiteral, Assign, Const, Expr, Module, Name, Shape


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
        # No rule yet; still annotate children so nested literals get shapes
        for child in _children(node):
            _infer_expr(child, symbols)
        shape = None
    node.shape = shape
    return shape


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
