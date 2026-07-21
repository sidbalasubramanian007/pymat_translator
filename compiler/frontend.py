"""
Python frontend: lowers Python/NumPy source into compiler.ir nodes.

Statements that can't be lowered become `Unsupported` records (kept in order in
`Module.body` and also returned as a list) so the rest of the program still lowers.
"""

from __future__ import annotations

import ast

from compiler.ir import (
    ArrayLiteral, Assign, Attribute, BinOp, Call, Const, Expr, Index, Module, Name, Slice, Tuple, Unsupported,
)

_BINOPS = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.Pow: "pow",
    ast.MatMult: "matmul",
}


class _UnsupportedError(Exception):
    """Raised inside a statement to abandon it; caught and recorded per statement."""


def lower(source: str) -> tuple[Module, list[Unsupported]]:
    tree = ast.parse(source)
    module = Module()
    unsupported: list[Unsupported] = []

    for stmt in tree.body:
        try:
            lowered = _lower_stmt(stmt)
        except _UnsupportedError as e:
            entry = Unsupported(
                lineno=stmt.lineno,
                source=ast.get_source_segment(source, stmt) or "",
                reason=str(e),
            )
            module.body.append(entry)
            unsupported.append(entry)
            continue
        if lowered is not None:
            module.body.append(lowered)

    return module, unsupported


def _lower_stmt(stmt: ast.stmt) -> Assign | None:
    # `import numpy as np` needs no IR; np.* calls are recognized directly
    if isinstance(stmt, ast.Import) and [(a.name, a.asname) for a in stmt.names] == [("numpy", "np")]:
        return None

    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            raise _UnsupportedError("only assignment to a single name is supported")
        return Assign(targets=[Name(stmt.targets[0].id)], value=_lower_expr(stmt.value))

    raise _UnsupportedError(f"unsupported statement: {type(stmt).__name__}")


def _lower_expr(node: ast.expr) -> Expr:
    if isinstance(node, ast.Constant):
        # bool is a subclass of int, so exclude it explicitly
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return Const(node.value)
        raise _UnsupportedError(f"unsupported constant: {node.value!r}")

    # Python parses -1 as USub(1); fold it so negative indices stay recognizable
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        return Const(-node.operand.value)

    if isinstance(node, ast.Name):
        return Name(node.id)

    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise _UnsupportedError(f"unsupported operator: {type(node.op).__name__}")
        return BinOp(op, _lower_expr(node.left), _lower_expr(node.right))

    if isinstance(node, ast.Call):
        func = _dotted_name(node.func)
        if func == "np.array" and len(node.args) == 1 and not node.keywords and isinstance(node.args[0], ast.List):
            return ArrayLiteral(_lower_list(node.args[0]))
        if any(kw.arg is None for kw in node.keywords):
            raise _UnsupportedError("**kwargs in calls is not supported")
        return Call(
            func,
            [_lower_call_arg(a) for a in node.args],
            {kw.arg: _lower_call_arg(kw.value) for kw in node.keywords},
        )

    if isinstance(node, ast.Subscript):
        # Python 3.9+: the subscript is the node itself; a[i, j] arrives as a Tuple
        parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return Index(_lower_expr(node.value), [_lower_index(p) for p in parts])

    if isinstance(node, ast.Attribute):
        return Attribute(_lower_expr(node.value), node.attr)

    raise _UnsupportedError(f"unsupported expression: {type(node).__name__}")


def _dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr}"
    raise _UnsupportedError(f"unsupported call target: {type(node).__name__}")


def _lower_call_arg(node: ast.expr) -> Expr:
    if isinstance(node, ast.Tuple):
        return Tuple([_lower_expr(e) for e in node.elts])
    if isinstance(node, ast.Starred):
        raise _UnsupportedError("*args in calls is not supported")
    return _lower_expr(node)


def _lower_index(node: ast.expr) -> Expr:
    if isinstance(node, ast.Slice):
        return Slice(*(None if part is None else _lower_expr(part) for part in (node.lower, node.upper, node.step)))
    return _lower_expr(node)


def _lower_list(node: ast.List) -> list:
    return [_lower_list(e) if isinstance(e, ast.List) else _lower_expr(e) for e in node.elts]
