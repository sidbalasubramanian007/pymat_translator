"""
Python frontend: lowers Python/NumPy source into compiler.ir nodes.

Statements that can't be lowered become `Unsupported` records (kept in order in
`Module.body` and also returned as a list) so the rest of the program still lowers.
"""

from __future__ import annotations

import ast

from compiler.ir import ArrayLiteral, Assign, BinOp, Const, Expr, Module, Name, Unsupported

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

    if isinstance(node, ast.Name):
        return Name(node.id)

    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise _UnsupportedError(f"unsupported operator: {type(node.op).__name__}")
        return BinOp(op, _lower_expr(node.left), _lower_expr(node.right))

    if _is_np_array_call(node):
        if len(node.args) != 1 or node.keywords or not isinstance(node.args[0], ast.List):
            raise _UnsupportedError("np.array is only supported with a single list literal argument")
        return ArrayLiteral(_lower_list(node.args[0]))

    raise _UnsupportedError(f"unsupported expression: {type(node).__name__}")


def _is_np_array_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "array"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
    )


def _lower_list(node: ast.List) -> list:
    return [_lower_list(e) if isinstance(e, ast.List) else _lower_expr(e) for e in node.elts]
