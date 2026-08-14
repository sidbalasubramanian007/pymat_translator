"""
Python frontend: lowers Python/NumPy source into compiler.ir nodes.

Statements that can't be lowered become `Unsupported` records (kept in order in
`Module.body` and also returned as a list) so the rest of the program still lowers.

np.* calls must have a rule in rules.yaml and fit its params. Aliases declared
there are rewritten here, so later passes only see the canonical form:
x.reshape(2, 3) -> np.reshape(x, (2, 3)), x.T -> np.transpose(x), np.dot(a, b) -> a @ b.
"""

from __future__ import annotations

import ast

from compiler import rules
from compiler.ir import (
    ArrayLiteral, Assign, Attribute, BinOp, Call, Const, Expr, Index, Module, Name, Slice, Tuple, UnaryOp,
    Unsupported,
)

_BINOPS = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.Pow: "pow",
    ast.MatMult: "matmul",
}

# `not` and `~` aren't lowered: on arrays they raise or mean bitwise inversion, not MATLAB's ~
_UNARYOPS = {
    ast.USub: "neg",
    ast.UAdd: "pos",
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
            lowered.lineno = stmt.lineno
            lowered.source = ast.get_source_segment(source, stmt) or ""
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

    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise _UnsupportedError(f"unsupported unary operator: {type(node.op).__name__}")
        return UnaryOp(op, _lower_expr(node.operand))

    if isinstance(node, ast.Call):
        func = _dotted_name(node.func)
        if func == "np.array" and len(node.args) == 1 and not node.keywords and isinstance(node.args[0], ast.List):
            return ArrayLiteral(_lower_list(node.args[0]))
        if any(kw.arg is None for kw in node.keywords):
            raise _UnsupportedError("**kwargs in calls is not supported")
        args = [_lower_call_arg(a) for a in node.args]
        kwargs = {kw.arg: _lower_call_arg(kw.value) for kw in node.keywords}

        # Method aliases from rules.yaml: x.reshape(2, 3) -> np.reshape(x, (2, 3))
        receiver, _, method = func.rpartition(".")
        method_rule = rules.by_method(method) if receiver and "." not in receiver and receiver != "np" else None
        if method_rule is not None:
            if method_rule.method_pack and len(args) > 1:
                args = [Tuple(args)]
            func, args = method_rule.name, [Name(receiver), *args]

        if func.startswith("np."):
            return _lower_np_call(func, args, kwargs)
        return Call(func, args, kwargs)

    if isinstance(node, ast.Subscript):
        # Python 3.9+: the subscript is the node itself; a[i, j] arrives as a Tuple
        parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return Index(_lower_expr(node.value), [_lower_index(p) for p in parts])

    if isinstance(node, ast.Attribute):
        # Attribute aliases from rules.yaml: x.T -> np.transpose(x)
        attribute_rule = rules.by_attribute(node.attr)
        if attribute_rule is not None:
            return Call(attribute_rule.name, [_lower_expr(node.value)])
        return Attribute(_lower_expr(node.value), node.attr)

    raise _UnsupportedError(f"unsupported expression: {type(node).__name__}")


def _lower_np_call(func: str, args: list[Expr], kwargs: dict[str, Expr]) -> Expr:
    """np.* calls must have a rule in rules.yaml and fit its params; anything else is unsupported."""
    rule = rules.get(func)
    if rule is None:
        raise _UnsupportedError(f"no translation rule for {func}")
    try:
        bound = rules.bind(rule, args, kwargs)
    except rules.RuleError as e:
        raise _UnsupportedError(str(e)) from None
    if rule.via:
        # e.g. np.dot(a, b) means exactly a @ b
        left, right = (bound.get(p) for p in rule.params)
        if left is None or right is None:
            raise _UnsupportedError(f"{func} needs both arguments")
        return BinOp(rule.via, left, right)
    return Call(func, args, kwargs)


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
