import pytest

from compiler.frontend import lower
from compiler.ir import ArrayLiteral, Assign, BinOp, Const, Name, Unsupported


def lower_single(source):
    module, unsupported = lower(source)
    assert unsupported == []
    assert len(module.body) == 1
    return module.body[0]


def test_numpy_import_is_skipped():
    module, unsupported = lower("import numpy as np\n")
    assert module.body == []
    assert unsupported == []


def test_int_constant():
    assert lower_single("x = 3") == Assign([Name("x")], Const(3))


def test_float_constant():
    assert lower_single("x = 2.5") == Assign([Name("x")], Const(2.5))


def test_name():
    assert lower_single("y = x") == Assign([Name("y")], Name("x"))


def test_np_array_flat():
    stmt = lower_single("a = np.array([1, 2, 3])")
    assert stmt == Assign([Name("a")], ArrayLiteral([Const(1), Const(2), Const(3)]))


def test_np_array_nested():
    stmt = lower_single("a = np.array([[1, 2], [3, 4.5]])")
    assert stmt.value == ArrayLiteral([[Const(1), Const(2)], [Const(3), Const(4.5)]])


@pytest.mark.parametrize("op_src, op", [
    ("+", "add"),
    ("-", "sub"),
    ("*", "mul"),
    ("/", "div"),
    ("**", "pow"),
    ("@", "matmul"),
])
def test_binary_ops(op_src, op):
    stmt = lower_single(f"c = a {op_src} b")
    assert stmt == Assign([Name("c")], BinOp(op, Name("a"), Name("b")))


def test_nested_binary_ops_follow_precedence():
    stmt = lower_single("c = a + b * 2")
    assert stmt.value == BinOp("add", Name("a"), BinOp("mul", Name("b"), Const(2)))


def test_unsupported_for_loop_does_not_stop_lowering():
    source = (
        "import numpy as np\n"
        "a = np.array([1, 2])\n"
        "for i in range(3):\n"
        "    a = a * 2\n"
        "b = a + 1\n"
    )
    module, unsupported = lower(source)

    assert len(unsupported) == 1
    entry = unsupported[0]
    assert entry.lineno == 3
    assert entry.source == "for i in range(3):\n    a = a * 2"
    assert "For" in entry.reason

    # Statements before and after the loop still lower, with the loop kept in place
    assert module.body == [
        Assign([Name("a")], ArrayLiteral([Const(1), Const(2)])),
        entry,
        Assign([Name("b")], BinOp("add", Name("a"), Const(1))),
    ]
    assert isinstance(module.body[1], Unsupported)
