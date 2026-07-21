import pytest

from compiler.frontend import lower
from compiler.ir import (
    ArrayLiteral, Assign, Attribute, BinOp, Call, Const, Index, Name, Slice, Tuple, Unsupported,
)


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


def lower_value(source):
    return lower_single(source).value


def test_index_single():
    assert lower_value("y = a[i]") == Index(Name("a"), [Name("i")])


def test_index_multi_dim_tuple_is_unpacked():
    assert lower_value("y = a[i, 0]") == Index(Name("a"), [Name("i"), Const(0)])


@pytest.mark.parametrize("src, expected", [
    ("a[1:3]", [Slice(Const(1), Const(3), None)]),
    ("a[::2]", [Slice(None, None, Const(2))]),
    ("a[2:]", [Slice(Const(2), None, None)]),
    ("a[1, :]", [Const(1), Slice(None, None, None)]),
])
def test_slices(src, expected):
    assert lower_value(f"y = {src}") == Index(Name("a"), expected)


def test_negative_constant_is_folded():
    assert lower_value("x = -1") == Const(-1)
    assert lower_value("x = -2.5") == Const(-2.5)


def test_negative_index_and_slice_bounds():
    assert lower_value("y = a[-1]") == Index(Name("a"), [Const(-1)])
    assert lower_value("y = a[:-1]") == Index(Name("a"), [Slice(None, Const(-1), None)])


def test_call_with_tuple_arg():
    assert lower_value("z = np.zeros((2, 3))") == Call("np.zeros", [Tuple([Const(2), Const(3)])], {})


def test_call_with_kwargs():
    assert lower_value("s = np.sum(x, axis=0)") == Call("np.sum", [Name("x")], {"axis": Const(0)})


def test_call_with_deeper_dotted_name():
    assert lower_value("y = np.linalg.inv(x)") == Call("np.linalg.inv", [Name("x")], {})


def test_np_array_with_non_list_arg_is_a_generic_call():
    assert lower_value("y = np.array(x)") == Call("np.array", [Name("x")], {})


@pytest.mark.parametrize("attr", ["T", "shape"])
def test_attribute(attr):
    assert lower_value(f"y = x.{attr}") == Attribute(Name("x"), attr)


def test_index_into_attribute():
    assert lower_value("n = x.shape[0]") == Index(Attribute(Name("x"), "shape"), [Const(0)])


@pytest.mark.parametrize("source, reason_part", [
    ("y = -x", "UnaryOp"),
    ("y = f(*args)", "*args"),
    ("y = f(**opts)", "**kwargs"),
    ("y = (lambda v: v)(1)", "call target"),
    ("a[0] = 1", "single name"),
])
def test_new_unsupported_cases(source, reason_part):
    module, unsupported = lower(source)
    assert len(unsupported) == 1
    assert reason_part in unsupported[0].reason
    assert module.body == unsupported
