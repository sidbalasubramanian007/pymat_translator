import itertools

import numpy as np
import pytest

from compiler.frontend import lower
from compiler.ir import ArrayLiteral, Assign, Const, Index, Module, Name, Slice, UnaryOp
from compiler.shapes import infer


def infer_source(source):
    module, _ = lower(source)
    return module, infer(module)


@pytest.mark.parametrize("value", ["3", "2.5", "-1"])
def test_numeric_const_is_scalar(value):
    module, symbols = infer_source(f"x = {value}")
    assert symbols == {"x": "scalar"}
    assert module.body[0].value.shape == "scalar"


def test_non_numeric_const_is_unknown():
    module = Module([Assign([Name("s")], Const("text"))])
    assert infer(module) == {"s": None}


def test_flat_array_literal():
    module, symbols = infer_source("a = np.array([1, 2, 3])")
    assert symbols["a"] == (3,)
    literal = module.body[0].value
    assert literal.shape == (3,)
    assert all(leaf.shape == "scalar" for leaf in literal.elements)


def test_empty_array_literal():
    _, symbols = infer_source("a = np.array([])")
    assert symbols["a"] == (0,)


def test_nested_array_literal():
    _, symbols = infer_source("m = np.array([[1, 2, 3], [4, 5, 6]])")
    assert symbols["m"] == (2, 3)


def test_three_level_nesting():
    _, symbols = infer_source("t = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])")
    assert symbols["t"] == (2, 2, 2)


def test_ragged_rows_are_unknown_but_leaves_still_annotated():
    module, symbols = infer_source("r = np.array([[1, 2], [3]])")
    assert symbols["r"] is None
    literal = module.body[0].value
    assert literal.shape is None
    assert literal.elements[1][0].shape == "scalar"


def test_mixed_rows_and_values_are_unknown():
    module = Module([Assign([Name("r")], ArrayLiteral([Const(1), [Const(2)]]))])
    assert infer(module) == {"r": None}
    assert module.body[0].value.elements[0].shape == "scalar"


def test_literal_with_array_elements_is_unknown():
    # np.array([a, a]) with a of shape (2,) is really (2, 2); don't claim (2,)
    _, symbols = infer_source("a = np.array([1, 2])\nb = np.array([a, a])")
    assert symbols["b"] is None


def test_name_lookup():
    module, symbols = infer_source("x = np.array([1, 2])\ny = x")
    assert symbols == {"x": (2,), "y": (2,)}
    assert module.body[1].value.shape == (2,)


def test_unknown_name_is_none():
    module, symbols = infer_source("y = x")
    assert symbols == {"y": None}
    assert module.body[0].value.shape is None


def test_assign_annotates_target():
    module, _ = infer_source("x = 5")
    assert module.body[0].targets[0].shape == "scalar"


def test_reassignment_uses_latest_shape():
    module, symbols = infer_source("x = 5\ny = x\nx = np.array([1, 2])\nz = x")
    assert symbols == {"x": (2,), "y": "scalar", "z": (2,)}
    # Each read sees the shape at that point in the program
    assert module.body[1].value.shape == "scalar"
    assert module.body[3].value.shape == (2,)


@pytest.mark.parametrize("source", [
    "y = a + b",
    "y = a @ b",
    "y = a[0]",
    "y = a.T",
])
def test_nodes_over_unknown_names_are_unknown(source):
    _, symbols = infer_source(source)
    assert symbols == {"y": None}


def test_nested_literal_inside_other_node_is_annotated():
    module, _ = infer_source("s = np.sum(np.array([[1, 2], [3, 4]]), axis=0)")
    call = module.body[0].value
    assert call.shape == (2,)
    assert call.args[0].shape == (2, 2)
    assert call.kwargs["axis"].shape == "scalar"


def test_unsupported_statements_are_skipped():
    _, symbols = infer_source("x = 1\nfor i in range(3):\n    x = x * 2\ny = x")
    assert symbols == {"x": "scalar", "y": "scalar"}


# --- Derived shapes ---

def shape_of(setup, expr):
    """Shape of `expr` after running the `setup` statements."""
    _, symbols = infer_source(f"{setup}\n__result = {expr}")
    return symbols["__result"]


A23 = "a = np.zeros((2, 3))"


@pytest.mark.parametrize("op", ["+", "-", "*", "/", "**"])
def test_elementwise_scalar_with_array(op):
    assert shape_of(A23, f"2 {op} a") == (2, 3)
    assert shape_of(A23, f"a {op} 2") == (2, 3)


def test_elementwise_scalar_with_scalar():
    assert shape_of("x = 1", "x * 2.5") == "scalar"


def test_elementwise_same_shape():
    assert shape_of(A23, "a + a") == (2, 3)


def test_broadcasting_column_and_row():
    assert shape_of("c = np.zeros((3, 1))\nr = np.ones(4)", "c + r") == (3, 4)


def test_broadcasting_pads_leading_dims():
    assert shape_of(A23 + "\nr = np.array([1, 2, 3])", "a * r") == (2, 3)


def test_broadcasting_incompatible_is_none():
    assert shape_of("x = np.zeros(3)\ny = np.zeros(4)", "x + y") is None
    assert shape_of(A23 + "\nb = np.zeros((3, 2))", "a - b") is None


def test_broadcasting_keeps_unknown_dims_unknown():
    # a[:n] with unknown n has an unknown first dim
    setup = "a = np.zeros((5, 3))\nb = a[:n]"
    assert shape_of(setup, "b") == (None, 3)
    assert shape_of(setup, "b + np.ones(3)") == (None, 3)


def test_elementwise_with_unknown_operand_is_none():
    assert shape_of(A23, "a + unknown") is None


@pytest.mark.parametrize("left, right, expected", [
    ("np.ones(3)", "np.ones(3)", "scalar"),
    ("np.ones((2, 3))", "np.ones(3)", (2,)),
    ("np.ones(3)", "np.ones((3, 4))", (4,)),
    ("np.ones((2, 3))", "np.ones((3, 4))", (2, 4)),
])
def test_matmul(left, right, expected):
    assert shape_of("", f"{left} @ {right}") == expected


@pytest.mark.parametrize("left, right", [
    ("np.ones((2, 3))", "np.ones((2, 3))"),  # Inner dims 3 vs 2
    ("np.ones(3)", "np.ones(4)"),
    ("2", "np.ones(3)"),  # Scalars aren't allowed with @
])
def test_matmul_invalid_is_none(left, right):
    assert shape_of("", f"{left} @ {right}") is None


@pytest.mark.parametrize("op, expected", [("neg", (2, 3)), ("pos", (2, 3)), ("not", None)])
def test_unary_op(op, expected):
    # The frontend only folds negative constants, so build UnaryOp directly
    module, _ = lower(A23)
    module.body.append(Assign([Name("y")], UnaryOp(op, Name("a"))))
    assert infer(module)["y"] == expected


@pytest.mark.parametrize("expr, expected", [
    ("a[0]", (3,)),
    ("a[-1]", (3,)),
    ("a[0, 1]", "scalar"),
    ("a[1, :]", (3,)),
    ("a[:, 0]", (2,)),
    ("a[:, 1:]", (2, 2)),
    ("a[i]", (3,)),  # Scalar-shaped name used as an integer index
])
def test_index(expr, expected):
    assert shape_of(A23 + "\ni = 1", expr) == expected


@pytest.mark.parametrize("expr", [
    "a[2]",  # Out of bounds for dim 2
    "a[-3]",
    "a[1.5]",  # Float index
    "a[0, 0, 0]",  # More indices than dims
    "a[a]",  # Fancy indexing has no rule
    "a[::0]",  # Zero step raises
])
def test_index_invalid_or_unsupported_is_none(expr):
    assert shape_of(A23, expr) is None


@pytest.mark.parametrize("expr, expected", [
    ("v[1:4]", (3,)),
    ("v[::2]", (3,)),
    ("v[-2:]", (2,)),
    ("v[::-1]", (5,)),
    ("v[10:]", (0,)),
    ("v[:n]", (None,)),  # Non-constant bound
])
def test_slice_lengths(expr, expected):
    assert shape_of("v = np.zeros(5)", expr) == expected


def test_slice_lengths_match_python_exactly():
    bounds = [None, *range(-7, 8)]
    steps = [None, -3, -2, -1, 1, 2, 3]
    for start, stop, step in itertools.product(bounds, bounds, steps):
        consts = [None if b is None else Const(b) for b in (start, stop, step)]
        module = Module([
            Assign([Name("v")], ArrayLiteral([Const(i) for i in range(5)])),
            Assign([Name("w")], Index(Name("v"), [Slice(*consts)])),
        ])
        expected = len(np.arange(5)[start:stop:step])
        assert infer(module)["w"] == (expected,), (start, stop, step)


@pytest.mark.parametrize("setup, expected", [
    (A23, (3, 2)),
    ("a = np.zeros(4)", (4,)),
    ("a = 7", "scalar"),
    ("a = np.array([[[1, 2]], [[3, 4]]])", None),  # np.transpose has no 3-D case
])
def test_transpose(setup, expected):
    assert shape_of(setup, "a.T") == expected


def test_shape_attribute_is_none_for_now():
    assert shape_of(A23, "a.shape") is None


@pytest.mark.parametrize("expr, expected", [
    ("np.zeros((2, 3))", (2, 3)),
    ("np.zeros(4)", (4,)),
    ("np.ones(0)", (0,)),
    ("np.eye(3)", (3, 3)),
])
def test_constructor_calls(expr, expected):
    assert shape_of("", expr) == expected


@pytest.mark.parametrize("expr", [
    "np.zeros((2, n))",  # Non-constant dim
    "np.zeros(n)",
    "np.zeros(-1)",  # NumPy raises on negative sizes
    "np.ones(2.5)",
    "np.zeros(())",  # 0-d array, not claimed as a scalar
    "np.ones((4, 1, 2))",  # Only 1-D and 2-D shapes have rules
    "np.sum(np.zeros((2, 2, 2)))",  # No 3-D case
])
def test_constructor_calls_unknown(expr):
    assert shape_of("", expr) is None


def test_tuple_nodes_are_not_arrays():
    module, _ = infer_source("z = np.zeros((2, 3))")
    call = module.body[0].value
    assert call.shape == (2, 3)
    assert call.args[0].shape is None
    assert [e.shape for e in call.args[0].elements] == ["scalar", "scalar"]
