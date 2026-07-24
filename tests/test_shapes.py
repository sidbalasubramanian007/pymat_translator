import pytest

from compiler.frontend import lower
from compiler.ir import ArrayLiteral, Assign, Const, Module, Name
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
    "y = np.zeros((2, 3))",
    "y = a[0]",
    "y = a.T",
])
def test_other_nodes_are_unknown_for_now(source):
    _, symbols = infer_source(source)
    assert symbols == {"y": None}


def test_nested_literal_inside_other_node_is_annotated():
    module, _ = infer_source("s = np.sum(np.array([[1, 2], [3, 4]]), axis=0)")
    call = module.body[0].value
    assert call.shape is None
    assert call.args[0].shape == (2, 2)
    assert call.kwargs["axis"].shape == "scalar"


def test_unsupported_statements_are_skipped():
    _, symbols = infer_source("x = 1\nfor i in range(3):\n    x = x * 2\ny = x")
    assert symbols == {"x": "scalar", "y": "scalar"}
