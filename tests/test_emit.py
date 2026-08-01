import pytest

from compiler.emit_matlab import emit
from compiler.frontend import lower
from compiler.ir import ArrayLiteral, Assign, BinOp, Const, Module, Name, UnaryOp
from compiler.shapes import infer

# Shared setup: s is a scalar, v and w are 1-D (3,), A is (2, 3), B is (3, 4)
SETUP = (
    "s = 2\n"
    "v = np.array([1, 2, 3])\n"
    "w = np.array([4, 5, 6])\n"
    "A = np.ones((2, 3))\n"
    "B = np.ones((3, 4))\n"
)


def emit_source(source):
    module, _ = lower(source)
    infer(module)
    return emit(module)


def emit_line(expr):
    """MATLAB line for `y = <expr>` after the shared setup."""
    return emit_source(f"{SETUP}y = {expr}").splitlines()[-1]


def emit_ir(*stmts):
    module = Module(list(stmts))
    infer(module)
    return emit(module)


# --- Literals, names, assignment ---

def test_int_const():
    assert emit_source("x = 3") == "x = 3;\n"


@pytest.mark.parametrize("value, text", [("0.1", "0.1"), ("2.5", "2.5"), ("1.0", "1.0"), ("-0.5", "-0.5")])
def test_float_const_uses_repr(value, text):
    assert emit_source(f"x = {value}") == f"x = {text};\n"


@pytest.mark.parametrize("value, text", [(True, "true"), (False, "false")])
def test_bool_const(value, text):
    # The frontend doesn't lower bools yet, so build the IR directly
    assert emit_ir(Assign([Name("x")], Const(value))) == f"x = {text};\n"


def test_1d_array_literal_is_row_vector():
    assert emit_source("a = np.array([1, 2, 3])") == "a = [1, 2, 3];\n"


def test_2d_array_literal():
    assert emit_source("m = np.array([[1, 2], [3, 4]])") == "m = [1, 2; 3, 4];\n"


def test_3d_array_literal_is_placeholder():
    assert emit_source("t = np.array([[[1, 2]], [[3, 4]]])") == (
        "% UNSUPPORTED: ArrayLiteral that is ragged or has more than 2 dimensions in assignment to t\n"
    )


def test_name_and_assign():
    assert emit_source("x = 1\ny = x") == "x = 1;\ny = x;\n"


# --- Elementwise operators ---

@pytest.mark.parametrize("expr, line", [
    ("v + w", "y = v + w;"),
    ("v - w", "y = v - w;"),
    ("v + 2", "y = v + 2;"),
    ("v * 2", "y = v * 2;"),
    ("2 * v", "y = 2 * v;"),
    ("v * w", "y = v .* w;"),
    ("A * A", "y = A .* A;"),
    ("2 / v", "y = 2 ./ v;"),
    ("v / 2", "y = v / 2;"),
    ("v / w", "y = v ./ w;"),
    ("v ** 2", "y = v .^ 2;"),
    ("2 ** v", "y = 2 .^ v;"),
    ("s ** 2", "y = s ^ 2;"),
    ("s * 2", "y = s * 2;"),
    ("s / 2", "y = s / 2;"),
])
def test_elementwise_ops(expr, line):
    assert emit_line(expr) == line


@pytest.mark.parametrize("expr, line", [
    ("p * q", "y = p .* q;"),
    ("p / q", "y = p ./ q;"),
    ("p ** q", "y = p .^ q;"),
    ("p ** 2", "y = p .^ 2;"),  # Both sides must be known scalars for ^
    ("2 / p", "y = 2 ./ p;"),
    # A known scalar on either side is enough for *, and on the right for /
    ("p * 2", "y = p * 2;"),
    ("p / 2", "y = p / 2;"),
])
def test_unknown_shapes_get_dotted_forms(expr, line):
    assert emit_line(expr) == line


# --- Matmul ---

@pytest.mark.parametrize("expr, line", [
    ("A @ B", "y = A * B;"),
    ("A @ v", "y = (A * v.').';"),
    ("v @ B", "y = v * B;"),
    ("v @ w", "y = v * w.';"),
])
def test_matmul(expr, line):
    assert emit_line(expr) == line


def test_matmul_unknown_shapes_gets_todo():
    assert emit_line("p @ q") == "y = p * q;  % TODO: verify matmul orientation"


def test_matmul_with_compound_operands():
    assert emit_line("(A + A) @ (v + w)") == "y = ((A + A) * (v + w).').';"
    assert emit_line("(v + w) @ w") == "y = (v + w) * w.';"


# --- Precedence ---

@pytest.mark.parametrize("expr, line", [
    ("v * (w + 2)", "y = v .* (w + 2);"),
    ("(v + w) * 2", "y = (v + w) * 2;"),
    ("v + w * 2", "y = v + w * 2;"),
    ("v - w - 2", "y = v - w - 2;"),
    ("v - (w - 2)", "y = v - (w - 2);"),
    ("v / (w * 2)", "y = v ./ (w * 2);"),
    # Python's ** is right-associative, MATLAB's ^ is left-associative
    ("s ** 3 ** 2", "y = s ^ (3 ^ 2);"),
    ("(s ** 3) ** 2", "y = s ^ 3 ^ 2;"),
    ("(-1) ** s", "y = (-1) ^ s;"),
    ("s ** -1", "y = s ^ (-1);"),
    ("A @ B * 2", "y = A * B * 2;"),
    ("v @ w + 1", "y = v * w.' + 1;"),
])
def test_precedence(expr, line):
    assert emit_line(expr) == line


# --- UnaryOp (built directly; the frontend only folds negative constants) ---

def test_unary_neg():
    out = emit_ir(Assign([Name("y")], UnaryOp("neg", Name("x"))))
    assert out == "y = -x;\n"


def test_unary_neg_of_sum_is_parenthesized():
    out = emit_ir(Assign([Name("y")], UnaryOp("neg", BinOp("add", Name("a"), Name("b")))))
    assert out == "y = -(a + b);\n"


def test_unary_neg_of_power_needs_no_parens():
    # -x ** 2 in Python is -(x ** 2), which MATLAB's -x .^ 2 also means
    out = emit_ir(Assign([Name("y")], UnaryOp("neg", BinOp("pow", Name("x"), Const(2)))))
    assert out == "y = -x .^ 2;\n"


def test_power_of_unary_neg_is_parenthesized():
    out = emit_ir(Assign([Name("y")], BinOp("pow", UnaryOp("neg", Name("x")), Const(2))))
    assert out == "y = (-x) .^ 2;\n"


# --- Placeholders ---

@pytest.mark.parametrize("expr, kind", [
    ("f(v)", "no translation rule for f"),
    ("A.shape", "Attribute"),
    ("v + A.shape", "Attribute"),
])
def test_unhandled_nodes_become_placeholders(expr, kind):
    assert emit_line(expr) == f"% UNSUPPORTED: {kind} in assignment to y"


def test_frontend_unsupported_and_surrounding_statements():
    source = "x = 1\nfor i in range(3):\n    x = x * 2\ny = x[0]\nz = x + 1"
    assert emit_source(source) == (
        "x = 1;\n"
        "% UNSUPPORTED (line 2): unsupported statement: For\n"
        "% UNSUPPORTED: Index into a scalar in assignment to y\n"
        "z = x + 1;\n"
    )


# --- Indexing ---
# Shared setup: v is (8,), A is (3, 4), i and j are scalars, u has an unknown shape

INDEX_SETUP = (
    "v = np.zeros(8)\n"
    "A = np.zeros((3, 4))\n"
    "i = 1\n"
    "j = 2\n"
)


def emit_index(expr):
    """MATLAB line for `y = <expr>` after the indexing setup."""
    return emit_source(f"{INDEX_SETUP}y = {expr}").splitlines()[-1]


@pytest.mark.parametrize("expr, line", [
    # Constant non-negative indices are folded to 1-based
    ("v[0]", "y = v(1);"),
    ("v[3]", "y = v(4);"),
    # Negative constants count from end
    ("v[-1]", "y = v(end);"),
    ("v[-2]", "y = v(end-1);"),
    # Non-constant expressions get + 1
    ("v[i]", "y = v(i + 1);"),
    ("v[i + j]", "y = v(i + j + 1);"),
    ("v[i - j]", "y = v(i - j + 1);"),
    ("v[2 * i]", "y = v(2 * i + 1);"),
])
def test_integer_indices(expr, line):
    assert emit_index(expr) == line


@pytest.mark.parametrize("expr, line", [
    ("v[2:5]", "y = v(3:5);"),
    ("v[:5]", "y = v(1:5);"),
    ("v[2:]", "y = v(3:end);"),
    ("v[:-1]", "y = v(1:end-1);"),
    ("v[-2:]", "y = v(end-1:end);"),
    ("v[-3:-1]", "y = v(end-2:end-1);"),
    ("v[::2]", "y = v(1:2:end);"),
    ("v[1:7:2]", "y = v(2:2:7);"),
    ("v[i:j]", "y = v(i + 1:j);"),
    ("v[:i + j]", "y = v(1:i + j);"),
])
def test_positive_step_slices(expr, line):
    assert emit_index(expr) == line


@pytest.mark.parametrize("expr, line", [
    ("v[::-1]", "y = v(end:-1:1);"),
    ("v[5:1:-1]", "y = v(6:-1:3);"),
    ("v[5::-1]", "y = v(6:-1:1);"),
    ("v[::-2]", "y = v(end:-2:1);"),
    ("v[6:0:-2]", "y = v(7:-2:2);"),
    ("v[-1:-4:-1]", "y = v(end:-1:end-2);"),
    ("v[:-3:-1]", "y = v(end:-1:end-1);"),
])
def test_negative_step_slices(expr, line):
    assert emit_index(expr) == line


@pytest.mark.parametrize("expr, reason", [
    ("v[i::-1]", "negative-step slice with non-constant bounds"),
    ("v[:j:-1]", "negative-step slice with non-constant bounds"),
    ("v[::i]", "slice with a zero or non-constant step"),
    ("v[::0]", "slice with a zero or non-constant step"),
])
def test_unsupported_slices(expr, reason):
    assert emit_index(expr) == f"% UNSUPPORTED: {reason} in assignment to y"


@pytest.mark.parametrize("expr, line", [
    ("A[1, 2]", "y = A(2, 3);"),
    ("A[i, j]", "y = A(i + 1, j + 1);"),
    ("A[1, :]", "y = A(2, :);"),
    # A[:, j] is 1-D in NumPy, so the MATLAB column is transposed to a row
    ("A[:, 0]", "y = A(:, 1).';"),
    ("A[0:2, -1]", "y = A(1:2, end).';"),
    ("A[-1, 1:3]", "y = A(end, 2:3);"),
    # A single index on a 2-D array selects rows, never a linear index
    ("A[1]", "y = A(2, :);"),
    ("A[1:3]", "y = A(2:3, :);"),
    ("A[-1]", "y = A(end, :);"),
    ("A[::-1]", "y = A(end:-1:1, :);"),
])
def test_2d_indexing(expr, line):
    assert emit_index(expr) == line


@pytest.mark.parametrize("expr", ["v[:]", "A[:, :]", "u[:]"])
def test_full_slice_is_the_whole_array(expr):
    # MATLAB's v(:) would turn a row vector into a column
    assert emit_index(expr) == f"y = {expr[0]};"


@pytest.mark.parametrize("expr, line", [
    # Python clamps out-of-range bounds where MATLAB would raise, so resolve them exactly
    ("v[:9]", "y = v(1:8);"),
    ("v[-10:]", "y = v(1:8);"),
    ("v[2:100:2]", "y = v(3:2:7);"),
    ("v[:-10]", "y = v(1:0);"),  # Empty, and still 1x0 like NumPy's (0,)
    ("v[9::-1]", "y = v(8:-1:1);"),
    ("v[5:-10:-1]", "y = v(6:-1:1);"),
    ("A[1:9]", "y = A(2:3, :);"),
    # Unknown sizes can't be clamped
    ("u[:9]", "y = u(1:9);  % TODO: verify indexing dimensionality"),
])
def test_out_of_range_slice_bounds_are_clamped(expr, line):
    assert emit_index(expr) == line


def test_3d_indexing_is_unsupported():
    # MATLAB keeps singleton dims, so t(2, :, :) would stay 1x3x4
    assert emit_source("t = np.array([[[1, 2]], [[3, 4]]])\ny = t[1]").splitlines()[-1] == (
        "% UNSUPPORTED: Index into an array with more than 2 dimensions in assignment to y"
    )


def test_unknown_shape_single_index_gets_todo():
    assert emit_index("u[0]") == "y = u(1);  % TODO: verify indexing dimensionality"
    assert emit_index("u[1:3]") == "y = u(2:3);  % TODO: verify indexing dimensionality"


def test_unknown_shape_two_indices_has_no_todo():
    assert emit_index("u[0, 1]") == "y = u(1, 2);"
    assert emit_index("u[1, :]") == "y = u(2, :);"


def test_unknown_shape_column_selection_is_transposed_with_todo():
    assert emit_index("u[:, 0]") == "y = u(:, 1).';  % TODO: verify indexing dimensionality"


def test_index_inside_expression():
    assert emit_index("v[0] + A[1, 2] * 2") == "y = v(1) + A(2, 3) * 2;"


@pytest.mark.parametrize("expr, reason", [
    ("A[0, 0, 0]", "Index with more indices than dimensions"),
    ("i[0]", "Index into a scalar"),
    ("v[v]", "array (fancy) index"),
    ("v[1.5]", "non-integer index"),
    ("(v + v)[0]", "Index into an expression"),
])
def test_unsupported_indexing(expr, reason):
    assert emit_index(expr) == f"% UNSUPPORTED: {reason} in assignment to y"
