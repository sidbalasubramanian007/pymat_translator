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
    ("v[0]", "Index"),
    ("np.zeros(3)", "Call"),
    ("A.T", "Attribute"),
    ("v + A.T", "Attribute"),
])
def test_unhandled_nodes_become_placeholders(expr, kind):
    assert emit_line(expr) == f"% UNSUPPORTED: {kind} in assignment to y"


def test_frontend_unsupported_and_surrounding_statements():
    source = "x = 1\nfor i in range(3):\n    x = x * 2\ny = x[0]\nz = x + 1"
    assert emit_source(source) == (
        "x = 1;\n"
        "% UNSUPPORTED (line 2): unsupported statement: For\n"
        "% UNSUPPORTED: Index in assignment to y\n"
        "z = x + 1;\n"
    )
