import re

import pytest

from compiler import rules
from compiler.emit_matlab import emit
from compiler.frontend import lower
from compiler.shapes import infer

# s is a scalar, m a non-constant scalar, v is (3,), w is (6,), A is (2, 3), B is (3, 2), k is unknown
SETUP = (
    "s = 2\n"
    "m = s + 1\n"
    "v = np.array([1, 2, 3])\n"
    "w = np.array([1, 2, 3, 4, 5, 6])\n"
    "A = np.array([[1, 2, 3], [4, 5, 6]])\n"
    "B = np.array([[1, 2], [3, 4], [5, 6]])\n"
)


def translate(expr):
    """(MATLAB line, inferred shape of y) for `y = <expr>` after the setup."""
    module, unsupported = lower(f"{SETUP}y = {expr}")
    assert unsupported == [], unsupported
    symbols = infer(module)
    return emit(module).splitlines()[-1], symbols["y"]


def check(expr, matlab, shape):
    line, inferred = translate(expr)
    assert line == f"y = {matlab};"
    assert inferred == shape


def check_no_case(expr):
    """Lowered fine, but no case matches: placeholder line and unknown shape."""
    line, inferred = translate(expr)
    func = re.match(r"(np\.\w+)", expr).group(1)
    assert line == f"% UNSUPPORTED: {func} with these argument values or shapes in assignment to y"
    assert inferred is None


# --- Constructors ---

@pytest.mark.parametrize("func", ["zeros", "ones"])
@pytest.mark.parametrize("args, matlab_args, shape", [
    ("3", "1, 3", (3,)),
    ("0", "1, 0", (0,)),
    ("(2, 3)", "2, 3", (2, 3)),
    ("(4,)", "1, 4", (4,)),
    ("m", "1, m", (None,)),
])
def test_zeros_ones(func, args, matlab_args, shape):
    check(f"np.{func}({args})", f"{func}({matlab_args})", shape)


@pytest.mark.parametrize("expr", ["np.zeros(-1)", "np.ones(2.5)", "np.zeros(())", "np.ones((4, 1, 2))", "np.zeros(k)"])
def test_zeros_ones_without_a_case(expr):
    check_no_case(expr)


def test_eye():
    check("np.eye(3)", "eye(3)", (3, 3))
    check("np.eye(m)", "eye(m)", (None, None))
    check_no_case("np.eye(-2)")


@pytest.mark.parametrize("expr, matlab, shape", [
    ("np.arange(5)", "0:4", (5,)),
    ("np.arange(0)", "0:-1", (0,)),
    ("np.arange(2, 7)", "2:6", (5,)),
    ("np.arange(-3, 2)", "-3:1", (5,)),
    ("np.arange(1, 10, 3)", "1:3:9", (3,)),
    ("np.arange(0, 10, 4)", "0:4:9", (3,)),
    ("np.arange(5, 2)", "5:1", (0,)),
])
def test_arange(expr, matlab, shape):
    check(expr, matlab, shape)


def test_arange_in_an_expression_is_parenthesized():
    # MATLAB's colon binds loosest: 0:4 + 1 would mean 0:5
    check("np.arange(5) + 1", "(0:4) + 1", (5,))
    check("2 * np.arange(3)", "2 * (0:2)", (3,))


@pytest.mark.parametrize("expr", [
    "np.arange(m)",  # Non-constant
    "np.arange(0.5, 3)",  # Float
    "np.arange(5, 0, -1)",  # Negative step
    "np.arange(0, 5, 0)",
])
def test_arange_without_a_case(expr):
    check_no_case(expr)


@pytest.mark.parametrize("expr, matlab, shape", [
    ("np.linspace(0, 1, 5)", "linspace(0, 1, 5)", (5,)),
    ("np.linspace(0, 1, num=7)", "linspace(0, 1, 7)", (7,)),
    ("np.linspace(s, 2 * s, m)", "linspace(s, 2 * s, m)", (None,)),
    # NumPy's default is 50 points, MATLAB's 100
    ("np.linspace(0, 1)", "linspace(0, 1, 50)", (50,)),
])
def test_linspace(expr, matlab, shape):
    check(expr, matlab, shape)


def test_linspace_without_a_case():
    check_no_case("np.linspace(v, 1, 5)")  # Array endpoints give a 2-D result
    check_no_case("np.linspace(0, 1, -3)")


# --- Reductions ---

@pytest.mark.parametrize("func", ["sum", "mean", "max", "min"])
def test_reductions(func):
    dim_arg = ", [], " if func in ("max", "min") else ", "
    # 1-D, with no axis or axis=0 (by keyword or position)
    check(f"np.{func}(v)", f"{func}(v)", "scalar")
    check(f"np.{func}(v, axis=0)", f"{func}(v)", "scalar")
    check(f"np.{func}(v, 0)", f"{func}(v)", "scalar")
    # 2-D, no axis: every element
    check(f"np.{func}(A)", f"{func}(A(:))", "scalar")
    check(f"np.{func}(A + 1)", f"{func}(reshape(A + 1, [], 1))", "scalar")
    # 2-D along an axis; axis=1 is transposed back to a row vector
    check(f"np.{func}(A, axis=0)", f"{func}(A{dim_arg}1)", (3,))
    check(f"np.{func}(A, axis=1)", f"{func}(A{dim_arg}2).'", (2,))


@pytest.mark.parametrize("func", ["sum", "mean", "max", "min"])
@pytest.mark.parametrize("args", ["A, axis=-1", "A, axis=2", "v, axis=1", "k", "s"])
def test_reductions_without_a_case(func, args):
    check_no_case(f"np.{func}({args})")


# --- Elementwise functions ---

@pytest.mark.parametrize("func", ["abs", "sqrt", "exp", "log", "sin", "cos"])
def test_elementwise_functions(func):
    check(f"np.{func}(A)", f"{func}(A)", (2, 3))
    check(f"np.{func}(v + 1)", f"{func}(v + 1)", (3,))
    check(f"np.{func}(s)", f"{func}(s)", "scalar")
    check(f"np.{func}(k)", f"{func}(k)", None)


# --- Linear algebra and layout ---

@pytest.mark.parametrize("expr, matlab, shape", [
    ("np.dot(A, B)", "A * B", (2, 2)),
    ("np.dot(A, v)", "(A * v.').'", (2,)),
    ("np.dot(v, B)", "v * B", (2,)),
    ("np.dot(v, v)", "v * v.'", "scalar"),
])
def test_dot_reuses_matmul(expr, matlab, shape):
    check(expr, matlab, shape)


@pytest.mark.parametrize("expr, matlab, shape", [
    ("np.transpose(A)", "A.'", (3, 2)),
    ("A.T", "A.'", (3, 2)),
    ("(A + 1).T", "(A + 1).'", (3, 2)),
    ("A.T @ A", "A.' * A", (3, 3)),
    # 1-D transpose is a no-op; v.' would make a column
    ("v.T", "v", (3,)),
    ("np.transpose(v)", "v", (3,)),
    ("2 * (v + 1).T", "2 * (v + 1)", (3,)),
    ("np.transpose(s)", "s", "scalar"),
])
def test_transpose(expr, matlab, shape):
    check(expr, matlab, shape)


def test_transpose_without_a_case():
    check_no_case("np.transpose(k)")


@pytest.mark.parametrize("expr, matlab, shape", [
    ("np.reshape(w, (2, 3))", "reshape(w.', 3, 2).'", (2, 3)),
    ("w.reshape(2, 3)", "reshape(w.', 3, 2).'", (2, 3)),
    ("w.reshape((3, 2))", "reshape(w.', 2, 3).'", (3, 2)),
    ("A.reshape(3, 2)", "reshape(A.', 2, 3).'", (3, 2)),
    ("np.reshape(A + 1, (3, 2))", "reshape((A + 1).', 2, 3).'", (3, 2)),
])
def test_reshape(expr, matlab, shape):
    check(expr, matlab, shape)


@pytest.mark.parametrize("expr", ["np.reshape(A, 6)", "np.reshape(A, (-1, 2))", "A.reshape(6)", "np.reshape(A, (1, 2, 3))"])
def test_reshape_without_a_case(expr):
    line, inferred = translate(expr)
    assert line == "% UNSUPPORTED: np.reshape with these argument values or shapes in assignment to y"
    assert inferred is None


# --- Calls with no rule, or arguments the rule doesn't accept, are recorded as Unsupported ---

@pytest.mark.parametrize("source, reason", [
    ("y = np.foo(x)", "no translation rule for np.foo"),
    ("y = np.array(x)", "no translation rule for np.array"),
    ("y = np.linalg.inv(x)", "no translation rule for np.linalg.inv"),
    ("y = 1 + np.cumsum(x)", "no translation rule for np.cumsum"),
    ("y = np.zeros(3, dtype=int)", "np.zeros does not accept keyword 'dtype'"),
    ("y = np.eye(3, 4)", "np.eye takes at most 1 positional arguments"),
    ("y = np.eye(3, k=1)", "np.eye does not accept keyword 'k'"),
    ("y = np.sum(x, 0, axis=0)", "np.sum got 'axis' twice"),
    ("y = np.dot(x)", "np.dot needs both arguments"),
])
def test_np_calls_without_a_rule_are_unsupported(source, reason):
    module, unsupported = lower(f"x = 1\n{source}\nz = 2")
    assert [u.reason for u in unsupported] == [reason]
    infer(module)
    assert emit(module) == f"x = 1;\n% UNSUPPORTED (line 2): {reason}\nz = 2;\n"


def test_non_np_method_without_a_rule_stays_a_generic_call():
    line, _ = translate("A.cumsum()")
    assert line == "% UNSUPPORTED: no translation rule for A.cumsum in assignment to y"


# --- rules.yaml itself ---

def test_every_template_placeholder_names_a_param():
    for rule in rules.load().values():
        for case in rule.cases:
            for name in re.findall(r"\{\(?(\w+)", case.matlab):
                assert name in rule.params, (rule.name, case.matlab)


@pytest.mark.parametrize("yaml_text, message", [
    ("np.f:\n  params: [x]\n  bogus: 1\n  cases: [{matlab: f}]\n", "unknown keys"),
    ("np.f:\n  params: [x]\n  cases: [{when: {x: weird}, matlab: f}]\n", "unknown condition"),
    ("np.f:\n  params: [x]\n  cases: [{when: {y: 1d}, matlab: f}]\n", "unknown param"),
    ("np.f:\n  params: [x]\n", "exactly one of"),
])
def test_malformed_rules_are_rejected(tmp_path, yaml_text, message):
    path = tmp_path / "rules.yaml"
    path.write_text(yaml_text)
    with pytest.raises(rules.RuleError, match=message):
        rules.load(path)
