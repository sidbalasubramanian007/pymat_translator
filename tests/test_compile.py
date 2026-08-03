from compiler import CompileStats, compile_python


def test_fully_supported_program():
    result = compile_python(
        "import numpy as np\n"
        "A = np.array([[1, 2], [3, 4]])\n"
        "v = np.array([1, 2])\n"
        "y = A @ v + 1\n"
        "s = np.sum(y)\n"
    )
    assert result.unsupported == []
    assert result.stats == CompileStats(total=4, compiled=4, percent=100.0)
    assert result.matlab_code == (
        "A = [1, 2; 3, 4];\n"
        "v = [1, 2];\n"
        "y = (A * v.').' + 1;\n"
        "s = sum(y);\n"
    )
    assert result.symbol_shapes == {"A": (2, 2), "v": (2,), "y": (2,), "s": "scalar"}


def test_for_loop_in_the_middle():
    source = (
        "import numpy as np\n"
        "x = np.zeros(3)\n"
        "for i in range(3):\n"
        "    x = x + 1\n"
        "y = x * 2\n"
        "z = np.sum(y)\n"
    )
    result = compile_python(source)

    assert len(result.unsupported) == 1
    entry = result.unsupported[0]
    assert (entry.id, entry.lineno) == ("U1", 3)
    assert entry.source == "for i in range(3):\n    x = x + 1"
    assert entry.reason == "unsupported statement: For"

    # Exactly one marker line, between the statements around the loop
    assert result.matlab_code.splitlines() == [
        "x = zeros(1, 3);",
        "% UNSUPPORTED[U1]: for i in range(3):\\n    x = x + 1",
        "y = x * 2;",
        "z = sum(y);",
    ]
    assert result.stats == CompileStats(total=4, compiled=3, percent=75.0)


def test_frontend_and_emitter_unsupported_get_ids_in_source_order():
    result = compile_python(
        "v = np.arange(4)\n"
        "w = np.cumsum(v)\n"  # No rule: the frontend gives up
        "n = 2\n"
        "u = np.arange(n)\n"  # Rule exists but no case matches: the emitter gives up
        "while n > 0:\n"
        "    n = n - 1\n"
    )
    assert [(u.id, u.lineno, u.reason) for u in result.unsupported] == [
        ("U1", 2, "no translation rule for np.cumsum"),
        ("U2", 4, "np.arange with these argument values or shapes"),
        ("U3", 5, "unsupported statement: While"),
    ]
    assert result.matlab_code.splitlines() == [
        "v = 0:3;",
        "% UNSUPPORTED[U1]: w = np.cumsum(v)",
        "n = 2;",
        "% UNSUPPORTED[U2]: u = np.arange(n)",
        "% UNSUPPORTED[U3]: while n > 0:\\n    n = n - 1",
    ]
    # Every id appears on exactly one line
    for entry in result.unsupported:
        assert sum(f"[{entry.id}]" in line for line in result.matlab_code.splitlines()) == 1
    assert result.stats == CompileStats(total=5, compiled=2, percent=40.0)


def test_percent_is_rounded():
    result = compile_python("a = 1\nb = 2\nfor i in range(2):\n    a = a + i\n")
    assert result.stats == CompileStats(total=3, compiled=2, percent=66.7)


def test_empty_program():
    result = compile_python("import numpy as np\n")
    assert result.matlab_code == ""
    assert result.unsupported == []
    assert result.stats == CompileStats(total=0, compiled=0, percent=100.0)
    assert result.symbol_shapes == {}
