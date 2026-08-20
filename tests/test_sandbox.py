import json

import sandbox


def collect(user_code, tmp_path):
    """Runs the Python data-collection harness locally (no Docker) and returns what it dumps."""
    out = tmp_path / "output_data.json"
    harness = sandbox.PYTHON_HARNESS.replace("/tmp/output_data.json", str(out))
    exec(user_code + harness, {})
    return json.loads(out.read_text())


def test_numpy_scalars_are_captured(tmp_path):
    data = collect(
        "import numpy as np\n"
        "total = np.sum(np.array([1, 2, 3]))\n"
        "mean = np.mean(np.array([1.0, 2.0]))\n"
        "flag = np.array([1, 2]).any()\n",
        tmp_path,
    )
    assert data["total"] == {"type": "scalar", "data": 6}
    assert data["mean"] == {"type": "scalar", "data": 1.5}
    assert data["flag"] == {"type": "scalar", "data": True}


def test_python_scalars_and_bools_are_captured(tmp_path):
    data = collect("n = 3\nx = 2.5\nok = False\n", tmp_path)
    assert data["n"] == {"type": "scalar", "data": 3}
    assert data["x"] == {"type": "scalar", "data": 2.5}
    assert data["ok"] == {"type": "scalar", "data": False}


def test_arrays_still_captured_and_modules_skipped(tmp_path):
    data = collect("import numpy as np\na = np.array([[1, 2], [3, 4]])\n", tmp_path)
    assert data == {"a": {"type": "ndarray", "shape": [2, 2], "data": [[1, 2], [3, 4]]}}
