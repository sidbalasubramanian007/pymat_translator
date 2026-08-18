# PyMat Translator

PyMat translates Python/NumPy scripts to MATLAB/Octave, compiler first. A deterministic compiler translates everything it has rules for. Only the statements it can't handle go to an LLM (Google Gemini 2.5 Flash), and only those statements are rewritten. If the compiler can't run at all, or its output fails verification, the whole program is translated by the LLM in a self-correcting loop. Whichever path produced the code, the result is checked by running it: the original Python and the generated MATLAB both execute in Docker containers, and their top-level variables are compared for type, shape and value. On the benchmark corpus below, 88% of snippets compile without any LLM call, and all of those pass verification.

## Architecture

```
Python source
  │
  ├─► Docker: python:3.10-slim ──────────────────────────────► ground-truth variables
  │
  ▼
compiler/frontend.py    Python AST → IR; unknown constructs become Unsupported records
  ▼
compiler/ir.py          language-neutral nodes: Assign, BinOp, UnaryOp, Index, Slice, Call, ...
  ▼
compiler/shapes.py      infers every expression's shape ("scalar" or dims) in program order
  ▼
compiler/emit_matlab.py MATLAB text, choosing operators and index forms from the shapes;
compiler/rules.yaml     np.* calls are translated by the declarative rule table
  ▼
MATLAB with "% UNSUPPORTED[Uk]: <python>" marker lines (if any)
  ▼
main.py (hybrid)        one Gemini call fills in only the marker lines; compiled lines are kept
  ▼
Docker: gnuoctave/octave ────────────────────────────────────► MATLAB variables
  ▼
verifier.py             same variables? same shapes? values within 1e-5?
  │
  └─ fails ─► full LLM translation, crash log / mismatches fed back, up to PYMAT_MAX_RETRIES
```

**Frontend** ([compiler/frontend.py](compiler/frontend.py)) parses the source with `ast` and lowers it into IR. `import numpy as np` is skipped. Method and attribute aliases declared in the rule table are rewritten to one canonical form, so `x.reshape(2, 3)` becomes `np.reshape(x, (2, 3))`, `x.T` becomes `np.transpose(x)` and `np.dot(a, b)` becomes `a @ b`. A statement that can't be lowered becomes an `Unsupported` record with a reason, and the rest of the program still compiles.

**IR** ([compiler/ir.py](compiler/ir.py)) is independent of both languages. Indices keep Python's convention (0-based, half-open slices), and converting them is the backend's job.

**Shape inference** ([compiler/shapes.py](compiler/shapes.py)) annotates every expression with a shape: `"scalar"`, a tuple of dimensions (where `None` means an unknown size), or `None` when nothing is known. It handles broadcasting, `@`, indexing, slicing and the shape specs in the rule table.

**Emitter** ([compiler/emit_matlab.py](compiler/emit_matlab.py)) writes MATLAB using those shapes and MATLAB's own operator precedence. It picks `*` or `.*` (and `/` or `./`, `^` or `.^`) by checking whether an operand is a scalar. It converts 0-based positions and half-open slices to 1-based inclusive ranges, emits 1-D arrays as row vectors, and adds a `% TODO: verify ...` comment to a matrix product or index whose shapes it couldn't infer. A statement the emitter can't translate becomes a marker line, `% UNSUPPORTED[U1]: <original Python>`.

**Rule table** ([compiler/rules.yaml](compiler/rules.yaml)) lists every supported `np.*` call, with typed argument conditions, a MATLAB template and the result shape. An `np.*` call that has no rule, or whose arguments match no case, is reported as unsupported; the compiler never guesses.

**Hybrid LLM fill-in** ([main.py](main.py)): if there are marker lines, Gemini gets the unsupported Python statements, the compiled MATLAB with markers in place, and the inferred variable shapes. It returns replacement lines for each marker id, which are spliced in. Compiled lines are never sent back for rewriting.

**Dual-execution verifier** ([sandbox.py](sandbox.py), [verifier.py](verifier.py)) runs the Python and the MATLAB in throwaway containers. The Python side dumps top-level variables to JSON; Octave saves its workspace to a `.mat` file. The verifier then compares variable presence, type, shape and values (absolute tolerance `1e-5`). Shape checks accept a 1-D NumPy array against either a MATLAB row or column vector of the same length, and a 1-element array against an Octave scalar.

The API response reports `path`, which says what produced the code: `"compiler"` (no LLM), `"hybrid"` or `"llm"`. It also reports `compile_stats` (statements compiled out of total) and `llm_calls`, and the Streamlit UI shows all three.

## Why this is hard

Every row below comes from the benchmark corpus in [tests/corpus/](tests/corpus/). The MATLAB column is what the compiler emits, and each line passed verification in Octave.

| Python | Naive MATLAB | What the compiler emits | Why |
|---|---|---|---|
| `x[0:3]` | `x(0:3)` | `x(1:3)` | 1-based indexing, and Python's stop is exclusive while MATLAB's end is inclusive |
| `x[2:]` | `x(2:end)` | `x(3:end)` | Same, for an open-ended slice |
| `x[::-1]` | `x(end:1)` | `x(end:-1:1)` | A negative step needs explicit bounds in MATLAB |
| `A * B` (2-D arrays) | `A * B` | `A .* B` | NumPy `*` is elementwise; MATLAB `*` is matrix multiplication |
| `A @ B` | `A .* B` | `A * B` | The reverse case |
| `1 / v` | `1 / v` | `1 ./ v` | MATLAB `1 / v` is a matrix right-division (an error for a 3-element vector), not a reciprocal |
| `v / 2` | `v ./ 2` | `v / 2` | Fine either way, but the compiler only uses `./` when the divisor is an array |
| `x.reshape(2, 3)` | `reshape(x, 2, 3)` | `reshape(x.', 3, 2).'` | NumPy fills row-major, MATLAB column-major; the naive version puts the elements in the wrong cells |
| `A[1]` (2-D `A`) | `A(2)` | `A(2, :)` | A single index on a 2-D NumPy array selects a row; in MATLAB `A(2)` is the second element in column-major order |
| `A[:, 1]` | `A(:, 2)` | `A(:, 2).'` | NumPy gives a 1-D array; MATLAB gives a column, so it's transposed to the row-vector convention |
| `np.sum(A, axis=1)` | `sum(A, 1)` | `sum(A, 2).'` | NumPy axes are 0-based; the result is again turned back into a row |
| `np.max(A, axis=0)` | `max(A, 1)` | `max(A, [], 1)` | `max(A, 1)` is the elementwise max of `A` and the number 1 |
| `np.arange(2, 7)` | `2:7` | `2:6` | `arange` excludes its stop |
| `np.linspace(-1, 1)` | `linspace(-1, 1)` | `linspace(-1, 1, 50)` | Default point counts differ: 50 in NumPy, 100 in MATLAB |
| `-v ** 2` | `-v ^ 2` | `-v .^ 2` | Elementwise power, and both languages parse it as `-(v ** 2)` |

## Benchmark

[scripts/benchmark.py](scripts/benchmark.py) runs the compiler alone (no Gemini) on the 25 snippets in [tests/corpus/](tests/corpus/). Three of them (`23`–`25`) deliberately use unsupported features. For each snippet that compiles completely, the script runs both sides in Docker and verifies them.

Latest run (`benchmark_results.json`):

| Snippet | Statements compiled | Compiled | Octave ran | Verified |
|---|---|---|---|---|
| 01_basic_slice | 4/4 | yes | yes | yes |
| 02_single_index | 3/3 | yes | yes | yes |
| 03_negative_index | 3/3 | yes | yes | yes |
| 04_negative_slice_bounds | 4/4 | yes | yes | yes |
| 05_negative_step | 4/4 | yes | yes | yes |
| 06_step_slice | 4/4 | yes | yes | yes |
| 07_row_vs_column | 4/4 | yes | yes | yes |
| 08_2d_element_and_block | 5/5 | yes | yes | yes |
| 09_matmul_vs_elementwise | 5/5 | yes | yes | yes |
| 10_power_elementwise | 5/5 | yes | yes | yes |
| 11_scalar_array_division | 4/4 | yes | yes | yes |
| 12_axis_sum | 4/4 | yes | yes | yes |
| 13_axis_mean | 4/4 | yes | yes | yes |
| 14_axis_max_min | 5/5 | yes | yes | yes |
| 15_transpose | 4/4 | yes | yes | yes |
| 16_reshape_order | 4/4 | yes | yes | yes |
| 17_broadcast_col_row | 4/4 | yes | yes | yes |
| 18_arange_endpoints | 3/3 | yes | yes | yes |
| 19_linspace_endpoints | 2/2 | yes | yes | yes |
| 20_precedence_chain | 6/6 | yes | yes | yes |
| 21_matrix_vector | 4/4 | yes | yes | yes |
| 22_constructors_math | 5/5 | yes | yes | yes |
| 23_unsupported_for_loop | 1/2 | partial | – | – |
| 24_unsupported_list_comp | 0/2 | partial | – | – |
| 25_unsupported_cumsum | 2/3 | partial | – | – |

- Fully compiled without LLM: **22/25 (88%)**
- Verified among fully compiled: **22/22 (100%)**
- Unsupported statements: 4 (the `for` loop, the list comprehension, `np.cumsum`, and the `np.sum` of the list-comprehension result, whose shape is unknown)
- Total time: 66 s, about 3 s per verified snippet, mostly container start-up and `pip install numpy`

This corpus was written alongside the compiler, so treat these numbers as a regression baseline, not as a measure of how often arbitrary NumPy code will compile.

## Quick start

### Prerequisites

- Docker Desktop, running. The first run pulls `python:3.10-slim` and `gnuoctave/octave:latest`. The Octave image runs as `linux/amd64`, so on Apple Silicon it runs under emulation.
- Python 3.12.
- A Google Gemini API key. You only need it when a program has statements the compiler can't handle, or when the compiled code fails verification.

### Install

```bash
git clone https://github.com/sidbalasubramanian007/pymat_translator.git
cd pymat_translator
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root. `main.py` loads it on start-up:

```
GEMINI_API_KEY=your_api_key_here
```

Optional settings:

| Variable | Default | Meaning |
|---|---|---|
| `PYMAT_MAX_RETRIES` | `3` | Limit on the full-LLM correction loop's attempts |
| `PYMAT_SANDBOX_TIMEOUT` | `60` | Seconds before a sandbox container is killed |

### Run

Start the API in one terminal:

```bash
uvicorn main:app --reload
```

Then start the UI in another:

```bash
streamlit run app.py
```

The UI talks to the API at `http://127.0.0.1:8000`. You can also call `POST /api/v1/translate_and_verify` directly with `{"python_code": "...", "notes": ""}`, or use the interactive docs at http://127.0.0.1:8000/docs.

## Tests and benchmark

Run the unit tests (no Docker or Gemini needed; the pipeline tests stub out both):

```bash
pytest
```

Run the compiler benchmark (needs Docker; never calls Gemini):

```bash
python scripts/benchmark.py                          # all 25 snippets
python scripts/benchmark.py --only 07_row_vs_column  # one snippet
```

The benchmark prints a table and a summary, then writes everything to `benchmark_results.json` (gitignored). That includes the generated MATLAB, unsupported reasons, mismatches and Octave errors. A run with `--only` overwrites the file with just that snippet.

## Adding a NumPy rule

Every `np.*` call the compiler supports is an entry in [compiler/rules.yaml](compiler/rules.yaml). The comment block at the top of that file is the full reference for conditions, template placeholders and shape specs. As an example, here is how to support `np.cumsum` on 1-D arrays (currently unsupported, see `25_unsupported_cumsum`):

```yaml
np.cumsum:
  params: [x]
  cases:
    - when: {x: 1d}
      matlab: "cumsum({x})"
      shape: {same: x}
```

- `params` lists the positional parameters in order. `kwargs` lists the ones that may also be passed by keyword.
- `cases` are tried in order, and the first case whose `when` conditions match is used. Conditions can test an argument's inferred shape (`scalar`, `1d`, `2d`), whether it's present (`present`, `absent`), or its constant value (`int`, `size`, `sizes:2`, `0`, ...). If no case matches, the statement is reported as unsupported. Here, 2-D input stays unsupported on purpose: NumPy's `cumsum` with no axis flattens a 2-D array, while MATLAB's works down the columns.
- `matlab` is the template. `{x}` inserts the argument's MATLAB code, and `{(x)}` adds parentheses when needed (use it before postfix operators like `.'`).
- `shape` tells shape inference the result's shape, so later statements that use the result still compile.
- `method:` and `attribute:` declare aliases such as `x.reshape(...)` and `x.T`. `via: matmul` maps a call to an operator, as `np.dot` does.

Then add a test in [tests/test_rules.py](tests/test_rules.py) or [tests/test_emit.py](tests/test_emit.py), run `pytest`, and add a corpus snippet if the case is worth benchmarking.

## Known limitations

- **Language subset.** Only top-level assignments to a single name are compiled. `for` and `while` loops, `if`, function definitions, comprehensions, augmented assignment (`a += 1`) and indexed assignment (`a[0] = 1`) all go to the LLM.
- **NumPy subset.** The supported calls are `array` (of a list literal), `zeros`, `ones`, `eye`, `arange` (integer constant arguments only), `linspace`, `sum`, `mean`, `max`, `min`, `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `dot`, `transpose`/`.T`, and `reshape`/`.reshape` (2-D target shapes only). Unary `not` and `~` are not supported.
- **Arrays.** At most 2 dimensions. No fancy or boolean indexing. Slice steps must be non-zero constants, and negative-step slices need constant bounds. dtypes are not modelled: everything becomes a MATLAB double.
- **Unknown shapes spread.** If a statement is unsupported, its result has no known shape, so later statements that depend on it can become unsupported too (as `np.sum(squares)` does in `24_unsupported_list_comp`). Where the emitter has to guess an orientation, it adds a `% TODO: verify ...` comment.
- **Broadcasting** of a column against a row (`col + row`) is emitted directly. That relies on implicit expansion, which Octave and MATLAB R2016b+ support but older MATLAB does not.
- **Verification is by example.** It proves the two programs agree on this input, not on all inputs. It only compares top-level variables that the Python harness captures: NumPy arrays, lists and tuples, and plain Python `int`/`float`. NumPy scalar types such as `np.int64` (for example `np.sum` of an integer array) are not captured, so they aren't compared.
- **Hybrid fallback.** If the hybrid fill-in fails verification, the pipeline doesn't retry just the markers. It falls back to a full LLM translation, and the compiled code is passed only as a hint.
- **Sandbox.** Containers are throwaway, but they are not a hardened sandbox: the Python container has network access because it runs `pip install numpy` (unpinned) on every run. That install is also why each verification takes about 3 s.
