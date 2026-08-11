"""
Benchmarks the deterministic compiler on tests/corpus/ with no Gemini calls.

For each snippet: compile_python(); if nothing is unsupported, run the Python and the
generated MATLAB in the Docker sandboxes and compare them with the verifier.

Usage:
    python scripts/benchmark.py                      # whole corpus
    python scripts/benchmark.py --only 03_negative_index
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# main.py is deliberately not imported: that's where the Gemini client lives
import sandbox  # noqa: E402
import verifier  # noqa: E402
from compiler import compile_python  # noqa: E402

CORPUS_DIR = ROOT / "tests" / "corpus"
RESULTS_PATH = ROOT / "benchmark_results.json"


def run_snippet(path: Path) -> dict:
    source = path.read_text()
    result = {
        "name": path.stem,
        "compiled": None,       # "yes" | "partial" | "error"
        "octave_ran": None,     # "yes" | "no" | None when not run
        "verified": None,       # "yes" | "no" | None when not run
        "stats": None,
        "unsupported": [],
        "matlab_code": None,
        "compared_variables": [],
        "mismatches": [],
        "error": None,
        "seconds": 0.0,
    }
    start = time.perf_counter()
    try:
        try:
            compiled = compile_python(source)
        except Exception as e:
            result["compiled"] = "error"
            result["error"] = f"compile_python raised {type(e).__name__}: {e}"
            return result

        result["matlab_code"] = compiled.matlab_code
        result["stats"] = vars(compiled.stats)
        result["unsupported"] = [
            {"id": u.id, "lineno": u.lineno, "source": u.source, "reason": u.reason}
            for u in compiled.unsupported
        ]
        if compiled.unsupported:
            result["compiled"] = "partial"
            return result
        result["compiled"] = "yes"

        py_result = sandbox.run_python_and_collect_data(source)
        if not py_result["success"]:
            result["error"] = f"Python sandbox failed ({py_result['error']}): {py_result['logs'].strip()}"
            return result

        mat_result = sandbox.run_octave_and_collect_data(compiled.matlab_code)
        if not mat_result["success"]:
            result["octave_ran"] = "no"
            result["verified"] = "no"
            result["error"] = f"Octave failed ({mat_result['error']}): {mat_result['logs'].strip()}"
            return result
        result["octave_ran"] = "yes"

        report = verifier.verify_semantic_equivalence(py_result["data"], mat_result["data"])
        result["verified"] = "yes" if report["equivalent"] else "no"
        result["compared_variables"] = report["matched_variables"]
        result["mismatches"] = report["mismatches"]
        return result
    finally:
        result["seconds"] = round(time.perf_counter() - start, 2)


def _note(r: dict) -> str:
    if r["error"]:
        return r["error"].splitlines()[0][:70]
    if r["mismatches"]:
        return "; ".join(f"{m['variable']}: {m['reason']}" for m in r["mismatches"])[:70]
    if r["unsupported"]:
        return "unsupported: " + ", ".join(u["source"].splitlines()[0] for u in r["unsupported"])[:57]
    return ""


def print_table(results: list[dict]) -> None:
    headers = ["snippet", "stmts", "compiled", "octave", "verified", "secs", "note"]
    rows = [
        [
            r["name"],
            f"{r['stats']['compiled']}/{r['stats']['total']}" if r["stats"] else "-",
            r["compiled"],
            r["octave_ran"] or "-",
            r["verified"] or "-",
            f"{r['seconds']:.1f}",
            _note(r),
        ]
        for r in results
    ]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))


def summarize(results: list[dict], total_seconds: float) -> dict:
    n = len(results)
    full = [r for r in results if r["compiled"] == "yes"]
    verified = [r for r in full if r["verified"] == "yes"]
    return {
        "snippets": n,
        "fully_compiled": len(full),
        "fully_compiled_percent": round(100 * len(full) / n, 1) if n else 0.0,
        "verified": len(verified),
        "verified_percent_of_fully_compiled": round(100 * len(verified) / len(full), 1) if full else 0.0,
        "unsupported_statements": sum(len(r["unsupported"]) for r in results),
        "total_seconds": round(total_seconds, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", metavar="NAME", help="run one snippet (file name, with or without .py)")
    args = parser.parse_args()

    paths = sorted(CORPUS_DIR.glob("*.py"))
    if args.only:
        paths = [p for p in paths if p.stem == args.only.removesuffix(".py")]
        if not paths:
            print(f"No snippet named {args.only!r} in {CORPUS_DIR}", file=sys.stderr)
            return 2

    start = time.perf_counter()
    results = []
    for path in paths:
        print(f"running {path.stem} ...", file=sys.stderr, flush=True)
        results.append(run_snippet(path))
    summary = summarize(results, time.perf_counter() - start)

    print_table(results)
    print()
    print(f"Fully compiled without LLM: {summary['fully_compiled']}/{summary['snippets']} "
          f"({summary['fully_compiled_percent']}%)")
    print(f"Verified among fully compiled: {summary['verified']}/{summary['fully_compiled']} "
          f"({summary['verified_percent_of_fully_compiled']}%)")
    print(f"Unsupported statements: {summary['unsupported_statements']}")
    print(f"Total time: {summary['total_seconds']}s")

    RESULTS_PATH.write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n")
    print(f"Results written to {RESULTS_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
