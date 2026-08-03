"""
Deterministic Python/NumPy -> MATLAB compiler.

compile_python() runs the whole pipeline: frontend.lower -> shapes.infer ->
emit_matlab.emit. Statements it can't translate are left as marker lines
("% UNSUPPORTED[U1]: <original Python source>") so they can be replaced later.
"""

from __future__ import annotations

from dataclasses import dataclass

from compiler.emit_matlab import emit_with_unsupported
from compiler.frontend import lower
from compiler.ir import Shape, Unsupported
from compiler.shapes import infer


@dataclass
class CompileStats:
    # Statements that would produce MATLAB code; `import numpy as np` isn't counted
    total: int
    # Of those, statements translated without an UNSUPPORTED marker
    compiled: int
    # compiled / total as a percentage, rounded to 1 decimal (100.0 for an empty program)
    percent: float


@dataclass
class CompileResult:
    matlab_code: str
    # In source order, with ids U1, U2, ... matching the markers in matlab_code
    unsupported: list[Unsupported]
    stats: CompileStats
    symbol_shapes: dict[str, Shape | None]


def compile_python(source: str) -> CompileResult:
    module, _ = lower(source)
    symbol_shapes = infer(module)
    # Includes the frontend's unsupported entries as well as statements the emitter couldn't translate
    matlab_code, unsupported = emit_with_unsupported(module)

    total = len(module.body)
    compiled = total - len(unsupported)
    percent = round(100 * compiled / total, 1) if total else 100.0
    return CompileResult(matlab_code, unsupported, CompileStats(total, compiled, percent), symbol_shapes)
