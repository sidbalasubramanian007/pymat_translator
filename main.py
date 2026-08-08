from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pathlib import Path
from typing import Literal
import logging
import os
import json

import sandbox
import verifier
from compiler import CompileResult, compile_python, splice

base_dir = Path(__file__).resolve().parent
env_path = base_dir / ".env"
load_dotenv(dotenv_path=env_path, override=True)

MAX_RETRIES = int(os.environ.get("PYMAT_MAX_RETRIES", "3"))

app = FastAPI(title="PyMat Agentic Translator")

logger = logging.getLogger("pymat")

GEMINI_MODEL = "gemini-2.5-flash"

# Created on first use, so the compiler-only path works without a GEMINI_API_KEY.
# The SDK reads os.environ["GEMINI_API_KEY"] when constructed with no arguments.
_client = None


def get_client():
    global _client
    if _client is None:
        _client = genai.Client()
    return _client

# --- Data Models ---

class TranslationRequest(BaseModel):
    python_code: str = Field(..., json_schema_extra={"example": "import numpy as np\nx = np.array([1, 2, 3])"})
    notes: str | None = None

class TranslationResponse(BaseModel):
    matlab_code: str = Field(description="The converted MATLAB code")
    translation_notes: list[str] = Field(description="Syntax changes made")

class MarkerReplacement(BaseModel):
    id: str = Field(description="The marker id, e.g. U1")
    matlab_lines: list[str] = Field(description="MATLAB lines that replace the marker line")

class HybridResponse(BaseModel):
    # A list rather than a dict keyed by id: structured output handles fixed fields far better
    replacements: list[MarkerReplacement]

class VerificationReport(BaseModel):
    equivalent: bool
    mismatches: list[dict]
    matched_variables: list[str]
    retries_used: int

class CompileStatsModel(BaseModel):
    total: int
    compiled: int
    percent: float

class FinalResponse(BaseModel):
    status: str
    final_matlab_code: str
    verification_report: VerificationReport
    # Which path produced final_matlab_code: the deterministic compiler alone, the compiler with
    # Gemini filling in unsupported statements, or a full Gemini translation
    path: Literal["compiler", "hybrid", "llm"]
    # None when compile_python couldn't run on the source
    compile_stats: CompileStatsModel | None
    llm_calls: int

# --- Helpers ---

class BadGeminiReply(Exception):
    """Gemini answered, but the reply isn't valid JSON for the requested schema."""


def _ask_gemini(prompt: str, schema: type[BaseModel]):
    """One structured-output Gemini call; returns the parsed schema object."""
    response = get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.1
        )
    )
    try:
        return schema(**json.loads(response.text))
    except ValueError as e:  # JSONDecodeError and pydantic's ValidationError are both ValueErrors
        raise BadGeminiReply(str(e)) from e


def _run_and_verify(matlab_code: str, py_data: dict) -> tuple[dict, str | None]:
    """
    Runs MATLAB in the sandbox and compares it with the Python ground truth.
    Returns (verification report, feedback for Gemini or None if equivalent).
    """
    mat_result = sandbox.run_octave_and_collect_data(matlab_code)
    if not mat_result["success"]:
        # Syntax error! Record the crash log so the final report shows it
        ver_report = {
            "equivalent": False,
            "mismatches": [{
                "variable": None,
                "reason": "Octave Crash",
                "details": mat_result["logs"]
            }],
            "matched_variables": []
        }
        return ver_report, f"Error: Your code crashed with this log: {mat_result['logs']}. Please fix it."

    # Consult the Judge (Semantic Verifier)
    ver_report = verifier.verify_semantic_equivalence(py_data, mat_result["data"])
    if ver_report["equivalent"]:
        return ver_report, None
    return ver_report, f"Error: The math is wrong. Mismatches: {ver_report['mismatches']}. Fix the translation."


def _hybrid_prompt(compiled: CompileResult, notes: str | None) -> str:
    snippets = "\n\n".join(
        f"[{u.id}] (Python line {u.lineno}; not compiled because: {u.reason})\n{u.source}"
        for u in compiled.unsupported
    )
    shapes = "\n".join(f"  {name}: {shape}" for name, shape in compiled.symbol_shapes.items())
    return (
        "You are an expert Python-to-MATLAB compiler engineer. A deterministic compiler translated "
        "most of a NumPy program to MATLAB, but left marker lines of the form "
        "'% UNSUPPORTED[Uk]: <python source>' where it could not translate a statement.\n\n"
        "Translate ONLY the marked Python statements below. Return one entry per marker id with the "
        "MATLAB lines that replace that marker line. Do not rewrite, repeat or reformat any other "
        "line; the compiled lines stay exactly as they are.\n\n"
        "Conventions used by the compiled code: Python indices are 0-based and MATLAB indices "
        "1-based; 1-D NumPy arrays are MATLAB row vectors; use elementwise operators (.*, ./, .^) "
        "unless both sides are scalars.\n\n"
        f"User Notes: {notes}\n\n"
        f"Unsupported Python statements:\n{snippets}\n\n"
        f"Compiled MATLAB, with the markers in place:\n{compiled.matlab_code}\n"
        "Inferred shapes of variables at the end of the program ('scalar', or dims with None for "
        f"unknown sizes):\n{shapes}\n"
    )


def _response(status, code, ver_report, attempts, path, compiled, llm_calls) -> FinalResponse:
    return FinalResponse(
        status=status,
        final_matlab_code=code,
        verification_report=VerificationReport(
            equivalent=ver_report["equivalent"],
            mismatches=[] if ver_report["equivalent"] else ver_report["mismatches"],
            matched_variables=ver_report["matched_variables"],
            retries_used=attempts
        ),
        path=path,
        compile_stats=CompileStatsModel(**vars(compiled.stats)) if compiled else None,
        llm_calls=llm_calls,
    )

# --- Core Agentic Loop ---

@app.post("/api/v1/translate_and_verify", response_model=FinalResponse)
async def translate_and_verify(request: TranslationRequest):
    current_attempt = 0
    llm_calls = 0

    # We build a continuous prompt string to maintain conversation history
    conversation_context = (
        "You are an expert Python-to-MATLAB compiler engineer. "
        "Translate the code. Account for 0-vs-1 indexing and tensor shapes.\n\n"
        f"User Notes: {request.notes}\n\nPython Code:\n{request.python_code}\n"
    )

    # Step 1: Run the Ground Truth Python execution once
    py_result = sandbox.run_python_and_collect_data(request.python_code)
    if not py_result["success"]:
        raise HTTPException(status_code=400, detail=f"Invalid Python Code: {py_result['logs']}")

    # Defaults in case every attempt crashes before reaching the verifier
    matlab_candidate = ""
    ver_report = {"equivalent": False, "mismatches": [], "matched_variables": []}

    # Step 2: Compiler first. If it can't run, or its attempt fails, use the full LLM loop.
    path = "llm"
    compiled = None
    try:
        compiled = compile_python(request.python_code)
    except Exception:
        logger.exception("compile_python failed; using the full LLM translation")

    if compiled is not None:
        feedback = None
        matlab_candidate = compiled.matlab_code
        if not compiled.unsupported:
            path = "compiler"
        else:
            # Hybrid: Gemini fills in only the marker lines; compiled lines are never sent for rewriting
            path = "hybrid"
            llm_calls += 1
            try:
                hybrid = _ask_gemini(_hybrid_prompt(compiled, request.notes), HybridResponse)
            except BadGeminiReply as e:
                # A malformed reply counts as a failed attempt, not a server error
                feedback = f"Error: the replacements could not be parsed: {e}"
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            if feedback is None:
                replacements = {r.id: r.matlab_lines for r in hybrid.replacements}
                missing = [u.id for u in compiled.unsupported if u.id not in replacements]
                if missing:
                    feedback = f"Error: no replacement was given for {missing}."
                else:
                    matlab_candidate = splice(compiled.matlab_code, replacements)

        if feedback is None:
            ver_report, feedback = _run_and_verify(matlab_candidate, py_result["data"])
            if feedback is None:
                return _response("Success", matlab_candidate, ver_report, current_attempt, path, compiled, llm_calls)

        # Fall back to the full LLM correction loop, with the compiled code as a starting hint
        logger.warning("%s path failed; falling back to full LLM translation: %s", path, feedback)
        conversation_context += (
            "\nA deterministic compiler produced the MATLAB below as a starting point, but it failed:\n"
            f"{matlab_candidate}\n{feedback}\n"
        )
        path = "llm"
        current_attempt += 1

    # Step 3: The Agentic Loop
    while current_attempt <= MAX_RETRIES:
        try:
            # 1. Ask Gemini for translation using Structured Outputs
            llm_calls += 1
            llm_output = _ask_gemini(conversation_context, TranslationResponse)
            matlab_candidate = llm_output.matlab_code

            # Append the attempt to context in case we need to self-correct
            conversation_context += f"\nYour MATLAB code attempt:\n{matlab_candidate}\n"

            # 2. Run the generated MATLAB code in the Sandbox and verify it
            ver_report, feedback = _run_and_verify(matlab_candidate, py_result["data"])
            if feedback is None:
                # VICTORY! The code works and matches perfectly.
                return _response("Success", matlab_candidate, ver_report, current_attempt, path, compiled, llm_calls)

            # Crash log or exact tensor deviation goes back to Gemini
            conversation_context += f"\n{feedback}\n"
            current_attempt += 1

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # If the loop exhausts retries and still fails
    return _response("Failed after retries", matlab_candidate, ver_report, current_attempt, path, compiled, llm_calls)
