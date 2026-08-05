import asyncio
import json

import pytest

import main

PY_DATA = {"x": {"type": "ndarray", "shape": [3], "data": [3.0, 3.0, 3.0]}}


class FakeGemini:
    """Stands in for genai.Client: returns queued JSON replies and records the prompts."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.prompts.append(contents)
        reply = self.replies.pop(0)

        class Response:
            text = reply if isinstance(reply, str) else json.dumps(reply)

        return Response()


@pytest.fixture
def octave_runs(monkeypatch):
    """Fake sandbox: Python always yields PY_DATA; Octave returns queued results (default: matching data)."""
    runs = {"code": [], "results": []}

    def run_python(code):
        return {"success": True, "logs": "", "data": PY_DATA, "error": None}

    def run_octave(code):
        runs["code"].append(code)
        if runs["results"]:
            return runs["results"].pop(0)
        return {"success": True, "logs": "", "data": PY_DATA, "error": None}

    monkeypatch.setattr(main.sandbox, "run_python_and_collect_data", run_python)
    monkeypatch.setattr(main.sandbox, "run_octave_and_collect_data", run_octave)
    return runs


@pytest.fixture
def gemini(monkeypatch):
    """Installs a FakeGemini as the lazy client; call it with the replies to queue."""
    def install(*replies):
        fake = FakeGemini(*replies)
        monkeypatch.setattr(main, "_client", fake)
        return fake
    return install


def translate(python_code):
    return asyncio.run(main.translate_and_verify(main.TranslationRequest(python_code=python_code)))


FULLY_SUPPORTED = "import numpy as np\nx = np.ones(3)\nx = x * 3\n"
WITH_LOOP = "import numpy as np\nx = np.zeros(3)\nfor i in range(3):\n    x = x + 1\ny = x * 2\n"
LOOP_REPLACEMENT = {"replacements": [{"id": "U1", "matlab_lines": ["for i = 0:2", "    x = x + 1;", "end"]}]}


def test_fully_supported_program_needs_no_gemini(octave_runs, gemini, monkeypatch):
    # No client is ever created, so this works with no API key
    def no_client():
        raise AssertionError("Gemini must not be used")
    monkeypatch.setattr(main, "_client", None)
    monkeypatch.setattr(main.genai, "Client", no_client)

    result = translate(FULLY_SUPPORTED)

    assert result.status == "Success"
    assert result.path == "compiler"
    assert result.llm_calls == 0
    assert result.final_matlab_code == "x = ones(1, 3);\nx = x * 3;\n"
    assert octave_runs["code"] == [result.final_matlab_code]
    assert result.compile_stats == main.CompileStatsModel(total=2, compiled=2, percent=100.0)
    assert result.verification_report.retries_used == 0


def test_for_loop_is_filled_in_by_one_gemini_call(octave_runs, gemini):
    fake = gemini(LOOP_REPLACEMENT)

    result = translate(WITH_LOOP)

    assert result.status == "Success"
    assert result.path == "hybrid"
    assert result.llm_calls == 1
    # Compiled lines untouched, the marker replaced by Gemini's lines
    assert result.final_matlab_code == (
        "x = zeros(1, 3);\n"
        "for i = 0:2\n"
        "    x = x + 1;\n"
        "end\n"
        "y = x * 2;\n"
    )
    assert "UNSUPPORTED" not in result.final_matlab_code
    assert octave_runs["code"] == [result.final_matlab_code]
    assert result.compile_stats == main.CompileStatsModel(total=3, compiled=2, percent=66.7)

    # Gemini saw only the snippet to translate, the compiled context and the shapes
    prompt = fake.prompts[0]
    assert "[U1]" in prompt
    assert "for i in range(3):\n    x = x + 1" in prompt
    assert "% UNSUPPORTED[U1]:" in prompt
    assert "x: (3,)" in prompt
    assert "Python Code:" not in prompt  # Not the full-translation prompt


def test_hybrid_mismatch_falls_back_to_full_llm_with_compiled_hint(octave_runs, gemini):
    fake = gemini(
        LOOP_REPLACEMENT,
        {"matlab_code": "x = zeros(1, 3) + 3;\ny = x * 2;", "translation_notes": []},
    )
    octave_runs["results"].append(
        {"success": True, "logs": "", "data": {"x": {"type": "ndarray", "shape": [1, 3], "data": [[0.0, 0.0, 0.0]]}}}
    )

    result = translate(WITH_LOOP)

    assert result.status == "Success"
    assert result.path == "llm"
    assert result.llm_calls == 2
    assert result.final_matlab_code == "x = zeros(1, 3) + 3;\ny = x * 2;"
    assert result.verification_report.retries_used == 1
    # The full-translation prompt carries the spliced hybrid code and why it failed
    fallback_prompt = fake.prompts[1]
    assert "Python Code:" in fallback_prompt
    assert "starting point" in fallback_prompt
    assert "for i = 0:2" in fallback_prompt
    assert "The math is wrong" in fallback_prompt


def test_compiler_crash_falls_back_to_full_llm(octave_runs, gemini):
    fake = gemini({"matlab_code": "x = ones(1, 3) * 3;", "translation_notes": []})
    octave_runs["results"].append({"success": False, "logs": "parse error", "data": {}, "error": "Runtime Exception"})

    result = translate(FULLY_SUPPORTED)

    assert result.path == "llm"
    assert result.llm_calls == 1
    assert result.status == "Success"
    assert "crashed with this log: parse error" in fake.prompts[0]
    assert "x = ones(1, 3);\nx = x * 3;" in fake.prompts[0]


def test_missing_replacement_falls_back_to_full_llm(octave_runs, gemini):
    fake = gemini(
        {"replacements": []},
        {"matlab_code": "x = [3, 3, 3];", "translation_notes": []},
    )

    result = translate(WITH_LOOP)

    assert result.path == "llm"
    assert result.llm_calls == 2
    assert "no replacement was given for ['U1']" in fake.prompts[1]
    # The unspliced compiled code never reached the sandbox
    assert octave_runs["code"] == ["x = [3, 3, 3];"]


def test_malformed_hybrid_reply_falls_back_to_full_llm(octave_runs, gemini):
    fake = gemini("not json", {"matlab_code": "x = [3, 3, 3];", "translation_notes": []})

    result = translate(WITH_LOOP)

    assert result.path == "llm"
    assert result.llm_calls == 2
    assert "could not be parsed" in fake.prompts[1]


def test_compile_error_uses_llm_path(octave_runs, gemini, monkeypatch):
    def broken(source):
        raise RuntimeError("compiler bug")
    monkeypatch.setattr(main, "compile_python", broken)
    fake = gemini({"matlab_code": "x = [3, 3, 3];", "translation_notes": []})

    result = translate(FULLY_SUPPORTED)

    assert result.path == "llm"
    assert result.compile_stats is None
    assert result.llm_calls == 1
    assert "starting point" not in fake.prompts[0]


def test_every_attempt_crashing_reports_the_last_crash(octave_runs, gemini, monkeypatch):
    monkeypatch.setattr(main, "MAX_RETRIES", 2)
    gemini(*[{"matlab_code": "x = [1 2", "translation_notes": []}] * 2)
    crash = {"success": False, "logs": "parse error", "data": {}, "error": "Runtime Exception"}
    octave_runs["results"].extend([crash] * 3)

    result = translate(FULLY_SUPPORTED)

    # Attempt 0 was the compiler; attempts 1 and 2 were Gemini
    assert result.status == "Failed after retries"
    assert result.path == "llm"
    assert result.llm_calls == 2
    assert result.verification_report.retries_used == 3
    assert result.verification_report.mismatches == [{"variable": None, "reason": "Octave Crash", "details": "parse error"}]
