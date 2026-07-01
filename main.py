from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pathlib import Path
import os
import json

import sandbox
import verifier

base_dir = Path(__file__).resolve().parent
env_path = base_dir / ".env"
load_dotenv(dotenv_path=env_path, override=True)

app = FastAPI(title="PyMat Agentic Translator")

# The SDK automatically checks os.environ["GEMINI_API_KEY"] if left completely blank!
client = genai.Client()

# --- Data Models ---

class TranslationRequest(BaseModel):
    python_code: str = Field(..., example="import numpy as np\nx = np.array([1, 2, 3])")
    notes: str | None = None

class TranslationResponse(BaseModel):
    matlab_code: str = Field(description="The converted MATLAB code")
    translation_notes: list[str] = Field(description="Syntax changes made")

class VerificationReport(BaseModel):
    equivalent: bool
    mismatches: list[dict]
    matched_variables: list[str]
    retries_used: int

class FinalResponse(BaseModel):
    status: str
    final_matlab_code: str
    verification_report: VerificationReport

# --- Core Agentic Loop ---

@app.post("/api/v1/translate_and_verify", response_model=FinalResponse)
async def translate_and_verify(request: TranslationRequest):
    max_retries = 1
    current_attempt = 0
    
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

    # Step 2: The Agentic Loop
    while current_attempt <= max_retries:
        try:
            # 1. Ask Gemini for translation using Structured Outputs
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=conversation_context,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TranslationResponse,
                    temperature=0.1
                )
            )
            
            # Parse the returned JSON string into our Pydantic model
            raw_dict = json.loads(response.text)
            llm_output = TranslationResponse(**raw_dict)
            matlab_candidate = llm_output.matlab_code
            
            # Append the attempt to context in case we need to self-correct
            conversation_context += f"\nYour MATLAB code attempt:\n{matlab_candidate}\n"

            # 2. Run the generated MATLAB code in the Sandbox
            mat_result = sandbox.run_octave_and_collect_data(matlab_candidate)
            if not mat_result["success"]:
                # Syntax error! Feed the crash log back to Gemini
                conversation_context += f"\nError: Your code crashed with this log: {mat_result['logs']}. Please fix it.\n"
                current_attempt += 1
                continue

            # 3. Consult the Judge (Semantic Verifier)
            ver_report = verifier.verify_semantic_equivalence(py_result["data"], mat_result["data"])
            
            if ver_report["equivalent"]:
                # VICTORY! The code works and matches perfectly.
                return FinalResponse(
                    status="Success",
                    final_matlab_code=matlab_candidate,
                    verification_report=VerificationReport(
                        equivalent=True,
                        mismatches=[],
                        matched_variables=ver_report["matched_variables"],
                        retries_used=current_attempt
                    )
                )
            else:
                # MATH MISMATCH! Feed the exact tensor deviation back to Gemini
                conversation_context += f"\nError: The math is wrong. Mismatches: {ver_report['mismatches']}. Fix the translation.\n"
                current_attempt += 1

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # If the loop exhausts retries and still fails
    return FinalResponse(
        status="Failed after retries",
        final_matlab_code=matlab_candidate,
        verification_report=VerificationReport(
            equivalent=False,
            mismatches=ver_report["mismatches"],
            matched_variables=ver_report["matched_variables"],
            retries_used=current_attempt
        )
    )