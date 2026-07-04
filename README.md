# PyMat Agentic Translator

An automated, self-correcting AI compiler that translates Python/NumPy scripts into mathematically verified MATLAB/Octave code. 

Unlike standard LLM wrappers, this pipeline utilizes **Agentic Reflection** and **Isolated Container Execution**. It does not just guess the syntax; it runs both the original and generated code in Docker sandboxes, compares the output tensors mathematically, and forces the LLM to fix its own code if the standard deviation exceeds floating-point tolerances.

## System Architecture

1. **Ingestion & Prompting Layer:** A FastAPI backend enforces structured JSON constraints using Pydantic, passing the Python script to the LLM (Google Gemini 2.5 Flash).
2. **Dual-Execution Sandbox:** The backend orchestrates ephemeral Docker containers (`python:3.10-slim` and `gnuoctave/octave`). It executes both scripts, intercepts runtime standard outputs, and dumps all global arrays/tensors to a serialized volume (`.json` and `.mat`).
3. **Semantic Verification Judge:** A deterministic Python script utilizes `SciPy` and `NumPy` to load the outputs, assert identical matrix dimensionalities, and prove element-wise numerical equality ($\epsilon = 10^{-5}$).
4. **The Agentic Loop:** If the Verifier catches an error (e.g., a 0-vs-1 indexing offset), the exact tensor deviation or stack trace is formatted into a prompt and fed back to the LLM for self-correction until the math matches perfectly.

## Quick Start

### Prerequisites
* Python 3.10+
* Docker Desktop (Running)
* Google Gemini API Key

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/pymat-translator.git](https://github.com/yourusername/pymat-translator.git)
   cd pymat-translator
2. Create a virtual environment and download all the dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set your API key in your terminal session (or .env file):

   export GEMINI_API_KEY="your_api_key_here"

4. Boot the FastAPI server:

   ```bash
   uvicorn main:app --reload
   ```

Open the interactive testing dashboard at http://127.0.0.1:8000/docs.

Submit a Python snippet to the /api/v1/translate_and_verify endpoint and watch the containers orchestrate the verification loop in real-time.
