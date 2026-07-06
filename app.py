import streamlit as st
import requests

# Set up the page layout
st.set_page_config(page_title="PyMat Translator", layout="wide")

st.title("🚀 PyMat Agentic Translator")
st.markdown("Deterministic Python-to-MATLAB compilation using Agentic Reflection.")
st.markdown("---")

# Create a split-screen layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Python (Input)")
    default_code = "import numpy as np\nx = np.array([1, 2, 3])"
    python_code = st.text_area("Paste your Python code here:", value=default_code, height=300)
    
    # The submit button
    if st.button("Translate & Verify", type="primary", use_container_width=True):
        with st.spinner("Executing Sandboxes & Verifying Math..."):
            try:
                # Send the code to your running FastAPI backend
                response = requests.post(
                    "http://127.0.0.1:8000/api/v1/translate_and_verify",
                    json={"python_code": python_code, "notes": ""}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Store the results in session state so they persist on the screen
                    st.session_state['matlab'] = data['final_matlab_code']
                    st.session_state['report'] = data['verification_report']
                else:
                    st.error(f"Backend Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect. Is your FastAPI server (uvicorn) running?")

with col2:
    st.subheader("MATLAB (Verified Output)")
    
    # Only show output if we have a successful response stored
    if 'matlab' in st.session_state:
        st.code(st.session_state['matlab'], language="matlab")
        
        report = st.session_state['report']
        
        # Verification Badges
        if report['equivalent']:
            st.success(f"✅ Mathematically Verified! (Self-Correction Retries: {report['retries_used']})")
            if report['matched_variables']:
                st.info(f"Matched Tensors: {', '.join(report['matched_variables'])}")
        else:
            st.error("❌ Verification Failed (Math Mismatch)")
            st.json(report['mismatches'])
    else:
        st.info("Awaiting compilation...")