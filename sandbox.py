import docker
import os
import tempfile
import json
import scipy.io
import numpy as np
import shutil

# Initialize the Docker client
try:
    client = docker.from_env()
except Exception as e:
    print(f"Error initializing Docker: {e}")

def run_python_and_collect_data(user_code: str) -> dict:
    """
    Wraps the user's Python code to serialize global numerical arrays to JSON,
    executes it inside an isolated container, and returns the data payload.
    """
    wrapper_suffix = """

# --- SYSTEM AUTOMATED DATA COLLECTION HARNESS ---
import json
import numpy as np

extracted_data = {}
# Capture all global variables that look like arrays or matrices
for var_name, var_val in list(globals().items()):
    if var_name.startswith('_') or var_name in ['json', 'np']:
        continue
    
    if isinstance(var_val, (list, tuple)):
        extracted_data[var_name] = {"type": "array", "data": list(var_val)}
    elif isinstance(var_val, np.ndarray):
        extracted_data[var_name] = {
            "type": "ndarray", 
            "shape": list(var_val.shape), 
            "data": var_val.tolist()
        }
    elif isinstance(var_val, (int, float)):
        extracted_data[var_name] = {"type": "scalar", "data": var_val}

with open('/tmp/output_data.json', 'w') as f:
    json.dump(extracted_data, f)
"""
    
    full_code = user_code + wrapper_suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_script:
        temp_script.write(full_code.encode('utf-8'))
        temp_script_path = temp_script.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_json:
        temp_json_path = temp_json.name

    try:
        container_logs = client.containers.run(
            "python:3.10-slim",
            command="sh -c 'pip install numpy > /dev/null 2>&1 && python /tmp/script.py'",
            volumes={
                temp_script_path: {'bind': '/tmp/script.py', 'mode': 'ro'},
                temp_json_path: {'bind': '/tmp/output_data.json', 'mode': 'rw'}
            },
            remove=True,
            stdout=True,
            stderr=True
        )
        
        with open(temp_json_path, 'r') as f:
            content = f.read()
            collected_data = json.loads(content) if content else {}
            
        return {
            "success": True,
            "logs": container_logs.decode('utf-8'),
            "data": collected_data,
            "error": None
        }

    except docker.errors.ContainerError as e:
        return {"success": False, "logs": e.stderr.decode('utf-8'), "data": {}, "error": "Runtime Exception"}
    except Exception as e:
        return {"success": False, "logs": "", "data": {}, "error": str(e)}
    finally:
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)

def run_octave_and_collect_data(user_code: str) -> dict:
    """
    Executes MATLAB/Octave code inside a container, saves the workspace to a .mat file,
    and parses the arrays back into Python.
    """
    # Notice we are saving to a shared directory now, not the temp folder
    wrapper_suffix = "\n\n% --- SYSTEM AUTOMATED DATA COLLECTION HARNESS ---\nsave('-v7', '/shared/output_data.mat');\n"
    full_code = user_code + wrapper_suffix

    # Create a single temporary directory on the Mac host
    host_shared_dir = tempfile.mkdtemp()
    temp_script_path = os.path.join(host_shared_dir, "script.m")
    temp_mat_path = os.path.join(host_shared_dir, "output_data.mat")

    # Write the script into the shared directory
    with open(temp_script_path, 'w') as f:
        f.write(full_code)

    try:
        # Mount the entire directory to '/shared' inside the container
        container_logs = client.containers.run(
            "gnuoctave/octave:latest",
            command="octave --no-gui --quiet /shared/script.m",
            volumes={
                host_shared_dir: {'bind': '/shared', 'mode': 'rw'}
            },
            remove=True,
            stdout=True,
            stderr=True,
            platform="linux/amd64"
        )
        
        # Parse the .mat file using SciPy
        mat_contents = scipy.io.loadmat(temp_mat_path)
        collected_data = {}
        
        # Standardize the output
        for var_name, var_val in mat_contents.items():
            if not var_name.startswith('__'): 
                if isinstance(var_val, np.ndarray):
                    if var_val.size == 1:
                        collected_data[var_name] = {"type": "scalar", "data": var_val.item()}
                    else:
                        collected_data[var_name] = {
                            "type": "ndarray", 
                            "shape": list(var_val.shape), 
                            "data": var_val.tolist()
                        }
                        
        return {
            "success": True,
            "logs": container_logs.decode('utf-8'),
            "data": collected_data,
            "error": None
        }

    except docker.errors.ContainerError as e:
        return {"success": False, "logs": e.stderr.decode('utf-8'), "data": {}, "error": "Runtime Exception"}
    except Exception as e:
        return {"success": False, "logs": "", "data": {}, "error": str(e)}
    finally:
        # Clean up the entire temporary directory from your Mac
        if os.path.exists(host_shared_dir):
            shutil.rmtree(host_shared_dir)

# --- Test harness script ---
if __name__ == "__main__":
    test_code = """
import numpy as np
matrix_a = np.array([[1, 2], [3, 4]])
scaled_matrix = matrix_a * 2
print("Calculations complete within container!")
"""
    print("Running sandbox verification test...")
    result = run_python_and_collect_data(test_code)
    print(json.dumps(result, indent=2))

    # <-- FIXED: Indented this entire block so it only runs when testing directly
    print("\n--- Testing MATLAB/Octave Sandbox ---")
    matlab_test_code = """
matrix_a = [1, 2; 3, 4];
scaled_matrix = matrix_a * 2;
disp('MATLAB logic executed!');
"""
    mat_result = run_octave_and_collect_data(matlab_test_code)
    print(json.dumps(mat_result, indent=2))