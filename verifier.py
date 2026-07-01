import numpy as np

def verify_semantic_equivalence(python_data: dict, matlab_data: dict, epsilon: float = 1E-5) -> dict:
    """
    Compares extracted variables from Python and MATLAB to ensure mathematical parity.
    Returns a verification report with detailed discrepancies if they exist.
    """
    report = {
        "equivalent": True,
        "mismatches": [],
        "matched_variables": []
    }
    
    # Iterate through variables caught in the Python context
    for var_name, py_info in python_data.items():
        # Check if the same variable name exists in the MATLAB output context
        if var_name not in matlab_data:
            report["equivalent"] = False
            report["mismatches"].append({
                "variable": var_name,
                "reason": "Missing Variable",
                "details": f"Variable '{var_name}' was computed in Python but is missing in the MATLAB output."
            })
            continue
            
        mat_info = matlab_data[var_name]
        
        # 1. Verify structural type matching
        if py_info["type"] != mat_info["type"] and not (py_info["type"] == "ndarray" and mat_info["type"] == "array"):
            report["equivalent"] = False
            report["mismatches"].append({
                "variable": var_name,
                "reason": "Type Mismatch",
                "details": f"Python type is '{py_info['type']}' but MATLAB type is '{mat_info['type']}'."
            })
            continue

        # 2. Verify array dimensionality shapes
        if py_info["type"] == "ndarray":
            py_shape = py_info.get("shape", [])
            mat_shape = mat_info.get("shape", [])
            
            # Account for MATLAB treating 1D arrays as 2D row/column vectors (e.g., [3] vs [1, 3])
            if len(py_shape) == 1 and len(mat_shape) == 2:
                if mat_shape != [1, py_shape[0]] and mat_shape != [py_shape[0], 1]:
                    shape_mismatch = True
                else:
                    shape_mismatch = False
            else:
                shape_mismatch = (py_shape != mat_shape)

            if shape_mismatch:
                report["equivalent"] = False
                report["mismatches"].append({
                    "variable": var_name,
                    "reason": "Shape Mismatch",
                    "details": f"Python shape is {py_shape} while MATLAB shape is {mat_shape}."
                })
                continue

        # 3. Numeric Precision Comparison using NumPy allclose
        try:
            py_arr = np.array(py_info["data"])
            mat_arr = np.array(mat_info["data"])
            
            # Check element-wise equality within floating-point tolerance
            if not np.allclose(py_arr, mat_arr, atol=epsilon):
                diff = np.abs(py_arr - mat_arr)
                max_diff = np.max(diff)
                
                report["equivalent"] = False
                report["mismatches"].append({
                    "variable": var_name,
                    "reason": "Value Divergence",
                    "details": f"Matrix elements deviate outside tolerance. Max delta: {max_diff}"
                })
                continue
                
        except Exception as e:
            report["equivalent"] = False
            report["mismatches"].append({
                "variable": var_name,
                "reason": "Evaluation Error",
                "details": f"Failed to cast arrays for mathematical comparison: {str(e)}"
            })
            continue

        # If it passes all gates, record a successful match
        report["matched_variables"].append(var_name)

    return report

# --- Quick Verification Check ---
if __name__ == "__main__":
    print("Testing Verifier Engine...")
    
    # Mock data representing a subtle 0-vs-1 indexing bug caught during matrix evaluation
    mock_python = {"result": {"type": "ndarray", "shape": [2], "data": [10, 20]}}
    mock_matlab_buggy = {"result": {"type": "ndarray", "shape": [2], "data": [0, 10]}} 
    
    test_report = verify_semantic_equivalence(mock_python, mock_matlab_buggy)
    import json
    print(json.dumps(test_report, indent=2))