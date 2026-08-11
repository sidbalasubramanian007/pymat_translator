import numpy as np
A = np.array([[3.0, 9.0, 1.0], [7.0, 2.0, 8.0]])
biggest = np.max(A)
smallest = np.min(A)
col_max = np.max(A, axis=0)
row_min = np.min(A, axis=1)
