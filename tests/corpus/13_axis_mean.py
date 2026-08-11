import numpy as np
A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
overall = np.mean(A)
col_means = np.mean(A, axis=0)
row_means = np.mean(A, axis=1)
