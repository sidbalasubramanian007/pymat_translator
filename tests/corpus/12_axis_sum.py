import numpy as np
A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
total = np.sum(A)
col_sums = np.sum(A, axis=0)
row_sums = np.sum(A, axis=1)
