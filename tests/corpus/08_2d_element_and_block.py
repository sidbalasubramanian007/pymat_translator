import numpy as np
A = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
corner = A[2, 3]
neg_corner = A[-1, -1]
block = A[0:2, 1:3]
last_col = A[:, -1]
