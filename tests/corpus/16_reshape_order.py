import numpy as np
x = np.arange(6)
r23 = x.reshape(2, 3)
r32 = np.reshape(x, (3, 2))
first_row = r23[0]
