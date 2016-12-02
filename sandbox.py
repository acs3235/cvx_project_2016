import numpy as np
indices = [0, 1]
x = np.asarray([1, 2, 3, 4, 5])
H = np.diag(np.ones(len(x)))
H[0][4] = 7
print H
H_sub = H[:,indices]
x_sub = x[indices]

print x_sub

# print H_sub