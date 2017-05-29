import numpy as np
from gpaw.matrix import Matrix
N = 4
A = Matrix(N, N, data=np.eye(N))
B = Matrix(N, N, data=np.ones((N, N)))

C = (A * B).eval()
C[:] = A * B
print(C.a)
C += A * B
print(C.a)
