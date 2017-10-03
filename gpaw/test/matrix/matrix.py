import numpy as np
from gpaw.matrix import Matrix, matrix_matrix_multiply as mmm
from gpaw.mpi import world

N = 4
A0 = Matrix(N, N, dist=(world.new_communicator([0]), 1, 1))
A0.array[:] = np.diag(np.arange(N) + 1)
A = Matrix(N, N, dist=(world, world.size, 1))
B = Matrix(N, N, dist=(world, world.size, 1))
C = Matrix(N, N, dist=(world, world.size, 1))
A0.redist(A)
A0.redist(B)
print(A.array)
mmm(2.0, A, 'N', B, 'N', 0.0, C)
print(C.array)
