import numpy as np
from gpaw.matrix import Matrix, matrix_matrix_multiply as mmm
from gpaw.mpi import world

N = 4
G = 7
# A0 = Matrix(N, N, dist=(world.new_communicator([0]), 1, 1))
A0 = Matrix(N, G, dist=(world, 1, 1))
if world.rank == 0:
    A0.array[:, 4:] = 0.0
    A0.array[:, :4] = np.diag(np.arange(N) + 1)
A = Matrix(N, G, dist=(world, world.size, 1))
B = Matrix(N, G, dist=(world, world.size, 1))
C = Matrix(N, N, dist=(world, world.size, 1))
C0 = Matrix(N, N, dist=(world, 1, 1))
A0.redist(A)
A0.redist(B)
print(A.array)
mmm(2.0, A, 'N', B, 'T', 0.0, C)
C.redist(C0)
print(C0.array)
