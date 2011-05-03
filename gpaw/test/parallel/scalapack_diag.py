import numpy as np
from gpaw.blacs import BlacsGrid, parallelprint
from gpaw.mpi import world, rank, size
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities.scalapack import scalapack_diagonalize_dc
from gpaw.blacs import Redistributor

# This test performs diagonalization of a matrix, originally distributed to all the processors, by scalapack


N = 512
assert N % size == 0
# Create a matrix, hermitian
A = np.arange(N**2,dtype=float).reshape(N,N)
for i in range(N):
    for j in range(i,N):
        A[i,j] = A[j,i]

# Distribute the matrix to all the processors
# In reality, A_local is the matrix that is stored and need to be diagonalized.
N_start = rank * N // size
N_end = (rank + 1) * N // size
A_local = A[N_start:N_end, :]

# Create Blacs grid, g1 has the same shape and structure for A_local matrix
# g2 is a new Blacs grid that used for diagonalizing the A matrix
mb = 16
g1 = BlacsGrid(world, size, 1)
g2 = BlacsGrid(world, 2, size//2)
nndesc1 = g1.new_descriptor(N, N, N//size,N)
nndesc2 = g2.new_descriptor(N, N, mb, mb)

# Distribute A to accomodate blacs grid g2
B = nndesc2.empty(dtype=float)
redistributor = Redistributor(world, nndesc1, nndesc2)
redistributor.redistribute(A_local, B)

# Diagonalize, C is the eigenvectors and eps is the eigenvalues
C = nndesc2.zeros()
eps = np.zeros(N)
nndesc2.diagonalize_dc(B, C, eps, 'L')

# After the diagonalization, the eigenvectors C needs to redistribute to the shape of the original A_local matrix
C_local = np.zeros_like(A_local)
redistributor = Redistributor(world, nndesc2, nndesc1)
redistributor.redistribute(C, C_local)

# make sure that the scalapack gives the same result as the lapack
eps1 = np.zeros(N)
diagonalize(A,eps1)
A_local =  A[N_start:N_end, :]
assert np.abs(eps - eps1).sum() < 1e-6
for i in range(N//size):
    # the eigenvectors are row of the matrix, it can be differ by a minus sign.
    if np.abs(A_local[i,:] - C_local[i,:]).sum() > 1e-6:
        if np.abs(A_local[i,:] + C_local[i,:]).sum() > 1e-6:
            raise ValueError('Check !')
        





