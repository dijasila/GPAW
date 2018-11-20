import numpy as np
from gpaw.utilities.elpa import LibElpa
from gpaw.blacs import BlacsGrid
from gpaw.mpi import world

rng = np.random.RandomState(87878787)

if world.size == 1:
    shape = 1, 1
else:
    shape = world.size // 2, 2
bg = BlacsGrid(world, *shape)

M = 4
blocksize = 2

desc = bg.new_descriptor(M, M, blocksize, blocksize)
A = desc.zeros()
A[:] = rng.rand(*A.shape)
A += A.T.copy()
C1 = desc.zeros()
C2 = desc.zeros()
eps1 = np.zeros(M)
eps2 = np.zeros(M)

elpa = LibElpa(desc)
print(elpa)

desc.diagonalize_dc(A.copy(), C1, eps1),

eps = np.zeros(M)
elpa.diagonalize(A.copy(), C2, eps2)
print('eps1', eps1)
print('eps2', eps2)
print(C1)
print('C2', C2)
print(C2.T)

err1 = np.abs(A @ C1.T - eps1 * C1.T).max()

err2 = np.abs(A @ C2.T - eps2 * C2.T).max()

print('err1', err1)
print('err2', err2)
assert err1 < 1e-13, err1
assert err2 < 1e-13, err2

