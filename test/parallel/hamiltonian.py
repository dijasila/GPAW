from time import time
import sys
import numpy as np
from gpaw import parsize, parsize_bands
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain
from gpaw.mpi import world
from gpaw.utilities.lapack import inverse_cholesky
from gpaw.operator import Operator
from gpaw.operators import Laplace

B = parsize_bands   # number of blocks
    
G = 120  # number of grid points (G x G x G)
N = 2000  # number of bands

G = 8
N = int(sys.argv[1])
K = int(sys.argv[2])

h = 0.2        # grid spacing
a = h * G      # side length of box
M = N // B     # number of bands per block
assert M * B == N

D = world.size // B  # number of domains
assert D * B == world.size

# Set up communicators:
r = world.rank // D * D
domain_comm = world.new_communicator(np.arange(r, r + D))
band_comm = world.new_communicator(np.arange(world.rank % D, world.size, D))

# Set up domain and grid descriptors:
domain = Domain((a, a, a))
domain.set_decomposition(domain_comm, parsize, N_c=(G, G, G))
gd = GridDescriptor(domain, (G, G, G))

# Random wave functions:
psit_mG = gd.empty(M)
for m in range(M):
    np.random.seed(world.rank * M + m)
    psit_mG[m] = np.random.uniform(-0.5, 0.5, tuple(gd.n_c))
if world.rank == 0:
    print 'Size of wave function array:', psit_mG.shape
P_ani = {0: psit_mG[:, :2, 0, 0].copy(),
         1: psit_mG[:, -1, -1, -3:].copy()}
X = M // K
assert K * X == M
if G**3 // D // K * K < G**3 // D:
    X += 1
print X
work1_xG = gd.empty(X)
work2_xG = gd.empty(X)

kin = Laplace(gd, -0.5, 2).apply
vt_G = gd.empty()
vt_G.fill(0.567)

def run(psit_mG):
    overlap = Operator(band_comm, domain_comm, gd.dv, K)
    overlap.work1_xG = work1_xG
    overlap.work2_xG = work2_xG
    H_nn = np.empty((N, N))
    def H(psit_xG):
        kin(psit_xG, work2_xG)
        for psit_G, y_G in zip(psit_xG, work2_xG):
            y_G += vt_G * psit_G
        return work2_xG
    dH_aii = {0: np.ones((2, 2)) * 0.123, 1: np.ones((3, 3)) * 0.321}
    overlap.calculate_matrix_elements(psit_mG, P_ani, H, dH_aii, H_nn)

    t1 = time()
    if world.rank == 0:
        eps_n, H_nn = np.linalg.eigh(H_nn)
        H_nn = H_nn.T
    t2 = time()

    if world.rank == 0:
        print 'Cholesky Time %f' % (t2-t1)
        print eps_n

    # Distribute matrix:
    world.broadcast(H_nn, 0)

    psit_mG = overlap.matrix_multiply(H_nn, psit_mG, P_ani)

    if world.rank == 0:
        print 'Made it past matrix multiply'

    # Check:
    assert not(P_ani[0] - psit_mG[:, :2, 0, 0]).round(10).any()
    assert not(P_ani[1] - psit_mG[:, -1, -1, -3:]).round(10).any()

    overlap.calculate_matrix_elements(psit_mG, P_ani, H, dH_aii, H_nn)

    if world.rank == 0:
        for n in range(N):
            assert abs(H_nn[n, n] - eps_n[n]) < 1e-10
            assert not H_nn[n + 1:, n].round(10).any()

ta = time()

# Do twenty iterations
for x in range(1):
    run(psit_mG)

tb = time()

if world.rank == 0:
    print 'Total Time %f' % (tb -ta)
    
