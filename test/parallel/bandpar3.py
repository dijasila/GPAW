from time import time
import sys
import numpy as np
from gpaw import parsize, parsize_bands
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain
from gpaw.operators import Laplace
from gpaw.mpi import world
from gpaw.utilities.blas import gemm
from gpaw.utilities.lapack import inverse_cholesky
import _gpaw

B = parsize_bands   # number of blocks
    
G = 120  # number of grid points (G x G x G)
N = 1000  # number of bands

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
if world.rank == 0: print "created domain_comm and band_comm successfully"
scalapack0_comm = world.new_communicator(np.arange(0, world.size, D))
scalapack1_comm = world.new_communicator(np.arange(0, world.size, D // B))
scalapack0_procs = np.arange(0, world.size, D)
scalapack1_procs = np.arange(0, world.size, D // B)

if world.rank == 0: print scalapack0_procs
if world.rank == 0: print scalapack1_procs
# if world.rank == 127:
#     print world.rank
#     print domain_comm.rank
#     print band_comm.rank
#     print scalapack0_comm.rank
#     print scalapack1_comm.rank
    
# Set up domain and grid descriptors:
domain = Domain((a, a, a))
domain.set_decomposition(domain_comm, parsize, N_c=(G, G, G))
gd = GridDescriptor(domain, (G, G, G))

# Random wave functions:
np.random.seed(world.rank)
psit_mG = np.random.uniform(-0.5, 0.5, size=(M,) + tuple(gd.n_c))
if world.rank == 0:
    print 'Size of wave function array:', psit_mG.shape

# Send and receive buffers:
send_mG = gd.empty(M)
recv_mG = gd.empty(M)

def run():
    S_nn = overlap(psit_mG, send_mG, recv_mG)

    # t1 = time()
    # if world.rank == 0:
    #     inverse_cholesky(S_nn)
    #     C_nn = S_nn
    # else:
    #     C_nn = np.empty((N, N))
    # t2 = time()

    # if world.rank == 0:
    #     print 'Cholesky Time %f' % (t2-t1)
        
    # Distribute matrix:
    # world.broadcast(C_nn, 0)

    # psit_mG[:] = matrix_multiply(C_nn, psit_mG, send_mG, recv_mG)

    if world.rank == 0:
        print 'Made it past matrix multiply'

    # Check:
#    S_nn = overlap(psit_mG, send_mG, recv_mG)

#    Assert below requires more memory.
#    if world.rank == 0:
#        # Fill in upper part:
#        for n in range(N - 1):
#            S_nn[n, n + 1:] = S_nn[n + 1:, n]
#      assert (S_nn.round(7) == np.eye(N)).all()

def overlap(psit_mG, send_mG, recv_mG):
    """Calculate overlap matrix.

    Compute the entire overlap matrix and put the columns
    in the correct order."""

    rank = band_comm.rank
    S_imm = np.empty((B, M, M))
    send_mG[:] = psit_mG

    # Shift wave functions:
    for i in range(B - 1):
        rrequest = band_comm.receive(recv_mG, (rank + 1) % B, 42, False)
        srequest = band_comm.send(send_mG, (rank - 1) % B, 42, False)
        # Index for correct order in S_imm
        j = (rank - i) % B
        gemm(gd.dv, psit_mG, send_mG, 0.0, S_imm[j], 'c')
        band_comm.wait(rrequest)
        band_comm.wait(srequest)
        send_mG, recv_mG = recv_mG, send_mG

    j = (rank - (B - 1)) % B
    gemm(gd.dv, psit_mG, send_mG, 0.0, S_imm[j], 'c')

    # This will put S_imm on every rank
    domain_comm.sum(S_imm)

    # Blocks of matrix on each processor become one matrix.
    # We can think of this as a 1D block matrix 
    S_nm = None; # Otherwise, blacs_redist will complain of UnboundLocalError
    if (scalapack0_comm):
        S_nm = S_imm.reshape(N,M)
    del S_imm

    # Test
    if world.rank == 0: print "before redist"

    # Create a simple matrix, diagonal elements 
    # will equal ranks
    if (scalapack0_comm):
        S_nm = scalapack0_comm.rank*np.eye(N,M,-M*scalapack0_comm.rank)
        # S_nm = np.random.uniform(0,1,(N,M))
        # print scalapack0_comm.rank, S_nm

    # Desc for S_nm
    desc0 = _gpaw.blacs_array(scalapack0_comm,N,N,1,B,N,M)

    # Desc for S_mm
    desc1 = _gpaw.blacs_array(scalapack1_comm,N,N,B,B,64,64)

    # Copy from S_nm -> S_mm
    S_mm = _gpaw.blacs_redist(S_nm,desc0,desc1)
    
    if world.rank == 0: print 'redistributed array'

    # Call new scalapack diagonalize
    W, Z_mm  = _gpaw.scalapack_diagonalize_dc(S_mm, desc1)

    # Save memory
    del S_mm

    # Delete W outside of communicator, mostly for testing
    if (scalapack0_comm == None):
        del W
        
    # if (scalapack1_comm):
    #     print scalapack1_comm.rank, Z_mm
        
    # Copy from Z_mm -> Z_nm 
    S_nm = _gpaw.blacs_redist(Z_mm,desc1,desc0)

    if world.rank == 0: print 'restore original array'
    if (scalapack0_comm):
        print scalapack0_comm.rank, W
                
def matrix_multiply(C_nn, psit_mG, send_mG, recv_mG):
    """Calculate new linear compination of wave functions."""
    rank = band_comm.rank
    C_imim = C_nn.reshape((B, M, B, M))
    send_mG[:] = psit_mG
    psit_mG[:] = 0.0
    beta = 0.0
    for i in range(B - 1):
        rrequest = band_comm.receive(recv_mG, (rank + 1) % B, 117, False)
        srequest = band_comm.send(send_mG, (rank - 1) % B, 117, False)
        gemm(1.0, send_mG, C_imim[rank, :, (rank + i) % B], beta, psit_mG)
        beta = 1.0
        band_comm.wait(rrequest)
        band_comm.wait(srequest)
        send_mG, recv_mG = recv_mG, send_mG
    gemm(1.0, send_mG, C_imim[rank, :, rank - 1], beta, psit_mG)

    return psit_mG

ta = time()

# Do twenty iterations
for x in range(1):
    run()

tb = time()

if world.rank == 0:
    print 'Total Time %f' % (tb -ta)
    
