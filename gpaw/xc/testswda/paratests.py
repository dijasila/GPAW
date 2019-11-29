import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw import PW
from gpaw import mpi
from gpaw.blacs import BlacsGrid, BlacsDescriptor, Redistributor

## SIMPLE
# comm = mpi.world

# bg1 = BlacsGrid(comm, comm.size, 1)
# bg2 = BlacsGrid(comm, 1, comm.size)
# md1 = BlacsDescriptor(bg1, 4, 4, 2, 4)
# md2 = BlacsDescriptor(bg2, 4, 4, 4, 2)

# r = Redistributor(comm, md1, md2)

# in_xx = np.arange(0, 16).reshape((4,4)).astype(float)

# rank = comm.rank
# size = comm.size
# myin_xx = in_xx[rank*size:rank*size+size, :].astype(float)
# out_xx = np.zeros((4, 2)).astype(float)
# r.redistribute(myin_xx, out_xx)

# print(f"At rank {rank} in_xx is: \n{myin_xx} \nout_xx is: \n{out_xx}")

# if rank == 0:
#     print(f"Full array is: \n{in_xx}")

## END SIMPLE



atoms = Atoms("H2", positions=[[0,0,0], [0,0,2]], cell=5*np.identity(3))
calc = GPAW(mode=PW(200), xc="WDA_standard", txt=None)
calc.initialize(atoms=atoms)
calc.set_positions(atoms)
xc = calc.hamiltonian.xc

s = mpi.size
n = 4

npts = 4**4

in_isg = np.arange(0 + npts*mpi.rank, npts + npts*mpi.rank).reshape((4, 1, 4, 4, 4))
full_out = np.arange(0, npts*mpi.size).reshape(4*s, 1, 4, 4, 4)

in_ix = np.arange(0, 16).reshape((4, 4)).astype(float)
rank = mpi.rank
size = mpi.size
myin_ix = in_ix[rank*size:rank*size + size, :].astype(float)
# print(f"At rank {rank} myin_ix is: {myin_ix}")
out_ix = np.zeros((in_ix.shape[0], in_ix.shape[1]//size)).astype(float)
# print(f"At rank {rank} out_ix is {out_ix}")

newout = xc.redistribute_i_to_g(myin_ix, out_ix, 2, 4)
# print(f"New out is: {newout}")
print(f"At rank {rank} myin_ix is: \n{myin_ix} \n At rank {rank} out_ix is: \n{out_ix} \n\n\n")


in_isg = np.arange(0, 8).reshape((2, 2, 2)).astype(float)
myin_isg = in_isg[rank: rank + 1, ...].astype(float)

out_isg = np.zeros((2, 1, 2))

o = xc.redistribute_i_to_g(myin_isg, out_isg, 1, 2)
print(f"At rank {rank} myin_isg is: \n{myin_isg}\nAt rank {rank} out_isg is: \n{out_isg}")
# out_isg = np.zeros((4 * s, 1, 4 // s, 4, 4)) 

# out_isg = xc.redistribute_i_to_g(in_isg, out_isg, 4, 4 * s)

# # assert np.allclose(full_

# print(out_isg)


# TODO: ALSO TRY WITH 2 SPINS

