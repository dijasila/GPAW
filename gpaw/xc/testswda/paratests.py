import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw import PW
from gpaw import mpi
from gpaw.blacs import BlacsGrid, BlacsDescriptor, Redistributor

def get_a_density(xc):
    gd = xc.gd.new_descriptor(comm=mpi.serial_comm)
    grid = gd.get_grid_point_coordinates()
    dens = np.zeros(grid.shape[1:])
    densf = np.fft.fftn(dens)
    densf = np.random.rand(*densf.shape) + 0.1
    densf[0,0,0] = densf[0,0,0] + 1.0
    
    dens = np.fft.ifftn(densf).real
    dens = dens + np.min(dens)
    dens[dens < 1e-7] = 1e-8
    norm = xc.gd.integrate(dens)

    nelec = 5
    dens = dens / norm * nelec
    
    assert np.allclose(xc.gd.integrate(dens), nelec)
    
    res = np.array([dens])
    #res[res < 1e-7] = 1e-8
    
    assert (res >= 0).all()
    assert res.ndim == 4
    assert not np.allclose(res, 0)
    return res


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

# in_isg = np.arange(0 + npts*mpi.rank, npts + npts*mpi.rank).reshape((4, 1, 4, 4, 4))
# full_out = np.arange(0, npts*mpi.size).reshape(4*s, 1, 4, 4, 4)

in_ix = np.arange(0, 16).reshape((4, 4)).astype(float)
rank = mpi.rank
size = mpi.size
myin_ix = in_ix[rank*size:rank*size + size, :].astype(float)
# print(f"At rank {rank} myin_ix is: {myin_ix}")
out_ix = np.zeros((in_ix.shape[0], in_ix.shape[1]//size)).astype(float)
# print(f"At rank {rank} out_ix is {out_ix}")

newout = xc.redistribute_i_to_g(myin_ix, 2, 4, rank, size)
# print(f"New out is: {newout}")
# print(f"At rank {rank} myin_ix is: \n{myin_ix} \n At rank {rank} out_ix is: \n{out_ix} \n\n\n")


# in_isg = np.arange(0, 8).reshape((2, 2, 2)).astype(float)
np.random.seed(123)
in_isg = np.random.rand(8).reshape((2, 2, 2)).astype(float)
myin_isg = in_isg[rank: rank + 1, ...].astype(float)

out_isg = np.zeros((2, 1, 2))
out_isg = None

out_ix, mynx, nx = xc.redistribute_i_to_g(myin_isg, 1, 2, rank, size)
out_isg = out_ix.reshape(2, 1, 2)

if size != 2:
    print(f"At rank {rank} myin_isg is: \n{myin_isg}\nAt rank {rank} out_isg is: \n{out_isg}")
    if rank == 0:
        print(f"Full matrix is: {in_isg}")

elif rank == 0:
    # expected_out = np.array([[[0, 1]], [[4, 5]]])
    # expected_out = np.array([in_isg[:, 0, ...]])
    expected_out = in_isg[:, 0, ...].reshape(2, 1, 2)
    # assert np.allclose(expected_out, out_isg), f"Expect: {expected_out.shape}, \nActual: {out_isg.shape}"
    assert np.allclose(expected_out, out_isg), f"Expect: {expected_out}, \nActual: {out_isg}\nFull: {in_isg}"
elif rank == 1:
    # expected_out = np.array([[[2, 3]], [[6, 7]]])
    expected_out = in_isg[:, 1, ...].reshape(2, 1, 2)
    #expected_out = np.array([in_isg[:, 1, ...]])
    # assert np.allclose(expected_out, out_isg), f"Expect: {expected_out.shape}, \nActual: {out_isg.shape}"
    assert np.allclose(expected_out, out_isg), f"Expect: {expected_out}, \nActual: {out_isg}"


myni = 1
ni = 2
mynx = 2
nx = 4

out2_isg = xc.redistribute_g_to_i(out_ix, myin_isg.copy(), myni, ni, mynx, nx, rank, size)

assert np.allclose(myin_isg, out2_isg)





n_g =  get_a_density(xc)

ni_j, ni_l, ni_u, numni = xc.get_ni_grid(rank, size, n_g)

Z_isg, Z_lower, Z_upper = xc.get_Zs(n_g,
                                    ni_j,
                                    ni_l,
                                    ni_u,
                                    xc.gd.get_grid_point_coordinates(),
                                    0,
                                    xc.gd, rank,
                                    size)
nZ_ix, mynx, nx = xc.redistribute_i_to_g(Z_isg, len(ni_j), numni, rank, size)
alpha_ix = xc._get_alphas(nZ_ix)

assert np.allclose(alpha_ix.sum(axis=0), 1), f"Mean sum: {alpha_ix.sum(axis=0).mean()}, std sum: {alpha_ix.sum(axis=0).std()}"







if rank == 0:
    print(f"All tests passed")


# TODO: ALSO TRY WITH 2 SPINS

