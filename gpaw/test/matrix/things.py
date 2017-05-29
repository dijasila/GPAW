# import functools
import numpy as np
# from gpaw.mpi import world
# from gpaw.fd_operators import Laplace
from gpaw.grid_descriptor import GridDescriptor
from gpaw.matrix import (Matrix, UniformGridFunctions, AtomBlockMatrix,
                         AtomCenteredFunctions, PlaneWaveExpansions,
                         UniformGridDensity)
from gpaw.spline import Spline


def test(desc, kd, spositions, proj, basis, dS_aii,
         bcomm=None,
         spinpolarized=False, collinear=True, kpt=None, dtype=None):
    phi_M = AtomCenteredFunctions(basis, desc)  # , [kpt])
    phi_M.positions = spositions
    nbands = len(phi_M)
    if desc.mode == 'fd':
        create = UniformGridFunctions
    else:
        create = PlaneWaveExpansions
    psi_n = create(desc, nbands, dtype=dtype, kpt=kpt,
                   collinear=collinear, dist=bcomm)
    psi_n[:] = phi_M
    S_nn = (psi_n.C * psi_n).integrate()
    pt_I = AtomCenteredFunctions(proj, [kpt])
    pt_I.positions = spositions
    P_In = Matrix((len(pt_I), len(psi_n)), dtype=dtype, dist=(bcomm, 1, -1))
    P_In[:] = (pt_I.C * psi_n).integrate()
    dSP_In = P_In.new()
    dS_II = AtomBlockMatrix((len(pt_I), len(pt_I)), dS_aii)
    dSP_In[:] = dS_II * P_In
    S_nn += P_In.H * dSP_In
    S_nn.cholesky()
    S_nn.inv()
    psi2_n = psi_n.new()
    psi2_n[:] = S_nn * psi_n
    S_nn[:] = psi2_n * psi2_n
    P_In[:] = pt_I.C * psi2_n
    dSP_In[:] = dS_II * P_In
    S_nn += P_In.H * dSP_In
    print(S_nn.array)

    n = UniformGridDensity(desc, spinpolarized, collinear)
    f_n = np.ones(len(psi_n))  # ???
    n.from_wave_functions(psi2_n, f_n)
    n.integrate()

    # kin(psit2_n, psit_n)
    # H_nn = Matrix((nbands, nbands), dtype, dist=?????)
    # H_nn = psit2_n.C *


spos = [(0.5, 0.5, 0.5)]
size = (10, 12, 14)
a = 3.5
cell = [a, a, a]
gd = GridDescriptor(size, cell)
gd.mode = 'fd'

p = Spline(0, 1.2, [1, 0.6, 0.1, 0.0])
b = Spline(0, 1.7, [1, 0.6, 0.1, 0.0])

proj = [[p]]
basis = [[b]]
dS_aii = {0: np.array([[0.3]])}

test(gd, None, spos, proj, basis, dS_aii)
