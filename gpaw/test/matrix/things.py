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
    phi_M = AtomCenteredFunctions(desc, basis)  # , [kpt])
    phi_M.set_positions(spositions)
    nbands = len(phi_M)
    if desc.mode == 'fd':
        create = UniformGridFunctions
    else:
        create = PlaneWaveExpansions
    psi_n = create(nbands, desc, dtype=dtype, kpt=kpt,
                   collinear=collinear, dist=bcomm)
    psi_n[:] = phi_M
    # psi_n.plot()
    S_nn = psi_n.matrix_elements(psi_n, hermitian=True)
    pt_I = AtomCenteredFunctions(desc, proj)
    pt_I.set_positions(spositions)
    P_In = ProjectionMatrix(len(pt_I), len(psi_n), dtype=dtype, dist=(bcomm, 1, -1))
    pt_I.matrix_elements(psi_n, out=P_In)
    dSP_In = P_In.new()
    dS_II = AtomBlockMatrix(dS_aii)
    dSP_In[:] = dS_II * P_In
    S_nn += P_In.H * dSP_In
    S_nn.cholesky()
    S_nn.inv()
    psi2_n = psi_n.new()
    psi2_n[:] = S_nn.T * psi_n

    (psi2_n.C * psi2_n).integrate(out=S_nn)
    (pt_I.C * psi2_n).integrate(out=P_In)
    dSP_In[:] = dS_II * P_In
    norm = S_nn.a.trace()
    S_nn += P_In.H * dSP_In
    print(S_nn.a)

    nt = UniformGridDensity(desc, spinpolarized, collinear)
    f_n = np.ones(len(psi_n))  # ???
    nt.from_wave_functions(psi2_n, f_n)
    nt.integrate()

    # kin(psit2_n, psit_n)
    # H_nn = Matrix((nbands, nbands), dtype, dist=?????)
    # H_nn = psit2_n.C *


spos = [(0.5, 0.5, 0.5), (0.5, 0.6, 0.75)]
size = np.array([20, 18, 22])
cell = 0.4 * size
gd = GridDescriptor(size, cell)
gd.mode = 'fd'

p = Spline(0, 1.2, [1, 0.6, 0.1, 0.0])
b = Spline(0, 1.7, [1, 0.6, 0.1, 0.0])
b2 = Spline(0, 1.7, [-1, -0.5, -0.2, 0.1, 0.0])

proj = [[p], [p]]
basis = [[b, b2], [b, b2]]
dS_aii = [np.array([[0.3]]), np.array([[0.3]])]

test(gd, None, spos, proj, basis, dS_aii)
