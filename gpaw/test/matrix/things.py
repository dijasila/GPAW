import functools
import numpy as np
from gpaw.mpi import world
from gpaw.fd_operators import Laplace
from gpaw.grid_descriptor import GridDescriptor
from gpaw.matrix import Matrix, UniformGridFunctions, AtomBlockMatrix


def test(desc, kd, spositions, proj, basis,
         bcomm,
         spinpolarize=False, collinear=True, kpt=None, dtype=None):
    phi = AtomCenteredFunctions(basis, d, [kpt])
    phi.positions = positions
    nbands = len(phi)
    B = world.size // gd.comm.size
    if desc.mode == 'fd':
        create = UniformGridFunctions
    else:
        create = PlaneWaveExpansions
    psi = create(desc, nbands, dtype=dtype, kpt=kpt,
                 collinear=collinear, dist=bcomm)
    psi[:] = phi
    S_nn = (psi.C * psi).integrate()
    pt = AtomCenteredFunctions(proj, [kpt])
    pt.positions = positions
    P_In = (pt.C * psi).integrate()
    P_In[:] = pt.C * psi
    P_In = Matrix((len(pt), len(psi)), dtype=dtype, dist=bcomm)
    P_In[:] = pt.C * psi
    dSP_In = P_In.new()
    dS_II = AtomBlockMatrix((len(pt), len(pt)), desc,
                            {})
    dSP_In[:] = dS_II * P_In
    S_nn += P_In.H * dSP_In
    S_nn.cholesky()
    S_nn.inv()
    psi2 = psi.new()
    psi2[:] = S_nn * psi
    S_nn[:] = psi2 * psi2
    P_In[:] = pt.C * psi2
    dSP_In[:] = dS_II * P_In
    S_nn += P_In.H * dSP_In
    print(S_nn.array)

    n = Density(desc.gd, spinpolarized, collinear)
    n.from_wave_functions(psi2_n, f_n)
    n.integrate()

    kin(psit2_n, psit_n)
    H_nn = Matrix((nbands, nbands), dtype, dist=?????)
    H_nn = psit2_n.C *
gd = GridDescriptor([2, 3, 4], [2, 3, 4])
dt = complex
ph = np.ones((3, 2), complex)
T = functools.partial(Laplace(gd, -0.5, 1, dt).apply, phase_cd=ph)
N = 2
a =
a.a[:] = 1
a.array[0, 0, world.rank] = -1j
c = Matrix(N, N, dt, dist=(world, world.size))
p = {0: np.arange(10).reshape((2, 5)) * 0.1,
     1: np.arange(10).reshape((2, 5)) * 0.2}
