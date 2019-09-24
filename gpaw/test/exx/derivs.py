import numpy as np
from ase import Atoms
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.grid_descriptor import GridDescriptor
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb as WSTC
from gpaw.xc.hf import EXX, KPoint
from gpaw.symmetry import Symmetry
from gpaw.wavefunctions.arrays import PlaneWaveExpansionWaveFunctions
from gpaw.wavefunctions.pw import PWDescriptor, PWLFC
from gpaw.projections import Projections
from gpaw.mpi import world
from gpaw.spline import Spline

N = 4
L = 2.
nb = 2
r2 = np.linspace(0, 1, 51)**2
spos_ac = np.zeros((1, 3)) + 0.25


class AP:
    my_indices = [0]

x = 0.0


class Setup:
    Delta_iiL = np.zeros((1, 1, 1)) + 0.1# * x
    X_p = np.zeros(1) + 0.3# * x
    ExxC = -10.0
    ghat_l = [Spline(0, 1.0, 1 - r2 * (1 - 2 * r2))]


gd = GridDescriptor([N, N, N], np.eye(3) * L)
sym = Symmetry([], gd.cell_cv)
kd = KPointDescriptor(None)
kd.set_symmetry(Atoms(pbc=True), sym)
coulomb = WSTC(gd.cell_cv, kd.N_c)
pd = PWDescriptor(10, gd, complex, kd)

data = pd.zeros(nb)
data[0, 1] = 3.0
data[1, 2] = -2.5
psit = PlaneWaveExpansionWaveFunctions(nb, pd, data=data)

proj = Projections(nb, [1], AP(), world, dtype=complex)

pt = PWLFC([[Spline(0, 1.0, 1 - r2 * (1 - 2 * r2))]], pd)
pt.set_positions(spos_ac)

f_n = np.array([1.0, 0.5])
kpt = KPoint(psit, proj, f_n, np.array([0.0, 0.0, 0.0]), 1.0)

xx = EXX(kd, [Setup()], pt, coulomb, spos_ac)

v_nG = pd.zeros(nb)
psit.matrix_elements(pt, out=proj)
VV_aii = [np.einsum('n, ni, nj -> ij', f_n, P_ni, P_ni.conj()) * 0.55
          for a, P_ni in proj.items()]

x = xx.calculate([kpt], [kpt], VV_aii, [v_nG])
v = v_nG[0, 1]
print(v_nG)
print(x)

eps = 0.0001

data[0, 1] = 3 + eps
psit.matrix_elements(pt, out=proj)
VV_aii = [np.einsum('n, ni, nj -> ij', f_n, P_ni, P_ni.conj()) * 0.55
          for a, P_ni in proj.items()]
xp = xx.calculate([kpt], [kpt], VV_aii, [v_nG])

data[0, 1] = 3 - eps
psit.matrix_elements(pt, out=proj)
VV_aii = [np.einsum('n, ni, nj -> ij', f_n, P_ni, P_ni.conj()) * 0.55
          for a, P_ni in proj.items()]
xm = xx.calculate([kpt], [kpt], VV_aii, [v_nG])

d = (xp[0] + xp[1] - xm[0] - xm[1]) / (2 * eps) * N**6 / L**3 / 2
print(v / d)
