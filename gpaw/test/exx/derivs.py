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
L = 2.0
nb = 2
r2 = np.linspace(0, 1, 51)**2
spos_ac = np.zeros((1, 3)) + 0.25


class AP:
    my_indices = [0]


class Setup:
    Delta_iiL = np.zeros((1, 1, 1)) + 0.1
    X_p = np.zeros(1) + 0.3
    ExxC = -10.0
    ghat_l = [Spline(0, 1.0, 1 - r2 * (1 - 2 * r2))]


gd = GridDescriptor([N, N, N], np.eye(3) * L)
sym = Symmetry([], gd.cell_cv)
kd = KPointDescriptor(None)
kd.set_symmetry(Atoms(pbc=True), sym)
coulomb = WSTC(gd.cell_cv, kd.N_c)
pd = PWDescriptor(10, gd, complex, kd)
data = pd.zeros(nb)
data[0, 1] = 1.0
data[1, 2] = -2.0
psit = PlaneWaveExpansionWaveFunctions(nb, pd, data=data)
proj = Projections(nb, [1], AP(), world, dtype=complex)
pt = PWLFC([[Spline(0, 1.0, 1 - r2 * (1 - 2 * r2))]], pd)
pt.set_positions(spos_ac)
kpt = KPoint(psit, proj, np.array([1.0, 0.5]), np.array([0.0, 0.0, 0.0]), 1.0)
xx = EXX(kd, [Setup()], pt, coulomb, spos_ac)
v_nG = pd.zeros(nb)
VV_aii = [np.zeros((1, 1)) + 0.5]
psit.matrix_elements(pt, out=proj)
x = xx.calculate_energy([kpt], [kpt], VV_aii, [v_nG])
v = v_nG[0, 1]
print(v_nG)
print(x)
eps = 0.001
data[0, 1] = 1 + eps
psit.matrix_elements(pt, out=proj)
xp = xx.calculate_energy([kpt], [kpt], VV_aii, [v_nG])
data[0, 1] = 1 - eps
psit.matrix_elements(pt, out=proj)
xm = xx.calculate_energy([kpt], [kpt], VV_aii, [v_nG])
d = (xp[0] + xp[1] - xm[0] - xm[1]) / (2 * eps) * N**3 * 4
print(v / d)
