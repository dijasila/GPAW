from ase import Atoms
from gpaw import GPAW
import numpy as np
from gpaw.grid_descriptor import GridDescriptor
from gpaw.pw.descriptor import PWDescriptor
from gpaw.pw.poisson import ChargedReciprocalSpacePoissonSolver as CRSPC


def test_charged_pw_poisson():
    n = 40
    L = 8.0
    gd = GridDescriptor((n, n, n), (L, L, L))
    pd = PWDescriptor(45.0, gd)
    ps = CRSPC(pd, np.zeros(3, bool), 1.0)
    a = ps.alpha
    print(a, ps.rcut, pd, gd)
    C = gd.cell_cv.sum(0) / 2
    rho = -np.exp(-1 / (4 * a) * pd.G2_qG[0] +
                  1j * (pd.get_reciprocal_vectors() @ C)) / gd.dv
    v = np.empty_like(rho)
    e = ps.solve(v, rho)
    print(pd.ifft(v)[20, 20])
    print(e)
    print((a / 2 / np.pi)**0.5)


def test_pw_proton():
    proton = Atoms('H')
    proton.center(vacuum=2.0)
    proton.calc = GPAW(mode='pw', charge=1)
    proton.get_potential_energy()
