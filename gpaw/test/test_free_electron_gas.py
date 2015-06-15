from itertools import product

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ase import Atoms
from ase.units import Hartree

from gpaw import GPAW
from gpaw.bztools import get_BZ, tesselate_brillouin_zone

from scipy.spatial import Delaunay, ConvexHull

from ase.lattice import bulk

from gpaw.response.chi0 import Chi0
from gpaw.response.df import DielectricFunction


class FreeElectronChi(Chi0):
    def __init__(self, *args, **kwargs):
        Chi0.__init__(self, *args, **kwargs)

    def get_intraband_response(self, k_v, s, n1=None, n2=None,
                               kd=None, symmetry=None, pd=None):
        return np.ascontiguousarray(np.array((k_v,), complex))

    def get_intraband_eigenvalue(self, k_v, s,
                                 n1=None, n2=None, kd=None, pd=None):
        return np.array(((k_v**2 / 2.).sum(),), float)

    def get_matrix_element(self, k_v, s, n,
                           m1=None, m2=None,
                           pd=None, kd=None,
                           symmetry=None):

        nG = pd.ngmax
        n_mG = np.zeros((m2 - m1, nG + 2), complex)
        return n_mG

    def get_eigenvalues(self, k_v, s, n,
                        m1=None, m2=None,
                        kd=None, pd=None):
        deps_m = np.zeros((m2 - m1), float)
        return deps_m

# Make simple gs calc
atoms = bulk('Na')
atoms.calc = GPAW(mode='pw', kpts={'size': (4, 4, 4)},
                  setups={'Na': '1'})
atoms.get_potential_energy()
atoms.calc.write('Na.gpw', 'all')

# Make refined kpoint grid
rk_kc = tesselate_brillouin_zone('Na.gpw', 5)
responseGS = GPAW('Na.gpw', fixdensity=True,
                  kpts=rk_kc, nbands=10, setups={'Na': '1'})

responseGS.diagonalize_full_hamiltonian(nbands=1, expert=True)
responseGS.write('gsresponse.gpw', 'all')

# Use free electorn chi to calculate free electron
# gas response
cell_cv = responseGS.wfs.gd.cell_cv
N = 1.0
V = np.abs(np.linalg.det(cell_cv))
n = N / V
kf = (3 * np.pi**2 * n)**(1. / 3)

A_cv = cell_cv / (2 * np.pi)
k_c = np.dot(A_cv, np.array([kf, 0, 0], float))
ef = kf**2 / 2.
wp = (4 * np.pi * n)**0.5

df = DielectricFunction('gsresponse.gpw', chi0=FreeElectronChi,
                        ecut=1.0)
df.chi0.pair.fermi_level = ef
df1, df2 = df.get_dielectric_function()

plt.figure()
ax = plt.gca()
ax.set_xlim(0, 10)
ax.set_ylim(-5, 10)
plt.plot(df.chi0.omega_w * Hartree, df2.imag, label='Im')
plt.plot(df.chi0.omega_w * Hartree, df2.real, label='Re')
plt.plot(df.chi0.omega_w * Hartree, - (df2**(-1)).imag, label='EELS')
plt.xlabel('Frequency (eV)')
plt.legend()

plt.savefig('/home/morten/free_electron_gas.pdf', bbox_inches='tight')

print('XXXXXXXXXXXXXX')
print(wp * Hartree)
print(ef * Hartree)

plt.show()
