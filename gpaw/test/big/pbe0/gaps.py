import ase.units as units
from ase.lattice import bulk

from gpaw.xc.hybridg import HybridXC
from ase.dft.kpoints import ibz_points, get_bandpath, monkhorst_pack
from ase.lattice import bulk
from gpaw import GPAW, PW
from gpaw.xc.tools import vxc
from gpaw.xc.hybridg import HybridXC


data = {
    'C': ['diamond', 3.553],
    'Si': ['diamond', 5.421],
    'GaAs': ['zincblende', 5.640],
    'MgO': ['rocksalt', 4.189],
    'NaCl': ['rocksalt', 5.569],
    'Ar': ['fcc', 5.26]}

k = 6
kpts = monkhorst_pack((k, k, k)) + 0.5 / k

for name in data:
    x, a = data[name]
    atoms = bulk(name, x, a=a)
    atoms.calc = GPAW(xc='PBE',
                      mode=PW(200),
                      parallel=dict(band=1),
                      nbands=-4,
                      convergence=dict(bands=-1),
                      kpts=kpts,
                      txt=name + '.txt')
    atoms.get_potential_energy()
    pbe0 = HybridXC('PBE0', alpha=5.0, bandstructure=True)
    de_skn = vxc(atoms.calc, pbe0) - vxc(atoms.calc, 'PBE')
    ibzk_kc = atoms.calc.get_ibz_k_points()
    n = int(atoms.calc.get_number_of_electrons()) // 2
    gamma = None
    for symbol, k_c in zip('GXL', [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0.5, 0.5)]):
        k = abs(ibzk_kc - k_c).max(1).argmin()
        if gamma is None:
            gamma = atoms.calc.get_eigenvalues(k)[n - 1]
            gamma0 = gamma + de_skn[0, k, n - 1]
        e = atoms.calc.get_eigenvalues(k)[n]
        e0 = e + de_skn[0, k, n]
        print '%4s (G->%s): %5.2f %5.2f' % (name, symbol,
                                            e - gamma, e0 - gamma0)
