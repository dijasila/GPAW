"""Calculate PBE0 bandgaps.

The numbers are compared to:

  Hybrid functionals within the all-electron FLAPW method:
  implementation and applications of PBE0

  Markus Betzinger, Christoph Friedrich, Stefan Blügel

  Physical Review B 81, 195117 (2010)
  DOI: 10.1103/PhysRevB.81.195117
"""

import numpy as np
import ase.db
from ase.build import bulk
from ase.dft.kpoints import monkhorst_pack

from gpaw import GPAW, PW
from gpaw.hybrids.eigenvalues import non_self_consistent_eigenvalues as nsceigs


# Data from Betzinger, Friedrich and Blügel:
bfb = {
    'Si': ['diamond', 5.421,
           2.56, 3.96, 2.57, 3.97,
           0.71, 1.93, 0.71, 1.93,
           1.54, 2.87, 1.54, 2.88],
    'C': ['diamond', 3.553,
          5.64, 7.74, 5.59, 7.69,
          4.79, 6.69, 4.76, 6.66,
          8.58, 10.88, 8.46, 10.77],
    'GaAs': ['zincblende', 5.640,
             0.55, 2.02, 0.56, 2.01,
             1.47, 2.69, 1.46, 2.67,
             1.02, 2.38, 1.02, 2.37],
    'MgO': ['rocksalt', 4.189,
            4.84, 7.31, 4.75, 7.24,
            9.15, 11.63, 9.15, 11.67,
            8.01, 10.51, 7.91, 10.38],
    'NaCl': ['rocksalt', 5.569,
             5.08, 7.13, 5.2, 7.26,
             7.39, 9.59, 7.6, 9.66,
             7.29, 9.33, 7.32, 9.41],
    'Ar': ['fcc', 5.26,
           8.71, 11.15, 8.68, 11.09]}

nk = 12
kpts = monkhorst_pack((nk, nk, nk)) + 0.5 / nk

c = ase.db.connect('gaps.db')

for name in ['Si', 'C', 'GaAs', 'MgO', 'NaCl', 'Ar']:
    id = c.reserve(name=name)
    if id is None:
        continue

    x, a = bfb[name][:2]
    atoms = bulk(name, x, a=a)
    atoms.calc = GPAW(xc='PBE',
                      mode=PW(500),
                      parallel=dict(band=1),
                      nbands=-8,
                      convergence=dict(bands=-7),
                      kpts=kpts,
                      txt='%s.txt' % name)

    atoms.get_potential_energy()
    atoms.calc.write(name + '.gpw', mode='all')

    ibzk_kc = atoms.calc.get_ibz_k_points()
    n = int(atoms.calc.get_number_of_electrons()) // 2

    ibzk = []
    for k_c in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0.5, 0.5)]:
        k = abs(ibzk_kc - k_c).max(1).argmin()
        ibzk.append(k)
        if name == 'Ar':
            break

    eps_skn, deps_skn, deps0_skn = nsceigs(name + '.gpw',
                                           'PBE0',
                                           n - 1, n + 1,
                                           ibzk)

    eps0_kn = (eps_skn - deps_skn + deps0_skn)[0]

    data = {}
    for k, point in enumerate('GXL'):
        data[point] = np.array(
            [eps_skn[0, k, 1] - eps_skn[0, 0, 0],
             eps0_kn[k, 1] - eps0_kn[0, 0]] +
            bfb[name][2 + k * 4:6 + k * 4])
        if name == 'Ar':
            break

    c.write(atoms, name=name, data=data)
    del c[id]
