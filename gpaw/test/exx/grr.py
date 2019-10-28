import sys
# import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, Davidson
from gpaw.hybrids import HybridXC

k = int(sys.argv[1])
L = 4.0
a = Atoms('H',
          cell=[L, L, 1],
          pbc=1)
a *= (1, 1, 2 // k)
a.center()
es = Davidson(1)
es.keep_htpsit = True
xc = HybridXC('HSE06')
a.calc = GPAW(
    mode=PW(400, force_complex_dtype=True),
    kpts={'size': (1, 1, k), 'gamma': True},
    symmetry='off',
    nbands=1,
    eigensolver=es,
    xc=xc)
e = a.get_potential_energy()
