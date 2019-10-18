import sys
# import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, Davidson
from gpaw.xc.hf import Hybrid

k = int(sys.argv[1])
L = 8.0
a = Atoms('H2',
          cell=[L, L, L],
          pbc=1)
a.positions[1, 0] = 0.75
a.center()
es = Davidson(1)
es.keep_htpsit = True
xc = Hybrid('HSE06')
a.calc = GPAW(
    mode=PW(400, force_complex_dtype=True),
    kpts={'size': (1, 1, k), 'gamma': not True},
    symmetry='off',
    nbands=1,
    eigensolver=es,
    xc=xc)
e = a.get_potential_energy()
