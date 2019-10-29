import sys
# import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, RMMDIIS, Davidson
from gpaw.hybrids import HybridXC

L = 4.0
a = Atoms('H',
          cell=[L, L, 1],
          pbc=1)
a.center()
a.calc = GPAW(
    mode=PW(400, force_complex_dtype=True),
    eigensolver=Davidson(1),
    kpts={'size': (1, 1, 2), 'gamma': True},
    xc='HSE06')
e1 = a.get_potential_energy()
a *= (1, 1, 2)
es = Davidson(1)
es.keep_htpsit = True
a.calc = GPAW(
    mode=PW(400, force_complex_dtype=True),
    #
    # eigensolver=RMMDIIS(1, niter=1),
    xc='HSE06')
e2 = a.get_potential_energy()


xc = HybridXC('EXX')  # HSE06')
a.calc = GPAW(
    mode=PW(400, force_complex_dtype=True),
    kpts={'size': (1, 1, k), 'gamma': True},
    setups='ae',
    symmetry='off',
    nbands=1,
    eigensolver=es,
    xc=xc)
e = a.get_potential_energy()
