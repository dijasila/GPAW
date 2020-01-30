from ase import Atoms
from gpaw import GPAW, PW, Davidson

L = 4.0
a = Atoms('H',
          cell=[L, L, 1],
          pbc=1)
a.center()

a.calc = GPAW(
    mode=PW(400, force_complex_dtype=True),
    parallel={'kpt': 1, 'band': 1},
    eigensolver=Davidson(1),
    kpts={'size': (1, 1, 2), 'gamma': True},
    txt='H.txt',
    xc='HSE06')
e1 = a.get_potential_energy()
eps1 = a.calc.get_eigenvalues(0)[0]

a *= (1, 1, 2)
es = Davidson(1)
es.keep_htpsit = True
a.calc = GPAW(
    mode=PW(400, force_complex_dtype=True),
    txt='H2.txt',
    xc='HSE06')
e2 = a.get_potential_energy()
eps2 = a.calc.get_eigenvalues(0)[0]

assert abs(e2 - 2 * e1) < 0.002
assert abs(eps1 - eps2) < 0.001
