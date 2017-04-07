"""Test Tran Blaha potential."""
from ase.dft.bandgap import get_band_gap
from ase.build import bulk
from gpaw import GPAW, PW

k = 8
atoms = bulk('Si')
atoms.calc = GPAW(mode=PW(300),
                  kpts={'size': (k, k, k), 'gamma': True},
                  xc='TB09',
                  convergence={'bands': -1},
                  txt='si.txt')
e = atoms.get_potential_energy()
gap, kv, kc = get_band_gap(atoms.calc)
c = atoms.calc.hamiltonian.xc.c
print(gap, kv, kc)
print('c:', c)
assert abs(gap - 1.23908911993) < 0.01, gap
assert kv == 0 and kc == 24
assert abs(c - 1.1384016666324328) < 0.01, c
