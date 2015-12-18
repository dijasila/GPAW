from __future__ import print_function

"""Test reading of gpaw output by ase"""

from ase.structure import molecule
from ase.io import read

from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.occupations import FixedOccupations, ZeroKelvin

calculate = 1

if calculate:
    H2 = Cluster(molecule('H2'))
    h=0.4
    H2.minimal_box(2, 0.4)

    c = GPAW(h=h, nbands=2)
    H2.set_calculator(c)

    c.set(txt='no_spin_no_pbc.out')
    H2.get_potential_energy()

    c.set(spinpol=True, txt='spin_no_pbc.out')
    H2.get_potential_energy()

    H2.set_pbc(True)
    c.set(symmetry='off', kpts=(2,3,1), txt='spin_pbc.out')
    H2.get_potential_energy()

    c.set(spinpol=False, txt='no_spin_pbc.out')
    H2.get_potential_energy()

s = read('no_spin_no_pbc.out')
c = s.get_calculator()
assert(len(c.kpts) == 1)

s = read('spin_no_pbc.out')
c = s.get_calculator()
assert(len(c.kpts) == 2)

s = read('no_spin_pbc.out')
c = s.get_calculator()
assert(len(c.kpts) == 6)

s = read('spin_pbc.out')
c = s.get_calculator()
assert(len(c.kpts) == 12)

