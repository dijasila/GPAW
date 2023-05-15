import numpy as np
from ase import Atoms
from gpaw import GPAW, FermiDirac
from gpaw import PW
from gpaw.bztools import find_high_symmetry_monkhorst_pack


a = 2.5
c = 3.22

# Graphene:
atoms = Atoms(
    symbols='C2', positions=[(0.5 * a, -np.sqrt(3) / 6 * a, 0.0),
                             (0.5 * a, np.sqrt(3) / 6 * a, 0.0)],
    cell=[(0.5 * a, -0.5 * 3**0.5 * a, 0),
          (0.5 * a, 0.5 * 3**0.5 * a, 0),
          (0.0, 0.0, c * 2.0)],
    pbc=[True, True, False])
# (Note: the cell length in z direction is actually arbitrary)

atoms.center(axis=2)

calc = GPAW(h=0.18,
            mode=PW(400),
            kpts={'density': 10.0, 'gamma': True},
            occupations=FermiDirac(0.1))

atoms.calc = calc
atoms.get_potential_energy()
calc.write('gs.gpw')

kpts = find_high_symmetry_monkhorst_pack('gs.gpw', density=30)
responseGS = GPAW('gs.gpw').fixed_density(
    kpts=kpts,
    parallel={'band': 1},
    nbands=30,
    occupations=FermiDirac(0.001),
    convergence={'bands': 20})

responseGS.get_potential_energy()
responseGS.write('gsresponse.gpw', 'all')
