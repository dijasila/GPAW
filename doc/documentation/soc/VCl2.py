from gpaw.new.ase_interface import GPAW
from ase import Atoms
import numpy as np

a = 6.339
d = 1.331

atoms = Atoms('V3Cl6',
              cell=[a, a, 1, 90, 90, 60],
              pbc=[1, 1, 0],
              scaled_positions=[
                  [0, 0, 0],
                  [1 / 3, 1 / 3, 0],
                  [2 / 3, 2 / 3, 0],
                  [0, 2 / 3, d],
                  [0, 1 / 3, -d],
                  [1 / 3, 0, d],
                  [1 / 3, 2 / 3, -d],
                  [2 / 3, 1 / 3, d],
                  [2 / 3, 0, -d]])
atoms.center(axis=2, vacuum=5)

m = 3.0
magmoms = np.zeros((9, 3))
magmoms[0] = [m, 0, 0]
magmoms[1] = [-m / 2, m * 3**0.5 / 2, 0]
magmoms[2] = [-m / 2, -m * 3**0.5 / 2, 0]

atoms.calc = GPAW(mode={'name': 'pw',
                        'ecut': 400},
                  magmoms=magmoms,
                  symmetry='off',
                  kpts={'size': (2, 2, 1), 'gamma': True},
                  parallel={'domain': 1, 'band': 1},
                  txt='VCl2_gs.txt')

atoms.get_potential_energy()
atoms.calc.write('VCl2_gs.gpw')
