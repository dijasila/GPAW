from ase import Atoms
from gpaw import GPAW

# Oxygen atom:
atom = Atoms('O', cell=[6, 6, 6], pbc=False)
atom.center()

calc = GPAW(h=0.2,
            hund=True,  # assigns the atom its correct magnetic moment
            txt='O.txt')

atom.set_calculator(calc)
atom.get_potential_energy()

# Write wave functions to gpw file
calc.write('O.gpw', mode='all')
