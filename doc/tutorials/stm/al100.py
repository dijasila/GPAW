from ase.lattice.surface import fcc100
atoms = fcc100('Al', size=(1, 1, 2))
atoms.center(vacuum=4.0, axis=2)

from gpaw import GPAW
calc = GPAW(mode='pw',
            kpts=(4, 4, 1),
            txt='al100.out')
atoms.set_calculator(calc)
energy = atoms.get_potential_energy() 
calc.write('al100.gpw', 'all')
