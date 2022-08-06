"""
from gpaw import GPAW
from ase.build import molecule
from ase import Atoms
from gpaw.new.ase_interface import GPAW

atoms = Atoms('He') # molecule('H2')
atoms.center(vacuum=4)
calc = GPAW(mode='lcaori', xc='EXX')

atoms.calc = calc

atoms.get_potential_energy()


"""
