from ase import Atoms
from gpaw import GPAW, PW


atoms = Atoms('H', cell=(2,2,2), pbc=True)
atoms.calc = GPAW(mode=PW(200),) #eigensolver='direct', dtype=complex)
atoms.get_potential_energy()

