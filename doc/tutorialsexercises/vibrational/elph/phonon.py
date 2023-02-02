from ase.build import bulk
from ase.phonons import Phonons

from gpaw import GPAW, FermiDirac

a = 3.567
atoms = bulk('C', 'diamond', a=a)

calc = GPAW(mode='lcao',
            basis='dzp',
            kpts=(5, 5, 5),
            xc='PBE',
            occupations=FermiDirac(0.01),
            symmetry={'point_group': False},
            convergence={'energy': 2e-5, 'density': 1e-5},
            txt='phonons.txt')

atoms.calc = calc

# Phonon calculator
ph = Phonons(atoms, atoms.calc, supercell=(1, 1, 1), delta=0.01)
ph.run()
