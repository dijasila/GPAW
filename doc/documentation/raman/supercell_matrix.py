from ase.build import bulk

from gpaw import GPAW
from gpaw.raman.elph import calculate_supercell_matrix

atoms = bulk('C', 'diamond', a=3.567)

calc = GPAW(mode='lcao', basis='dzp',
            kpts=(5, 5, 5), xc='PBE',
            symmetry={'point_group': False},
            convergence={'bands': 'nao'},
            txt='supercell.txt')

atoms.calc = calc
atoms.get_potential_energy()
atoms.calc.write("gs.gpw", mode="all")

# Supercell matrix
calculate_supercell_matrix(atoms, calc=calc, dump=2)
