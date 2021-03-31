from ase.build import bulk

from gpaw import GPAW, FermiDirac
from gpaw.raman.elph import run_elph, calculate_supercell_matrix

a = 3.567
atoms = bulk('C', 'diamond', a=a)

calc = GPAW(mode='lcao',
            basis='dzp',
            kpts=(5, 5, 5),
            xc='PBE',
            occupations=FermiDirac(0.01),
            symmetry={'point_group': False},
            convergence={'bands': 'nao'},
            txt='elph.txt')

atoms.calc = calc
atoms.get_potential_energy()
atoms.calc.write("gs.gpw", mode="all")

# Displacements
run_elph(atoms, calc, calculate_forces=False)
# Supercell matrix
calculate_supercell_matrix(atoms, calc=calc, dump=2)
