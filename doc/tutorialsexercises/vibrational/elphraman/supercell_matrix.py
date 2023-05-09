from ase.build import bulk
from gpaw import GPAW
from gpaw.raman.elph import EPC

supercell = (2, 2, 2)
atoms = bulk('C', 'diamond', a=3.567)
atoms_sc = atoms * supercell
calc = GPAW(mode='lcao', basis='dzp',
            kpts=(5, 5, 5), xc='PBE',
            symmetry={'point_group': False},
            convergence={'bands': 'nao', 'density': 1e-5},
            parallel={'domain': 1},
            txt='gs_super.txt')
atoms_sc.calc = calc
atoms_sc.get_potential_energy()

# Supercell matrix
elph = EPC(atoms, supercell=supercell)
elph.calculate_supercell_matrix(calc, include_pseudo=True)
