from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.elph import Supercell

atoms = bulk('Si', 'diamond', a=5.431)
atoms_N = atoms * (3, 3, 3)

calc = GPAW(mode='lcao', h=0.18, basis='dzp',
            kpts=(4, 4, 4),
            xc='PBE',
            occupations=FermiDirac(0.01),
            symmetry={'point_group': False},
            txt='supercell.txt',
            parallel={'domain': 1, 'band': 1})

atoms_N.calc = calc
atoms_N.get_potential_energy()

sc = Supercell(atoms, supercell=(3, 3, 3))
sc.calculate_supercell_matrix(calc, fd_name='elph')
