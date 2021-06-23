from ase.build import bulk
from gpaw import GPAW
from gpaw.elph.electronphonon import ElectronPhononCoupling

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
elph = elph = ElectronPhononCoupling(atoms, supercell=supercell)
elph.set_lcao_calculator(calc)
elph.calculate_supercell_matrix(dump=2, include_pseudo=True)

