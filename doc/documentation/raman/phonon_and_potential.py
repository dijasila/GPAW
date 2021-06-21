from ase.build import bulk
from gpaw import GPAW
from gpaw.raman.elph import run_elph

atoms = bulk('C', 'diamond', a=3.567)

calc = GPAW(mode='lcao', basis='dzp',
            kpts=(5, 5, 5), xc='PBE',
            symmetry={'point_group': False},
            convergence={'density': 1e-5},
            txt='phonon_elph.txt')

# Displacements
run_elph(atoms, calc, calculate_forces=True)
