from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.elph import DisplacementRunner

atoms = bulk('Si', 'diamond', a=5.431)

calc = GPAW(mode='lcao', h=0.18, basis='dzp',
            kpts=(4, 4, 4),
            xc='PBE',
            occupations=FermiDirac(0.01),
            symmetry={'point_group': False},
            convergence={'energy': 2e-5, 'density': 1e-5},
            txt='displacement.txt')

elph = DisplacementRunner(atoms=atoms, calc=calc,
                          supercell=(3, 3, 3), name='elph',
                          calculate_forces=True)
elph.run()
