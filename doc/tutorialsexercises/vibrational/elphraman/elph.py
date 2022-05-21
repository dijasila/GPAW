from ase.build import bulk
from gpaw import GPAW
from gpaw.elph.electronphonon import ElectronPhononCoupling

atoms = bulk('C', 'diamond', a=3.567)
calc = GPAW(mode='lcao', basis='dzp',
            kpts=(3, 3, 3), xc='PBE',
            symmetry={'point_group': False},
            txt='elph.txt')

# Displacements for potential
elph = ElectronPhononCoupling(atoms, calc, supercell=(2, 2, 2),
                              calculate_forces=False)
elph.run()
