from ase.build import bulk

from gpaw import GPAW, FermiDirac
from gpaw.elph.electronphonon import ElectronPhononCoupling

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
calc.write("scf.gpw", 'all')

elph = ElectronPhononCoupling(atoms, atoms.calc, calculate_forces=False)
elph.set_lcao_calculator(atoms.calc)
elph.run()
elph.calculate_supercell_matrix()
