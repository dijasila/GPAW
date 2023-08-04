from ase.build import bulk
from gpaw import GPAW, FermiDirac


atoms = bulk('Si', 'diamond', a=5.431)

calc = GPAW(mode='lcao', h=0.18, basis='dzp',
            kpts=(11, 11, 11),
            xc='PBE',
            occupations=FermiDirac(0.01),
            symmetry={'point_group': False, 'time_reversal': False},
            convergence={'energy': 2e-5, 'density': 1e-5},
            txt='scf.txt')
atoms.calc = calc
atoms.get_potential_energy()

calc.write("scf.gpw", 'all')
