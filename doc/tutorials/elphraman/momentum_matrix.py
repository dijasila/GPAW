from ase.build import bulk
from gpaw import GPAW
from gpaw.raman.dipoletransition import get_momentum_transitions

atoms = bulk('C', 'diamond', a=3.567)
calc = GPAW(mode='lcao', basis='dzp',
            kpts=(5, 5, 5), xc='PBE',
            symmetry={'point_group': False},
            convergence={'bands': 'nao', 'density': 1e-5},
            parallel={'band': 1},
            txt='mom.txt')

atoms.calc = calc
atoms.get_potential_energy()
atoms.calc.write("gs.gpw", mode="all")
get_momentum_transitions(calc.wfs, savetofile=True)
