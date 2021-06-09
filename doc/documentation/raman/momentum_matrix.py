from ase.build import bulk

from gpaw import GPAW
from gpaw.raman.dipoletransition import get_momentum_transitions

atoms = bulk('C', 'diamond', a=3.567)

calc = GPAW(mode='lcao', basis='dzp',
            kpts=(5, 5, 5), xc='PBE',
            symmetry='off',
            convergence={'bands': 'nao'},
            txt='mom.txt')

atoms.calc = calc
atoms.get_potential_energy()

get_momentum_transitions(calc.wfs, savetofile=True)
