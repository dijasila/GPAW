"""test correct loading of hybrid calculations."""
from ase import Atoms
from gpaw import GPAW
from gpaw.xc.hybrid import HybridXC
from gpaw.occupations import FermiDirac
from gpaw.eigensolvers import RMMDIIS

h = 0.25
fname = 'O_PBE0.gpw'
work_atom = Atoms('O', [(0, 0, 0)])
work_atom.center(vacuum=4)
c = {'energy': 0.01, 'eigenstates': 3, 'density': 3}

calc = GPAW(convergence=c, eigensolver=RMMDIIS(),
            occupations=FermiDirac(width=0.0, fixmagmom=True),
            txt='O.PBE0.txt', h=h)
calc.set(xc=HybridXC('PBE0'))
work_atom.set_initial_magnetic_moments([2.0])
work_atom.set_calculator(calc)
work_atom.get_potential_energy()
calc.write(fname)
calcl = GPAW(fname)
func = calcl.get_xc_functional()
assert func['name'] == 'PBE0', 'wrong name for functional'
assert func['hybrid'] == 0.25, 'wrong factor for functional'
