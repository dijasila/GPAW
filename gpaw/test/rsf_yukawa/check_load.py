"""test correct loading of RSF calculations."""
from ase import Atoms
from ase.units import kcal, mol
from gpaw import GPAW
from gpaw.xc.hybrid import HybridXC
from gpaw.occupations import FermiDirac
from gpaw.eigensolvers import RMMDIIS

h = 0.25
fname = 'O_LCY_PBE.gpw'
work_atom = Atoms('O', [(0, 0, 0)])
work_atom.center(4)
c = {'energy': 0.01, 'eigenstates': 3, 'density': 3}

calculator = GPAW(convergence=c, eigensolver=RMMDIIS(),
                  occupations=FermiDirac(width=0.0, fixmagmom=True),
                  txt='O.LCY_PBE.txt', h=h)
calculator.set(xc=HybridXC('LCY_PBE', omega=0.65))
work_atom.set_initial_magnetic_moments([2.0])
work_atom.set_calculator(calculator)
work_atom.get_potential_energy()
calculator.write(fname)
calc = GPAW(fname)
func = calc.get_xc_functional()
assert func['name'] == 'LCY_PBE', 'wrong name for functional'
assert func['omega'] == 0.65, 'wrong value for RSF omega'
