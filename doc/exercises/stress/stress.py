from ase import Atoms
from ase.structure import bulk
from ase.optimize.bfgs import BFGS
from ase.constraints import UnitCellFilter
from gpaw import GPAW
from gpaw import PW
import numpy as np

cell = bulk('Si', 'fcc', a=6.0).get_cell()
# Experimental Lattice constant is a=5.421 A
a = Atoms('Si2', cell=cell, pbc=True,
          scaled_positions=((0,0,0), (0.25,0.25,0.25)))

calc = GPAW(mode=PW(400),
            xc='PBE',
            kpts=(4,4,4),
            txt='stress.txt')
a.set_calculator(calc)

uf = UnitCellFilter(a)
relax = BFGS(uf)
relax.run(fmax=0.05)

a = np.dot(a.get_cell()[0], a.get_cell()[0])**0.5 * 2**0.5
print 'Relaxed lattice parameter: a = %s A' % a
