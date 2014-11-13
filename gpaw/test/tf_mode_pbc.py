from ase.lattice import bulk
from gpaw import GPAW
from gpaw.mixer import Mixer
from gpaw.test import equal
from gpaw.test import gen
from gpaw.eigensolvers import CG


symbol = 'C'
result = -224.119823656
electrons = 48

xcname = 'LDA_K_TF+LDA_X'
g = gen(symbol, xcname=xcname, scalarrel=False, orbital_free=True)
h = 0.18
a = 2.8
atoms = bulk(symbol, 'diamond', a=a, cubic=True)   # Generate diamond
mixer = Mixer(0.01, 5, 1)
eigensolver = CG(rtol=0.001,niter=8)
        
calc = GPAW(h=h,
            xc=xcname,
            maxiter=120,
            eigensolver=eigensolver,
            mixer=mixer)


atoms.set_calculator(calc)

e = atoms.get_potential_energy()

n = calc.get_all_electron_density()

dv = atoms.get_volume() / calc.get_number_of_grid_points().prod()
I = n.sum() * dv / 2**3

equal(I, electrons, 1.0e-6)
equal(result, e, 1.0e-4)
