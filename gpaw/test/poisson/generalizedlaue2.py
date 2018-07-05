from ase.build import fcc100
from gpaw.poisson import GeneralizedLauePoissonSolver, FDPoissonSolver
from gpaw import GPAW
from gpaw.grid_descriptor import GridDescriptor
import numpy as np

slab = fcc100('Al', (1, 1, 2), a=4.05, vacuum=3.0)
slab.center(axis=2)

E=[]
for i in range(2):
    if i==0:
        poisson=GeneralizedLauePoissonSolver(nn=3)
    else:
        poisson=FDPoissonSolver(nn=3, eps=1e-20)
    slab.calc = GPAW(mode='lcao', basis='sz(dzp)', xc='LDA',
                     setups={'Na': '1'}, convergence={'density':1e-7},
                     kpts=(1, 1, 1), poissonsolver=poisson)
    E.append(slab.get_potential_energy())

print E
assert np.abs(E[0]-E[1])<1e-5
