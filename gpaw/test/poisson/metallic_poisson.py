from gpaw import GPAW
from gpaw.eigensolvers import *
from ase.build import molecule
from ase.parallel import parprint,rank
import numpy as np
from ase.build import fcc111
from gpaw.poisson import PoissonSolver 
slab = fcc111('Au', (1, 1, 3), a=4.08, vacuum=8)
metallic = 'both'
charge = 0.1  

slab.calc = GPAW(xc='LDA', h=0.2,
                    txt= 'metallic.txt', charge = charge,
                    convergence = {'density': 1e-2, 'energy': 1e-3, 'eigenstates': 1e-3},
                    eigensolver = Davidson(3), kpts=(2, 2, 1),
                    poissonsolver=PoissonSolver(metallic_electrodes=metallic))

E = slab.get_potential_energy()
electrostatic = slab.calc.get_electrostatic_potential().mean(0).mean(0)
phi0 = slab.calc.hamiltonian.vHt_g.mean(0).mean(0)
if rank==0:
   np.save('metallic', electrostatic)
