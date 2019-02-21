from gpaw import GPAW
from ase.build import bcc111
from gpaw.poisson import PoissonSolver

slab = bcc111('Na', (1, 1, 2), vacuum=8)
metallic = ['both', 'single']
charge = 0.05

for metal in metallic:
    slab.calc = GPAW(xc='LDA', h=0.22,
                        txt= 'metallic.txt', charge = charge,
                        convergence = {'density': 1e-1, 'energy': 1e-1, 'eigenstates': 1e-1},
                        kpts=(2, 2, 1),
                        poissonsolver=PoissonSolver(metallic_electrodes=metal))

    E = slab.get_potential_energy()
    electrostatic = slab.calc.get_electrostatic_potential().mean(0).mean(0)
    phi0 = slab.calc.hamiltonian.vHt_g.mean(0).mean(0)
    if metal == 'single':
        assert abs(phi0[0])<0.001
    else:
        assert abs(phi0[0])<0.001
        assert abs(phi0[-1])<0.001
