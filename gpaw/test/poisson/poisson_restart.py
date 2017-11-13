import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.poisson import PoissonSolver
from gpaw.poisson_extravacuum import ExtraVacuumPoissonSolver

name = 'Na2'
poissoneps = 1e-16
gpts = np.array([16, 16, 24])


def PS():
    return PoissonSolver(eps=poissoneps)

poissonsolver_i = []

ps = PS()
poissonsolver_i.append(ps)

ps = ExtraVacuumPoissonSolver(gpts * 2, PS())
poissonsolver_i.append(ps)

ps = ExtraVacuumPoissonSolver(gpts * 2, PS(), PS(), 2)
poissonsolver_i.append(ps)

ps1 = ExtraVacuumPoissonSolver(gpts, PS(), PS(), 1)
ps = ExtraVacuumPoissonSolver(gpts, ps1, PS(), 1)
poissonsolver_i.append(ps)

for poissonsolver in poissonsolver_i:
    atoms = molecule(name)
    atoms.center(vacuum=3.0)

    # Standard ground state calculation
    # Use loose convergence criterion for speed
    calc = GPAW(nbands=2, gpts=gpts / 2, setups={'Na': '1'}, txt=None,
                poissonsolver=poissonsolver,
                convergence={'energy': 0.1,
                             'density': 0.1,
                             'eigenstates': 0.1})
    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    ps = calc.hamiltonian.poisson
    calc.write('%s_gs.gpw' % name, mode='all')

    # Restart ground state
    calc = GPAW('%s_gs.gpw' % name, txt=None)
    ps1 = calc.hamiltonian.poisson
    assert ps.get_description() == ps1.get_description()
