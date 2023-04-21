from ase.io import read
from gpaw import GPAW, FermiDirac, Mixer, PoissonSolver
from gpaw import setup_paths
from gpaw.poisson_moment import MomentCorrectionPoissonSolver

# Insert the path to the created basis set
setup_paths.insert(0, '.')

# Read the structure from the xyz file
atoms = read('Ag55.xyz')
atoms.center(vacuum=6.0)

# Increase the accuracy of density for ground state
convergence = {'density': 1e-12}

# Use occupation smearing and weak mixer to facilitate convergence
occupations = FermiDirac(25e-3)
mixer = Mixer(0.02, 5, 1.0)

# Parallelzation settings
parallel = {'sl_auto': True, 'domain': 2, 'augment_grids': True}

# Apply multipole corrections for monopole and dipoles
poissonsolver = MomentCorrectionPoissonSolver(poissonsolver=PoissonSolver(),
                                              moment_corrections=1 + 3)

# Ground-state calculation
calc = GPAW(mode='lcao', xc='GLLBSC', h=0.3, nbands=360,
            setups={'Ag': 'my'},
            basis={'Ag': 'GLLBSC.dz', 'default': 'dzp'},
            convergence=convergence, poissonsolver=poissonsolver,
            occupations=occupations, mixer=mixer, parallel=parallel,
            maxiter=1000,
            txt='gs.out')
atoms.calc = calc
atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')
