from ase.io import read

from gpaw import GPAW, FermiDirac, Mixer
from gpaw1.defects.defect import Defect

# Calculator and supercell parameters
PP = 'PAW'       # PAW pseudopotentials
mode = 'FD'      # Other options: 'LCAO' and 'PW'
basis = 'SZ'     # Basis used for the initialization of the wave functions
                 # Use suitable LCAO basis if mode='LCAO' !
h = 0.2          # Grid spacing
k = 3            # k points
kpts = (k, k, 1) # Use only in periodic directions
width = 0.2      # Fermi smearing (eV)
N = 5            # Supercell size: NxN

# Load graphene atoms
atoms = read('graphene.traj')

# Number of valence electrons
Ne = 8

# Setup GPAW calculator
convergence = {'energy': 5e-3/(Ne*N**2),
               'bands': -10}
parallel = {'domain': (1, 1, 2)}

calc = GPAW(parallel=parallel,
            convergence=convergence,
            kpts=kpts,
            maxiter=500,
            mode=mode.lower(),
            basis='%s(dzp)' % basis.lower(),
            symmetry={'point_group': True,
                      'tolerance': 1e-6},
            xc='LDA',
            mixer=Mixer(beta=0.05, nmaxold=5, weight=100.0),
            nbands=-25,
            setups={'default': PP.lower()},
            occupations=FermiDirac(width),
            h=h,
            txt=None,
            verbose=1)


# Vacancy specs
# (c, a): c'th primitive cell (0=center cell) and atom #a in primitive cell
vac = [(0, 0), ]

# Base name used for file writing
fname = 'vacancy_%ux%u' % (N, N)

# Defect calculator
calc_def = Defect(atoms, calc=calc, supercell=(N, N, 1), defect=vac,
                  name=fname, pckldir='.')
calc_def.run()
