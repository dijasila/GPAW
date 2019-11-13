from math import sqrt

from ase import Atoms

from gpaw import GPAW, FermiDirac, Mixer
from gpaw1.defects.defect import Defect

# Calculator and supercell parameters
PP = 'PAW'       # PAW pseudopotentials
mode = 'FD'      # 'LCAO' # 'PW'
basis = 'SZ'     # 'DZP'
h = 0.2          # Grid spacing
k = 3            # k points
kpts = (k, k, 1) # 
width = 0.1      # Fermi smearing (eV)
N = 5            # Supercell size: NxN

# Build graphene (nonperiodic BCs in the z direction)
# Lattice constant and vacuum size
a = 2.46 ; c = 15.0
atoms = Atoms('C2',
              scaled_positions=[(0, 0, 0),
                                (1/3., 1/3., 0.0)],
              cell=[(a,         0,     0),
                    (a/2, a*sqrt(3)/2, 0),
                    (0,         0,     c)],
              pbc=[1, 1, 0])
atoms.center(axis=2)
# Number of valence electrons
Ne = 8

# Setup GPAW calculator
convergence = {'energy': 5e-4/(Ne*N**2),
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
# (c, a): c'th primitive cell (center cell) and atom #a in primitive cell
vac = [(0, 0), ]

fname = 'vacancy1_supercell_%ux%u' % (N, N)
def_calc = Defect(atoms, calc=calc, supercell=(N, N, 1), defect=vac,
                  name=fname, pckldir='.')
def_calc.run()
