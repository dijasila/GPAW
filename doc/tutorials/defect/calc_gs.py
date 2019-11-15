from math import sqrt

from ase.io import write
from ase import Atoms

from gpaw import GPAW, FermiDirac, Mixer


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

# Write to traj file for later use
write('graphene.traj', atoms)

# Setup calculator
PP = 'PAW'
mode = 'LCAO'
basis = 'DZP'
h = 0.11
k = 21
kpts = (k, k, 1)
Ne = 8
convergence = {'energy': 2e-4/Ne}

calc = GPAW(convergence=convergence,
            mode=mode.lower(),
            maxiter=500,
            basis='%s(dzp)' % basis.lower(),
            xc='LDA',
            mixer=Mixer(beta=0.05, nmaxold=5, weight=90.0),
            h=h,
            kpts=kpts,
            nbands='nao', # number of lcao orbitals
            occupations=FermiDirac(0.005),
            symmetry={'point_group': False,
                      'tolerance': 1e-6},
            setups={'default': PP.lower()},
            txt='calc_lcao.txt',
            verbose=True)

atoms.set_calculator(calc)

e = atoms.get_potential_energy()

calc.write('calc_lcao.gpw', mode='all')
