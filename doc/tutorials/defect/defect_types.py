from math import sqrt

from ase import Atoms
from ase.io import write, read

from gpaw1.defects.defect import Defect

# Supercell size
N = 5
# Dummy ``calc`` keyword used here
calc = None


# 1) Multiple vacancies
atoms = read('graphene.traj')
defect = [(0, 0), (0, 1)]

# Base name used for file writing
fname = 'vacancy_AB_%ux%u' % (N, N)

# Defect calculator
calc_def = Defect(atoms, calc=calc, supercell=(N, N, 1), defect=defect,
                  name=fname, pckldir='.')


# 2) Graphene with substitutional atom
atoms = read('graphene.traj')
sub = 'N'
defect = {0: sub}

# Base name used for file writing
fname = 'substitutional_%s_%ux%u' % (sub, N, N)

# Defect calculator
calc_def = Defect(atoms, calc=calc, supercell=(N, N, 1), defect=defect,
                  name=fname, pckldir='.')


# 3) Graphene with Li adatom in hollow site


# Graphene lattice constant
a = 2.46
# Vacuum region
c = 20.0
# Li-graphene distance
ad = 'Li'
dist = 1.75
# Include monolayer of Li-hollow atoms on both sides in order not to break the
# mirror-plane symmetry of graphene
atoms = Atoms('C2%s2' % ad,
              scaled_positions=[(0, 0, 1./2),
                                (1./3, 1./3, 1./2),
                                (2./3, 2./3, 1./2 - dist/c),
                                (2./3, 2./3, 1./2 + dist/c)],
              cell=[(a,         0,     0),
                    (a/2, a*sqrt(3)/2, 0),
                    (0,         0,     c)],
              pbc=[1, 1, 0])

write('graphene_%s.traj' % ad.lower(), atoms)

# The Li-adatom defect is specified by: 1) deactivating the Li atoms in the
# 'pristine' calculation by marking them as "ghosts" atoms, and 2) activate the
# defect adatom in the 'defect' calculation (here #3).
defect = {'ghosts':
          {'pristine': [2, 3,],
           'defect': [2,]} }

# Base name used for file writing
fname = 'adatom_%s_%ux%u' % (ad, N, N)

# Defect calculator
calc_def = Defect(atoms, calc=calc, supercell=(N, N, 1), defect=defect,
                  name=fname, pckldir='.')
