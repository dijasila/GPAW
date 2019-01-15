import gpaw.pgs.pointgroups as pointgroups
from gpaw.pgs import SymmetryCalculator
from gpaw.test import equal
import gpaw.mpi
from ase import Atoms
import numpy as np

pglib = pointgroups.list_of_pointgroups

class TestSymmetryCalculator(SymmetryCalculator):
    """
    Calculator for a given numpy N*M*P array.
    """

    def initialize(self, array):
        self.array = array
        self.load_data()

    def get_atoms(self):
        return Atoms()

    def load_data(self):
        return 1

    def get_wf(self, n, k=0, s=0):
        return self.array

    def get_energy(self, band):
        return 0.0

N = 50
c = (N-1) / 2.

def fx(x,y,z):
    # Function for p_x orbital
    x,y,z = x-c, y-c, z-c
    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    return x * np.exp(-0.5 * r)

def fy(x,y,z):
    # Function for p_y orbital
    x,y,z = x-c, y-c, z-c
    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    return y * np.exp(-0.5 * r)


def fz(x,y,z):
    # Function for p_z orbital
    x,y,z = x-c, y-c, z-c
    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    return z * np.exp(-0.5 * r)


px = np.fromfunction(function=fx, shape=[N, N, N])
px /= np.sqrt(np.square(px).sum()) # normalize
py = np.fromfunction(function=fy, shape=[N, N, N])
py /= np.sqrt(np.square(py).sum())
pz = np.fromfunction(function=fz, shape=[N, N, N])
pz /= np.sqrt(np.square(pz).sum())

# Go through each point group class:
for pgname in pglib:

    # Pass the upper class:
    if pgname == 'Pointgroup':
        continue

    pg = pglib[pgname]()
    pg.character_table = np.array(pg.character_table)




    # Check that the number of irreducible representation is equal to the number of symmetry
    # transform classes:
    equal(len(pg.character_table), len(pg.character_table[0]))




    # Checks for groups with real character tables:
    if not hasattr(pg, 'complex'):

        h = float(sum(pg.nof_operations))

        # Check that the sum of squared dimensions of the irreps
        # equals to the number of symmetry elements h
        sumofsqdim = np.square(pg.character_table[:, 0]).sum()
        equal(sumofsqdim, h, 1e-6)

        # Rows:
        for i, row1 in enumerate(pg.character_table):

            # Check normalization:
            norm = (np.square(row1) * pg.nof_operations).sum()
            equal(norm, h, 1e-6)

            for j, row2 in enumerate(pg.character_table):

                if i >= j:
                    continue

                # Check orthogonality:
                norm = np.multiply(np.multiply(row1, row2), pg.nof_operations).sum()
                equal(norm, 0.0, 1e-6)

        # Columns:
        for i, row1 in enumerate(pg.character_table.T):

            # Check normalization:
            norm = np.square(row1).sum()
            equal(norm, h / pg.nof_operations[i], 1e-6)

            for j, row2 in enumerate(pg.character_table.T):

                if i >= j:
                    continue

                # Check orthogonality:
                norm = np.multiply(row1, row2).sum()
                equal(norm, 0.0, 1e-6)




    # Checks for complex groups:
    else:
        h = float(sum(pg.nof_operations))
        reps = pg.symmetries
        # Rows:
        for i, row1 in enumerate(pg.character_table):

            # Check normalization:
            norm = (np.square(row1) * pg.nof_operations).sum()

            # Real rows:
            if reps[i].find('E') < 0:
                correctnorm = h
            else: # complex rows
                correctnorm = h*2
            
            equal(norm, correctnorm, 1e-6)


            for j, row2 in enumerate(pg.character_table):
                    
                if i >= j:
                    continue

                # Compare real with real rows and complex with complex rows:
                if ((reps[i].find('E') >= 0 and reps[j].find('E') >= 0) 
                    or
                    (reps[i].find('E') < 0 and reps[j].find('E') < 0)):

                    # Check orthogonality:
                    norm = np.multiply(np.multiply(row1, row2), 
                                       pg.nof_operations).sum()
                    equal(norm, 0.0, 1e-6)





    # Calculate the symmetry representations of p-orbitals:
    symcalc = TestSymmetryCalculator('none', 
                                     [0], 
                                     str(pg), 
                                     mpi=gpaw.mpi, 
                                     symmetryfile='sym-%s-x.txt'%str(pg))
    symcalc.initialize(px)
    symcalc.calculate()
    symcalc = TestSymmetryCalculator('none',
                                     [0], 
                                     str(pg), 
                                     mpi=gpaw.mpi, 
                                     symmetryfile='sym-%s-y.txt'%str(pg))
    symcalc.initialize(py)
    symcalc.calculate()

    symcalc = TestSymmetryCalculator('none', 
                                     [0], 
                                     str(pg), 
                                     mpi=gpaw.mpi, 
                                     symmetryfile='sym-%s-z.txt'%str(pg))
    symcalc.initialize(pz)
    symcalc.calculate()



    # Check that the representation indices Tx_i, Ty_i, Tz_i are correct
    # for each group:

    # X
    f = open('sym-%s-x.txt' % str(pg), 'r')
    results = []
    for line in f:
        if line.startswith('#'):
            continue
        results.append(line.split()[:-1])
    f.close()

    results = np.array(results).astype(float)
    norm = results[0, 2]
    equal(results[0, 3 + pg.Tx_i], norm, 1.e-2)





    # Y
    f = open('sym-%s-y.txt' % str(pg), 'r')
    results = []
    for line in f:
        if line.startswith('#'):
            continue
        results.append(line.split()[:-1])
    f.close()

    results = np.array(results).astype(float)
    norm = results[0, 2]
    equal(results[0, 3 + pg.Ty_i], norm, 1.e-2)




    # Z
    f = open('sym-%s-z.txt' % str(pg), 'r')
    results = []
    for line in f:
        if line.startswith('#'):
            continue
        results.append(line.split()[:-1])
    f.close()

    results = np.array(results).astype(float)
    norm = results[0, 2]
    equal(results[0, 3 + pg.Tz_i], norm, 1.e-2)


