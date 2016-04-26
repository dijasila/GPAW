import numpy as np
from ase import *
from gpaw import GPAW, FermiDirac

a = 3.184	    
b = 3.127	    
L = 15.		   
m1,n1 = [3, 3]

cell = np.array([(1., 0., 0), 
                 (-1. / 2, np.sqrt(3.) / 2., 0.),
                 (0, 0, L)])

a_sc = a * np.linalg.norm(m1 * cell[0] + n1 * cell[1])
structure = Atoms(cell=[(a_sc, 0, 0), 
                        (-a_sc / 2, np.sqrt(3.) / 2. * a_sc, 0), 
                        (0, 0, L)],
                  pbc=(1, 1, 1))

for isup in range(0, m1):
    for isup2 in range(0, n1):
        structure.append(Atom('Mo', (0, 0, L / 2.)+isup * a * cell[0] + isup2 * a
                              * cell[1]))
        structure.append(Atom('S', 2. * a / 3. * cell[0] + a / 3. * cell[1] +
                              (0, 0, L / 2. + b / 2.) + isup * a * cell[0] +
                              isup2 * a * cell[1]))   
        if not (isup == 0 and isup2 == 0):
            structure.append(Atom('S', 2. * a / 3. * cell[0] + a / 3. * cell[1]
                                  + (0, 0, L / 2. -b / 2.) + isup * a * cell[0]
                                  + isup2 * a * cell[1]))

structure.center(axis=2)

calc = GPAW(mode='lcao',
            basis='dzp',
            xc='LDA',
            kpts=(4,4,1),			
            occupations=FermiDirac(0.01),
            txt='gs_3x3_defect.txt')

structure.set_calculator(calc)       
structure.get_potential_energy() 
calc.write('gs_3x3_defect.gpw', 'all')

