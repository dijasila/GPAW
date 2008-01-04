from ase import *
#from gpaw.utilities.vector import Vector3d
from gpaw.utilities import equal

R = 2.0
CO = Atoms([Atom('C', (1, 0, 0)), Atom('O', (1, 0, R))])

CO.rotate('y', pi/2)
equal(CO.positions[1, 0], R, 1e-10)

# translate
CO.translate(-CO.get_center_of_mass())
p = CO.positions.copy()
for i in range(2):
    equal(p[i, 1], 0, 1e-10)
    equal(p[i, 2], 0, 1e-10)

# rotate the nuclear axis to the direction (1,1,1)
CO.rotate(p[1] - p[0], (1, 1, 1))
q = CO.positions.copy()
for c in range(3):
    equal(q[0, c], p[0, 0] / sqrt(3), 1e-10)
    equal(q[1, c], p[1, 0] / sqrt(3), 1e-10)
