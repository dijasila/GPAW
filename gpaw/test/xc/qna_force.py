from __future__ import print_function
from gpaw import GPAW, PW
from ase import Atoms
from ase.parallel import parprint
from ase.lattice.compounds import L1_2

name = 'Cu3Au'
ecut = 300
kpts = (2,2,2)

QNA = {'alpha': 2.0,
       'name': 'QNA',
       'orbital_dependent': False,
       'parameters': {'Au': (0.125, 0.1), 'Cu': (0.0795, 0.005)},
       'setup_name': 'PBE',
       'type': 'qna-gga'}

atoms = L1_2(['Au','Cu'],latticeconstant=3.74)

dx_array = [0.0,0.01,0.02]
E = []

for i,dx in enumerate(dx_array):
    calc = GPAW(mode=PW(ecut),
                xc = QNA,
                kpts=kpts,
                txt=name + '.txt'
                )

    atoms[0].position[0] += dx
    atoms.set_calculator(calc)
    E.append(atoms.get_potential_energy())
    if i == 1:
        F = atoms.get_forces()[0,0]

F_num = -(E[-1]-E[0])/(2*(dx_array[-1]-dx_array[0]))
F_err = F_num - F

parprint('Analytical force = ',F)
parprint('Numerical  force = ',F_num)
parprint('Difference       = ',F_err)
assert abs(F_err) < 5e-3
