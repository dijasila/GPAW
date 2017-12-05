from __future__ import print_function
from gpaw import GPAW, PW, restart
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

calc = GPAW(mode=PW(ecut),
            xc = QNA,
            kpts=kpts,
            txt='gs.txt')

atoms.set_calculator(calc)
atoms.get_potential_energy()
eigs = calc.wfs.kpt_u[0].eps_n[:24]
calc.write('gs.gpw')

print(calc.wfs.kpt_u[0])
print(calc.wfs.kpt_u[0])
atoms, calc = restart('gs.gpw', fixdensity=True, kpts=[ [-0.25, 0.25, -0.25] ])
atoms.get_potential_energy()
eigs_new = calc.wfs.kpt_u[0].eps_n[:24]
for eold, enew in zip(eigs, eigs_new):
    print("%.7f %.7f %.7f" % (eold, enew, eold-enew))
    assert(abs(eold-enew) < 1e-6)
