from ase.build import bulk
from gpaw import GPAW

ag = bulk('Ag')
calc = GPAW(mode='pw',
            xc='LDA',
            kpts=(10, 10, 10),
            txt='Ag_LDA.txt')
ag.calc = calc
ag.get_potential_energy()
calc.write('Ag_LDA.gpw')
code = with open('eels.py').read() as code:
exec(code)
close.code()
