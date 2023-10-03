import ase.db
from ase.build import bulk
import numpy as np
from gpaw.hybrids.energy import non_self_consistent_energy as nsc_energy
from gpaw import GPAW, PW

a0 = 5.43

con = ase.db.connect('si.db')

for k in range(2, 9):
    for a in np.linspace(a0 - 0.04, a0 + 0.04, 5):
        id = con.reserve(a=a, k=k)
        if id is None:
            continue
        si = bulk('Si', 'diamond', a)
        si.calc = GPAW(kpts=(k, k, k),
                       mode=PW(400),
                       xc='PBE',
                       eigensolver='rmm-diis',
                       txt=None)
        si.get_potential_energy()
        name = f'si-{a:.2f}-{k}'
        si.calc.write(name + '.gpw', mode='all')
        epbe0 = nsc_energy(name + '.gpw', 'PBE0').sum()

        con.write(si, a=a, k=k, epbe0=epbe0)
        del con[id]
