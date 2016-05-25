import ase.db
from ase.build import bulk
import numpy as np
from gpaw.xc.exx import EXX
from gpaw.xc.tools import vxc
from gpaw import GPAW, PW, FermiDirac

a0 = 5.43
si = bulk('Si', 'diamond', a)

for k in range(2, 9):
    name = 'Si-{0:.2f}-{1}'.format(a, k)
    si.calc = GPAW(kpts={'size': (k, k, k), 'gamma': True},
                   mode=PW(300),
                   xc='PBE',
                   convergence={'bands': 5},
                   txt=name + '.txt')
    si.get_potential_energy()
    si.calc.write(name, mode='all')
        
        eig_pbe_n = si.calc.get_eigenvalues(0)[:5]  # Gamma point
        deig_pbe_n = vxc(si.calc, 'PBE')[0, 0, :5]
        
        pbe0 = EXX(name, 'PBE0', bands=[0, 5], txt=name + '.pbe0.txt')
        pbe0.calculate()
        epbe0 = pbe0.get_total_energy()
        deig_pbe0_n = pbe0.get_eigenvalue_contributions()[0, 0]
        eig_pbe0_n = eig_pbe_n - deig_pbe_n + deig_pbe0_n
        
        gap = eig_pbe_n[4] - eig_pbe_n[3]
        gap0 = eig_pbe0_n[4] - eig_pbe0_n[3]
        
        con.write(si, a=a, k=k, epbe0=epbe0, gap=gap, gap0=gap0)
        del con[id]
