import ase.db
from ase.build import bulk
import numpy as np
from gpaw.xc.exx import EXX
from gpaw.xc.tools import vxc
from gpaw import GPAW, PW, FermiDirac

a0 = 5.43
si = bulk('Si', 'diamond', a)

for k in range(2, 9, 2):
    name = 'Si-{0}'.format(k)
    si.calc = GPAW(kpts={'size': (k, k, k), 'gamma': True},
                   mode=PW(300),
                   xc='PBE',
                   convergence={'bands': 5},
                   txt=name + '.txt')
    si.get_potential_energy()
    si.calc.write(name, mode='all')
    
    ibzkpts = si.calc.get_ibz_k_points()
    n1 = 3
    n2 = 5
    kpt_indices = []
    for kpt in [(0, 0, 0), (0.5, 0.5, 0)]:
        # Find k-point index:
        i = abs(ibzkpts - kpt).sum(1).argmin()
        kpt_indices.append(i)
        
        eps_kn.append(atoms.calc.get_eigenvalues(k)[n - 1:n + 1])
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
