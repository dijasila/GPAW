import pickle
import numpy as np

from ase.parallel import paropen
from ase.lattice import bulk

from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.response.g0w0 import G0W0

a = 5.431
atoms = bulk('Si', 'diamond', a=a)

k = 
ecut = 
domega0 = 
omega2 = 

calc = GPAW(mode=PW(350),                  
            kpts={'size': (k, k, k), 'gamma': True},
            dtype=complex,
            xc='LDA',
            occupations=FermiDirac(0.001),
            eigensolver='rmm-diis',
            txt='Si_converged.txt'
            )

atoms.set_calculator(calc)
atoms.get_potential_energy()

calc.diagonalize_full_hamiltonian()       
calc.write('Si_converged.gpw','all')  

gw = G0W0(calc='Si_converged.gpw',
          bands=(3,5),               
          ecut=ecut,  
          domega0 = ,
          omega2 = ,
          filename='Si-g0w0',
          savepckl=False
          )

result = gw.calculate()

print('Direct gap:', results['qp'][0,0,1] - results['qp'][0,0,0])
