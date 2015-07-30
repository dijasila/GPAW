from ase import *
from ase.lattice import bulk
#from ase.dft import monkhorst_pack
from gpaw import *
from gpaw.wavefunctions.pw import PW

a = bulk('Cu', 'fcc')

calc = GPAW(#h=0.16,
            mode=PW(600),
            xc='PBE',
            occupations=FermiDirac(width=0.1),
            kpts=(12, 12, 12),
            txt='Cu_scf.txt')
a.set_calculator(calc)
a.get_potential_energy()

calc.set(kpts={'size': (4, 4, 4), 'gamma': True}, 
         nbands=12,
         symmetry='off',
         fixdensity=True, 
         txt='Cu_nscf.txt',
         convergence={'bands': 9})
calc.get_potential_energy()

calc.write('Cu.gpw', mode='all')
