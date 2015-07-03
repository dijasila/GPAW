from ase import *
from ase.lattice import bulk
#from ase.dft import monkhorst_pack
from gpaw import *
from gpaw.wavefunctions.pw import PW

a = bulk('Fe', 'fcc', a=2.87)
a.set_initial_magnetic_moments([0.7])

calc = GPAW(mode=PW(600),
            xc='PBE',
            occupations=FermiDirac(width=0.1),
            kpts=(12, 12, 12),
            txt='Fe_scf.txt')
a.set_calculator(calc)
a.get_potential_energy()

calc.set(kpts={'size': (8, 8, 8), 'gamma': True},
         symmetry='off',
         nbands=14,
         fixdensity=True,
         txt='Fe_nscf.txt')
calc.get_potential_energy()
calc.write('Fe.gpw', mode='all')
