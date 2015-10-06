from ase import *
from ase.lattice import bulk
from gpaw import *
from gpaw.wavefunctions.pw import PW

cell = bulk('Si', 'fcc', a=5.421).get_cell()
a = Atoms('Si2', cell=cell, pbc=True,
          scaled_positions=((0,0,0), (0.25,0.25,0.25)))

calc = GPAW(mode=PW(600),
            xc='PBE',
            occupations=FermiDirac(width=0.0011),
            kpts=(12, 12, 12),
            txt='Si2_scf.txt')
a.set_calculator(calc)
a.get_potential_energy()

calc.set(kpts={'size': (4, 4, 4), 'gamma': True}, 
         nbands=16,
         symmetry='off',
         fixdensity=True, 
         txt='Si2_nscf.txt',
         convergence={'bands': 8})
calc.get_potential_energy()

calc.write('Si2.gpw', mode='all')
