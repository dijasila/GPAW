from ase import *
from ase.lattice import bulk
from gpaw import *
from gpaw.wavefunctions.pw import PW

cell = bulk('C', 'fcc', a=3.553).get_cell()
a = Atoms('C2', cell=cell, pbc=True,
          scaled_positions=((0,0,0), (0.25,0.25,0.25)))

calc = GPAW(mode=PW(600),
            xc='PBE',
            occupations=FermiDirac(width=0.0011),
            kpts=(12, 12, 12),
            txt='C2_scf.txt')
a.set_calculator(calc)
a.get_potential_energy()

calc.set(kpts={'size': (4, 4, 4), 'gamma': True}, 
         nbands=16,
         symmetry='off',
         fixdensity=True, 
         txt='C2_nscf.txt',
         convergence={'bands': 8})
calc.get_potential_energy()

calc.write('C2.gpw', mode='all')
