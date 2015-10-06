from ase import *
from ase.lattice import bulk
#from ase.dft import monkhorst_pack
from gpaw import *
from gpaw.wavefunctions.pw import PW

a = Atoms('BaTiO3',
          pbc=True,
          cell=(3.9385, 3.9385, 3.9385),
          scaled_positions=[[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5],
                            [0.0, 0.5, 0.5],
                            [0.5, 0.0, 0.5],
                            [0.5, 0.5, 0.0]])

calc = GPAW(mode=PW(600),
            xc='PBE',
            occupations=FermiDirac(width=0.001),
            kpts=(12, 12, 12),
            txt='BaTiO3_scf.txt')
a.set_calculator(calc)
a.get_potential_energy()

calc.set(kpts={'size': (4, 4, 4), 'gamma': True}, 
         symmetry='off',
         fixdensity=True, 
         txt='BaTiO3_nscf.txt')
calc.get_potential_energy()

calc.write('BaTiO3.gpw', mode='all')
