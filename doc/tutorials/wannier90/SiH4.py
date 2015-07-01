from ase import Atoms
from gpaw import GPAW, PW

cell = [[6.350127, 0.0, 0.0],
        [0.0, 6.350127, 0.0],
        [0.0, 0.0, 6.350127]]
a = Atoms('SiH4', cell=cell, pbc=True,
          scaled_positions=((0.0, 0.0, 0.0), 
                            (0.16638, 0.16638, 0.16638),
                            (0.16638, -0.16638, -0.16638),
                            (-0.16638, 0.16638, -0.16638),
                            (-0.16638, -0.16638, 0.16638)))

calc = GPAW(mode=PW(600),
            xc='PBE',
            nbands=16,
            parallel={'domain': 1, 'band': 1},
            convergence={'bands': 8},
            txt='SiH4_scf.txt')
a.set_calculator(calc)
a.get_potential_energy()

calc.write('SiH4.gpw', mode='all')
