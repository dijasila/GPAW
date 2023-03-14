import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, FermiDirac
from ase.io import read

atom = read('structure.json')
kpts_list = [50]
for k in kpts_list:

    calc = GPAW(mode=PW(800),
                xc='PBE',
                occupations=FermiDirac(0.05),
                kpts={'size': (k, k, 1), 'gamma': True},
                convergence={'bands': 'CBM+3.0'},
                nbands = '200%',
                spinpol=True,
                txt=f'gs_BrCuS_kpts{k}x{k}.txt')

    atom.set_calculator(calc)
    atom.get_potential_energy()
    calc.write(f'gs_BrCuS_kpts{k}x{k}.gpw', mode='all')


