from ase.build import bulk
import numpy as np
from gpaw.response.g0w0 import G0W0
from gpaw import GPAW, PW
from ase.units import Hartree as Ha
if 1:
    atoms = bulk('Si')
    calc = GPAW(mode=PW(400), 
                xc='LDA', occupations=dict(width=0),
                kpts={'size': (2,2,2),'gamma':True},
                nbands=50, 
                convergence={'bands':30})
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('wfs.gpw', mode='all')

import os
os.mkdir('ff')
os.chdir('ff')
gw = G0W0('../wfs.gpw', 'ff', 
          bands=(3, 5),
          kpts=[ (0,0,0) ],
          nbands=30,
          evaluate_sigma=np.linspace(-2, 2, 250),
          nblocks=1,
          ecut=50)

results = gw.calculate()
os.chdir('..')

os.mkdir('mpa')
os.chdir('mpa')

mpa_dict = {'npoles':8, 'wrange':[1j*Ha,(1.5+1j)*Ha], 'wshift':[0.01*Ha, 0.1*Ha], 'alpha':1 }
gw = G0W0('../wfs.gpw', 'mpa',
          bands=(3, 5),
          kpts=[ (0,0,0) ],
          nbands=30,
          nblocks=1,
          evaluate_sigma=np.linspace(-2, 2, 250),
          ecut=50,  
          mpa=mpa_dict)
results = gw.calculate()
os.chdir('..')
