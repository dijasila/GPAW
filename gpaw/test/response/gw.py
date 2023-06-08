from ase.build import bulk
import numpy as np
from gpaw.response.g0w0 import G0W0
from gpaw import GPAW, PW
from ase.units import Hartree as Ha
atoms = bulk('Si')
calc = GPAW(mode=PW(400), xc='LDA', kpts={'size': (4,4,4),'gamma':True}, nbands=100, convergence={'bands':50})
atoms.calc = calc
atoms.get_potential_energy()
import os
calc.write('wfs.gpw', mode='all')

os.mkdir('ff')
os.chdir('ff')
gw = G0W0('../wfs.gpw', 
          bands=(3, 5),
          nbands=50,
          nblocks=1,
          ecut=40)

results = gw.calculate()
os.chdir('..')
os.mkdir('mpa')
os.chdir('mpa')

mpa_dict = {'npoles':8, 'wrange':[1j*Ha,(1.5+1j)*Ha], 'wshift':[0.01*Ha, 0.1*Ha], 'alpha':1 }
gw = G0W0('../wfs.gpw', 
          bands=(3, 5),
          nbands=50,
          nblocks=1,
          ecut=40,  
          mpa=mpa_dict)
results = gw.calculate()

