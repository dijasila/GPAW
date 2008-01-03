"""Test automatically write out of restart files"""

import os
import sys
from gpaw import Calculator
from ase import *
from gpaw.utilities import equal
from gpaw.cluster import Cluster

endings = ['gpw']
try:
    import Scientific.IO.NetCDFXXXX
    endings.append('nc')
except ImportError:
    pass

for ending in endings:
    restart = 'gpaw-restart.' + ending
    restart_wf = 'gpaw-restart-wf.' + ending
    # H2
    H = Cluster([Atom('H', (0,0,0)), Atom('H', (0,0,1))])
    H.minimal_box(2.)

    wfdir = 'wfs_tmp'
    mode = ending+':' + wfdir + '/psit_s%dk%dn%d'

    if 1:
        calc = Calculator(nbands=2, convergence={'eigenstates': 1e-3})
        H.set_calculator(calc)
        H.get_potential_energy()
        calc.write(restart_wf, 'all')
        calc.write(restart, mode)

    # refine the restart file containing the wfs 
    E1 = Calculator(restart_wf,
                    convergence={'eigenstates': 1.e-5}).get_potential_energy()
        
    # refine the restart file and seperate wfs 
    calc = Calculator(restart, convergence={'eigenstates': 1.e-5})
    calc.read_wave_functions(mode)
    E2 = calc.get_potential_energy()

    print E1, E2
    equal(E1, E2, 1e-12)

    os.remove(restart_wf)
    os.remove(restart)
    for f in os.listdir(wfdir):
        os.remove(wfdir + '/' + f)
    os.rmdir(wfdir)
