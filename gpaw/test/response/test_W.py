import os

import numpy as np
import pytest

from gpaw import GPAW, PW
from gpaw.mpi import world, serial_comm 
from gpaw.test import equal, findpeak
from gpaw.response.chi0 import Chi0
from gpaw.response.screened_interaction import WCalculator, initialize_w_calculator
from ase.build import bulk
from ase.units import Bohr, Hartree


@pytest.mark.response
def test_W_in_Wann(in_tmp_dir):
    atoms=bulk('Na')
    atoms.calc = GPAW(mode=PW(300),
                   kpts={'size': (4, 4, 4), 'gamma': True},
                   parallel={'band': 1},
                   txt='gs.txt')
    atoms.get_potential_energy()
    atoms.calc.diagonalize_full_hamiltonian(nbands=20)
    atoms.calc.write('gs.gpw',mode='all')

    calc = GPAW('gs.gpw', txt=None, communicator=serial_comm)
    omega = np.array([0, 1.0, 2.0])
    chi0calc = Chi0(calc, frequencies=omega, hilbert=False,ecut=100, txt='test.log',intraband=False)
    """
    chi0calc = Chi0('gs.gpw',
                    frequencies={'type': 'nonlinear',
                                 'domega0': 0.03},
                    ecut=10,
                    intraband=False,
                    hilbert=True,
                    nbands=20,
                    txt='chi0.txt',
                    world=world)
    """
    print('test: ',chi0calc.gs.nspins)
    q_c=[0,0,0]
    chi0calc.calculate(q_c,spin='all')

    
    wcalc = initialize_w_calculator(chi0calc, world=world)
    wcalc.calc_in_Wannier(chi0calc,Uwan=None,bandrange=[0,2])
