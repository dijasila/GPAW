import os

import numpy as np
import pytest

from gpaw import GPAW, PW, FermiDirac, Mixer
from gpaw.mpi import world, serial_comm 
from gpaw.test import equal, findpeak
from gpaw.response.chi0 import Chi0
from gpaw.response.screened_interaction import WCalculator, initialize_w_calculator
from ase.build import bulk
from ase.units import Bohr, Hartree
from gpaw.wannier import calculate_overlaps
from ase.dft.wannier import Wannier

@pytest.mark.response
def test_W_in_Wann(in_tmp_dir):
    atoms=bulk('Na')
    atoms.calc = GPAW(mode=PW(100),
                   kpts={'size': (2, 2, 2), 'gamma': True},
                   parallel={'band': 1},
                   txt='gs.txt')
    atoms.get_potential_energy()
    atoms.calc.diagonalize_full_hamiltonian(nbands=20)
    atoms.calc.write('gs.gpw',mode='all')

    calc = GPAW('gs.gpw', txt=None, communicator=serial_comm)
    omega = np.array([0, 1.0, 2.0])
    chi0calc = Chi0(calc, frequencies=omega, hilbert=False,ecut=100, txt='test.log',intraband=False)
    wcalc = initialize_w_calculator(chi0calc, world=world)
    wcalc.calc_in_Wannier(chi0calc,Uwan=None,bandrange=[0,2])

def test_w90(in_tmp_dir):
    from ase import Atoms
    from ase.build import bulk
    from gpaw import GPAW, FermiDirac, PW
    import os
    import gpaw.wannier90 as w90
    from gpaw import GPAW

    cell = bulk('Ga', 'fcc', a=5.68).cell
    a = Atoms('GaAs', cell=cell, pbc=True,
              scaled_positions=((0, 0, 0), (0.25, 0.25, 0.25)))

    calc = GPAW(mode=PW(600),
                xc='LDA',
                occupations=FermiDirac(width=0.01),
                convergence={'density': 1.e-6},
                symmetry='off',
                kpts={'size': (2, 2, 2), 'gamma': True},
                txt='gs_GaAs.txt')

    a.calc = calc
    a.get_potential_energy()
    calc.write('GaAs.gpw', mode='all')
    
    seed = 'GaAs'

    calc = GPAW(seed + '.gpw', txt=None)

    w90.write_input(calc, orbitals_ai=[[], [0, 1, 2, 3]],
                    bands=range(4),
                    seed=seed,
                    num_iter=1000,
                    plot=True,
                    write_u_matrices=True)

    w90.write_wavefunctions(calc, seed=seed)
    os.system('wannier90.x -pp ' + seed)

    w90.write_projections(calc, orbitals_ai=[[], [0, 1, 2, 3]], seed=seed)
    w90.write_eigenvalues(calc, seed=seed)
    w90.write_overlaps(calc, seed=seed)

    os.system('wannier90.x ' + seed)

    omega = np.array([0, 1.0, 2.0])
    chi0calc = Chi0(calc, frequencies=omega, hilbert=False,ecut=100, txt='test.log',intraband=False)
    wcalc = initialize_w_calculator(chi0calc, world=world)
    wcalc.calc_in_Wannier(chi0calc,Uwan=seed,bandrange=[0,4])
