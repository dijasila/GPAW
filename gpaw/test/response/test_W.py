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
    atoms.calc = GPAW(mode=PW(300),
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

def test_W_in_Wann2(in_tmp_dir):
    def wan(calc):
        centers = [([0.125, 0.125, 0.125], 0, 1.5),
                   ([0.125, 0.625, 0.125], 0, 1.5),
                   ([0.125, 0.125, 0.625], 0, 1.5),
                   ([0.625, 0.125, 0.125], 0, 1.5)]
        w = Wannier(4, calc,
                    nbands=4,
                    # log=print,
                    initialwannier=centers)
        w.localize()
        x = w.get_functional_value()
        centers = (w.get_centers(1) * k) % 1
        c = (centers - 0.125) * 2
        print(w.get_radii())  # broken! XXX
        assert abs(c.round() - c).max() < 0.03
        c = sorted(c.round().astype(int).tolist())
        assert c == [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
        if 0:
            from ase.visualize import view
            from ase import Atoms
            watoms = calc.atoms + Atoms(symbols='X4',
                                        scaled_positions=centers,
                                        cell=calc.atoms.cell)
            view(watoms)
        return x
    k = 4
    si = bulk('Si', 'diamond', a=5.43)

    if 0:
        si.calc = GPAW(kpts={'size': (k, k, k), 'gamma': True},
                       parallel=dict(augment_grids=True),
                       mixer=Mixer(0.8, 7, 50.0),
                       txt='Si-ibz.txt')
        e1 = si.get_potential_energy()
        si.calc.write('Si.gpw', mode='all')
    else:
        si.calc = GPAW(mode=PW(50),kpts={'size': (k, k, k), 'gamma': True},
                       parallel=dict(augment_grids=True),
                       mixer=Mixer(0.8, 7, 50.0),
                       txt='Si-ibz.txt')
        e1 = si.get_potential_energy()
        si.calc.write('Si.gpw', mode='all')

    # WF:s
    calc2 = GPAW('Si.gpw', txt=None, communicator=serial_comm)
    calc2.wfs.ibz2bz(si)
    x2 = wan(calc2)

    calc = GPAW('Si.gpw', txt=None, communicator=serial_comm)
    # W
    omega = np.array([0, 1.0, 2.0])
    chi0calc = Chi0(calc, frequencies=omega, hilbert=False,ecut=50, txt='test.log', nbands=8)
    wcalc = initialize_w_calculator(chi0calc, world=world)
    wcalc.calc_in_Wannier(chi0calc,Uwan=None,bandrange=[0,2])
