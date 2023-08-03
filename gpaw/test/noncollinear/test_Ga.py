import numpy as np
import pytest
from ase import Atoms

from gpaw.new.ase_interface import GPAW
from gpaw.new.pot_calc import calculate_non_local_potential1
from gpaw.new.xc import create_functional
from gpaw.setup import create_setup
from gpaw.core import UniformGrid
from gpaw.xc import XC


def test_noncollinear_Ga():

    # Parameters

    atom = Atoms('Ga', positions=[[3, 3, 3]], cell=[6, 6, 6], pbc=True)

    calc = GPAW(mode={'name': 'pw', 'ecut': 400}, xc='LDA', nbands=13,
                kpts={'size': (1, 1, 1), 'gamma': True}, symmetry='off',
                soc=True, magmoms=[[0, 0, 1]], dtype=complex)

    atom.calc = calc
    e = atom.get_potential_energy()

    assert e == pytest.approx(-0.0779024429, abs=1.0e-6)


def test_noncollinear_Ga_2():
    setup = create_setup('Ga')
    grid = UniformGrid(cell=[1, 1, 1], size=[9, 9, 9])
    xc = create_functional(XC('LDA', collinear=False), grid, 1, 2, 3, 4, 5)
    D_sii = np.zeros((4, setup.ni, setup.ni), complex)
    D_sii[0, 0, 0] = 2
    P_sm = np.array([[0.2 + 0.3j, 0.1 + 0.2j, 0.3 + 0.4j],
                     [0.4 - 0.5j, 0.2 + 0.3j, 0.6 - 0.7j]])
    for U_mm in [1.0, [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]]:
        P_sm = P_sm @ U_mm
        D_ssmm = np.einsum('si, zj -> szij', P_sm.conj(), P_sm)
        soc = not False
        D_sii[:, 1:4, 1:4] = [D_ssmm[0, 0] + D_ssmm[1, 1],
                              D_ssmm[0, 1] + D_ssmm[1, 0],
                              -1j * (D_ssmm[0, 1] - D_ssmm[1, 0]),
                              D_ssmm[0, 0] - D_ssmm[1, 1]]
        # D_sii.imag[:] = 0.0
        dH_sii, energies = calculate_non_local_potential1(
            setup, xc, D_sii, np.zeros(1), soc)
        print(energies)
