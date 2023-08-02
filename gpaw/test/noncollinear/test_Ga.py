import pytest

from ase import Atoms

from gpaw.new.ase_interface import GPAW


def test_noncollinear_Ga():

    # Parameters

    atom = Atoms('Ga', positions=[[3, 3, 3]], cell=[6, 6, 6], pbc=True)

    calc = GPAW(mode={'name': 'pw', 'ecut': 400}, xc='LDA', nbands=13,
                kpts={'size': (1, 1, 1), 'gamma': True}, symmetry='off',
                soc=True, magmoms=[[0, 0, 1]], dtype=complex)

    atom.calc = calc
    e = atom.get_potential_energy()

    assert e == pytest.approx(-0.0779024429, abs=1.0e-6)
