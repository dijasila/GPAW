import time
import pytest

from ase import Atom, Atoms
from ase.parallel import parprint, world

from gpaw import GPAW
from gpaw.test import equal
from gpaw.lrtddft import LrTDDFT
from gpaw.lrtddft.excited_state import ExcitedState


def get_H2(calculator=None):
    """Define H2 and set calculator if given"""
    R = 0.7  # approx. experimental bond length
    a = 3.0
    c = 4.0
    H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
                Atom('H', (a / 2, a / 2, (c + R) / 2))],
               cell=(a, a, c))
    
    if calculator is not None:
        H2.set_calculator(calculator)

    return H2


def test_lrtddft_excited_state():
    txt = None
    
    calc = GPAW(xc='PBE', h=0.25, nbands=3, spinpol=False, txt=txt)
    H2 = get_H2(calc)

    xc = 'LDA'
    lr = LrTDDFT(calc, xc=xc)

    # excited state with forces
    accuracy = 0.015
    exst = ExcitedState(lr, 0, d=0.01,
                        parallel=2)

    t0 = time.time()
    parprint("########### first call to forces --> calculate")
    forces = exst.get_forces(H2)
    parprint("time used:", time.time() - t0)
    for c in range(2):
        equal(forces[0, c], 0.0, accuracy)
        equal(forces[1, c], 0.0, accuracy)
    equal(forces[0, 2] + forces[1, 2], 0.0, accuracy)

    parprint("########### second call to potential energy --> just return")
    t0 = time.time()
    E = exst.get_potential_energy()
    parprint("E=", E)
    parprint("time used:", time.time() - t0)
    t0 = time.time()
    E = exst.get_potential_energy(H2)
    parprint("E=", E)
    parprint("time used:", time.time() - t0)

    parprint("########### second call to forces --> just return")
    t0 = time.time()
    exst.get_forces()
    parprint("time used:", time.time() - t0)
    t0 = time.time()
    exst.get_forces(H2)
    parprint("time used:", time.time() - t0)

    parprint("###########  moved atoms, call to forces --> calculate")
    p = H2.get_positions()
    p[1, 1] += 0.1
    H2.set_positions(p)

    t0 = time.time()
    exst.get_forces(H2)
    parprint("time used:", time.time() - t0)


def test_forces():
    """Test whether force calculation works"""
    calc = GPAW(xc='PBE', h=0.25, nbands=3, txt=None)
    exlst = LrTDDFT(calc)
    exst = ExcitedState(exlst, 0)
    H2 = get_H2(exst)
    
    parprint('---------------- serial')

    forces = H2.get_forces()
    accuracy = 1.e-3
    # forces in x and y direction should be 0
    assert forces[:, :2] == pytest.approx(0.0, abs=accuracy)
    # forces in z direction should be opposite
    assert -forces[0, 2] == pytest.approx(forces[1, 2], abs=accuracy)
   
    # next call should give back the stored forces
    forces1 = exst.get_forces(H2)
    assert (forces1 == forces).all()

    # test parallel
    if world.size > 1:
        parprint('---------------- parallel', world.size)
        exstp = ExcitedState(exlst, 0, parallel=2)
        forcesp = exstp.get_forces(H2)
        assert forcesp == pytest.approx(forces, abs=0.001)
        
    
if __name__ == '__main__':
    test_forces()
    # test_lrtddft_excited_state()
