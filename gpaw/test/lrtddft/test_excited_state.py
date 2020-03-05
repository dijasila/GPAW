import sys
import time

from ase import Atom, Atoms
from ase.parallel import parprint

from gpaw import GPAW
from gpaw.test import equal
from gpaw.lrtddft import LrTDDFT
from gpaw.lrtddft.excited_state import ExcitedState
from gpaw.lrtddft.finite_differences import FiniteDifference


def get_H2():
    R=0.7 # approx. experimental bond length
    a = 3.0
    c = 4.0
    H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
                Atom('H', (a / 2, a / 2, (c + R) / 2))],
               cell=(a, a, c))
    return H2


def test_lrtddft_excited_state():
    txt='-'
    txt='/dev/null'
    txt=None

    H2 = get_H2()
    calc = GPAW(xc='PBE', h=0.25, nbands=3, spinpol=False, txt=txt)
    H2.set_calculator(calc)

    parprint("########### define LrTDDFT")
    xc='LDA'
    lr = LrTDDFT(calc, xc=xc)

    # excited state with forces
    accuracy = 0.015
    parprint("########### define ExcitedState")
    exst = ExcitedState(lr, 0, d=0.01,
                        parallel=2,
                        txt='gpaw_out')

    t0 = time.time()
    parprint("########### first call to forces --> calculate")
    forces = exst.get_forces(H2)
    parprint("time used:", time.time() - t0)
    for c in range(2):
        equal(forces[0,c], 0.0, accuracy)
        equal(forces[1,c], 0.0, accuracy)
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

def test_finite_differences():
    H2 = get_H2()
    calc = GPAW(xc='PBE', h=0.25, nbands=3, spinpol=False, txt='-')
    H2.set_calculator(calc)
    fd = FiniteDifference(H2, H2.get_potential_energy,
                          parallel=2)


if __name__ == '__main__':
    test_finite_differences()
    #test_lrtddft_excited_state()
