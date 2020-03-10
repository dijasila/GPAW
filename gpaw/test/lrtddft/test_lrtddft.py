import time
import pytest
import numpy as np

from ase import Atom, Atoms
from ase.parallel import parprint, world
from ase.units import Ha

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


def test_io():
    calc = GPAW(xc='PBE', h=0.25, nbands=5, txt=None)
    calc.calculate(get_H2(calc))
    exlst = LrTDDFT(calc, restrict={'jend': 3})
    
    fname = 'lr.dat.gz'
    exlst.write(fname)

    lr2 = LrTDDFT(fname)
    assert len(lr2) == len(exlst) == 3

    lr3 = LrTDDFT(fname, restrict={'jend': 2})
    print(lr3)
    assert len(lr3) == 2


if __name__ == '__main__':
    test_io()
