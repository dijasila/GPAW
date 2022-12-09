from gpaw.wannier.w90 import read_wout_all
from myqueue.workflow import run
import numpy as np


def check():
    with open('GaAs.wout') as fd:
        dct = read_wout_all(fd)
    x, y, z = dct['centers'].sum(axis=0)
    w = dct['spreads'].sum()
    a = 5.68
    assert abs(np.array([x, y, z, w]) - [a, a, a, 4.499]).max() < 0.005

    with open('Fe.wout') as fd:
        dct = read_wout_all(fd)
    xyz = dct['centers'].sum(axis=0)
    w = dct['spreads'].sum()
    print(xyz, w)
    assert abs(xyz).max() < 0.005
    assert abs(w - 14.5) < 0.15


def workflow():
    with run(script='GaAs.py'):
        r1 = run(script='GaAs_wannier.py')
    with run(script='Fe.py', cores=8):
        r2 = run(script='Fe_wannier.py', tmax='1h')
    with r1, r2:
        run(function=check)


if __name__ == '__main__':
    check()
