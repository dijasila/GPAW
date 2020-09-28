from ase.io.wannier90 import read_wout_all
from myqueue.task import task
import numpy as np


def create_tasks():
    t1 = task('GaAs.py')
    t2 = task('GaAs_wannier.py', deps=t1)
    t3 = task('Fe.py', cores=8)
    t4 = task('Fe_wannier.py', deps=t3)
    t5 = task('agts.py', deps=[t2, t4])
    return [t1, t2, t3, t4, t5]


if __name__ == '__main__':
    dct = read_wout_all('GaAs.wout')
    x, y, z = dct['centers'].sum(axis=0)
    w = dct['spreads'].sum()
    a = 5.68
    assert abs(np.array([x, y, z, w]) - [a, a, a, 4.14]).max() < 0.01

    dct = read_wout_all('Fe.wout')
    x, y, z = dct['centers'].sum(axis=0)
    w = dct['spreads'].sum()
    assert abs(np.array([x, y, z, w]) - [0, 0, 0, 14.49]).max() < 0.01
