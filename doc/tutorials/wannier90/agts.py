from typing import List
from myqueue.task import task
import numpy as np


def create_tasks():
    t1 = task('GaAs.py')
    t2 = task('GaAs_wannier.py', deps=t1)
    t3 = task('Fe.py', cores=8)
    t4 = task('Fe_wannier.py', deps=t3)
    t5 = task('agts.py', deps=[t2, t4])
    return [t1, t2, t3, t4, t5]


def read(filename: str, nwan: int) -> List[float]:
    with open(filename) as fd:
        lines = [line.strip() for line in fd.readlines()]

    for i, line in enumerate(lines):
        if line == 'Final State':
            break
    line = lines[i + nwan + 1]
    assert line.startswith('Sum of centres and spreads')
    line = line.split('(')[1].replace(',', '').replace(')', '')
    x, y, z, w = (float(x) for x in line.split())
    return x, y, z, w


if __name__ == '__main__':
    x, y, z, w = read('GaAs.wout', 4)
    a = 5.68
    assert abs(np.array([x, y, z, w]) - [a, a, a, 4.14]).max() < 0.01

    x, y, z, w = read('Fe.wout', 18)
    assert abs(np.array([x, y, z, w]) - [0, 0, 0, 14.49]).max() < 0.01
