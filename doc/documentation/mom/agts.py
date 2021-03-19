from pathlib import Path

from ase.io import read
from myqueue.workflow import run


def workflow():
    with run(script='mom_co.py', cores=8):
        run(function=check_co)
    with run(script='mom_h2o.py', cores=8):
        run(function=check_h2o)


def check_co():
    co = read('co.txt')
    assert abs(co.get_distance(0, 1) - 1.248) < 0.001


def check_h2o():
    text = Path('h2o_energies.txt').read_text()
    for line in text.splitlines():
        if line.startswith('Excitation energy triplet'):
            et = float(line.split()[-2])
        elif line.startswith('Excitation energy singlet'):
            es = float(line.split()[-2])
    assert abs(et - 9.21) < 0.005
    assert abs(es - 9.68) < 0.005
