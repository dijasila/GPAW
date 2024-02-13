from pathlib import Path

from ase.io import read
from myqueue.workflow import run


def workflow():
    with run(script='domom_co.py', cores=8):
        run(function=check_co)
    with run(script='mom_h2o.py', cores=8):
        run(function=check_h2o)
    with run(script='constraints.py', cores=8):
        run(function=check_constraints)


def check_co():
    for tag in ['spinpol', 'spinpaired']:
        co = read('co_' + tag + '.txt')
        assert abs(co.get_distance(0, 1) - 1.248) < 0.01


def check_h2o():
    text = Path('h2o_energies.txt').read_text()
    for line in text.splitlines():
        if line.startswith('Excitation energy triplet'):
            et = float(line.split()[-2])
        elif line.startswith('Excitation energy singlet'):
            es = float(line.split()[-2])
    assert abs(et - 9.21) < 0.005
    assert abs(es - 9.68) < 0.005


def check_constraints():
    text = Path('N-Phenylpyrrole_EX_direct.txt').read_text()
    for line in text.splitlines():
        if line.startswith('Dipole moment:'):
            direct = float(line.split()[-2].replace(')', ''))
    text = Path('N-Phenylpyrrole_EX_from_constrained.txt').read_text()
    for line in text.splitlines():
        if line.startswith('Dipole moment:'):
            constrained = float(line.split()[-2].replace(')', ''))
    assert abs(direct * 4.803 + 3.396) < 0.01
    assert abs(constrained * 4.803 + 10.227) < 0.01
