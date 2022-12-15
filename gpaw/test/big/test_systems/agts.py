from myqueue.workflow import run
from gpaw.test.big.test_systems.create import create_test_systems
from gpaw import GPAW


CORES = [1, 4, 8, 16, 14, 40, 48, 56, 72, 96, 120]


def workflow():
    for name, (atoms, params) in create_test_systems().items():
        cores = len(atoms)**2 / 10
        # Find best match:
        _, cores = min((abs(cores - c), c) for c in CORES)
        run(function=calculate, args=[name, atoms, params],
            name=name,
            cores=cores,
            tmax='1d')


def calculate(name, atoms, params):
    atoms.calc = GPAW(**params, txt=name + '.txt')
    atoms.get_potential_energy()
