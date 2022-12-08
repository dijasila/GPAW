from myqueue.workflow import run
from gpaw.test.big.test_systems.create import create_test_systems
from gpaw import GPAW, PW
from ase.optimize import BFGS


N = [1, 4, 8, 16, 14, 40, 48, 56, 72, 96, 120]


def workflow():
    for name, (atoms, params) in create_test_systems().items():
        n = len(atoms)**2 / 10
        _, cores = min((abs(n - c), c) for c in N)
        run(function=relax, args=[name, atoms],
            name=name,
            cores=cores, tmax='1d')


def relax(name, atoms):
    atoms.calc = GPAW(mode=PW(500),
                      xc='PBE',
                      kpts=dict(density=3.0),
                      txt=name + '.txt')
    BFGS(atoms,
         logfile=name + '.log',
         trajectory=name + '.traj').run(fmax=0.02)
