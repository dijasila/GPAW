from ase import Atoms
from ase.parallel import parprint
from .info import info
from gpaw import GPAW, PW, setup_paths
from gpaw.mpi import size


class CLICommand:
    """Test GPAW installation."""

    @staticmethod
    def add_arguments(parser):
        pass

    @staticmethod
    def run(args):
        info()
        test()


def test():
    if not setup_paths:
        raise RuntimeError('Could not find any atomic PAW-data or '
                           'pseudopotentials!')

    parprint(f'Doing a test calculation (cores: {size}): ... ',
             end='', flush=True)
    a = 2.5
    d = 0.9
    chain = Atoms('H', cell=[a, a, d], pbc=(False, False, True))
    chain.calc = GPAW(mode=PW(200),
                      kpts=(1, 1, 8),
                      txt='test.txt')
    chain.get_forces()
    chain.get_stress()
    parprint('Done')
    if size == 1:
        print()
        print('Test parallel calculation with "gpaw -P 4 test".')
