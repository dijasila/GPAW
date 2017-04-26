# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Main gpaw module."""

import os
import sys
from distutils.util import get_platform

from os.path import join, isfile

import numpy as np

assert not np.version.version.startswith('1.6.0')

__version__ = '1.3.0b1'
__ase_version_required__ = '3.13.0'

__all__ = ['GPAW',
           'Mixer', 'MixerSum', 'MixerDif', 'MixerSum2',
           'CG', 'Davidson', 'RMMDIIS', 'DirectLCAO',
           'PoissonSolver',
           'FermiDirac', 'MethfesselPaxton',
           'PW', 'LCAO', 'restart', 'FD']


class ConvergenceError(Exception):
    pass


class KohnShamConvergenceError(ConvergenceError):
    pass


class PoissonConvergenceError(ConvergenceError):
    pass


# Check for special command line arguments:
memory_estimate_depth = 2
parsize_domain = None
parsize_bands = None
augment_grids = False
sl_default = None
sl_diagonalize = None
sl_inverse_cholesky = None
sl_lcao = None
sl_lrtddft = None
buffer_size = None
extra_parameters = {}
profile = False


def parse_extra_parameters(arg):
    from ase.cli.run import str2dict
    return str2dict(arg)


i = 1
while len(sys.argv) > i:
    arg = sys.argv[i]
    if arg.startswith('--memory-estimate-depth'):
        memory_estimate_depth = -1
        if len(arg.split('=')) == 2:
            memory_estimate_depth = int(arg.split('=')[1])
    elif arg.startswith('--domain-decomposition='):
        parsize_domain = [int(n) for n in arg.split('=')[1].split(',')]
        if len(parsize_domain) == 1:
            parsize_domain = parsize_domain[0]
        else:
            assert len(parsize_domain) == 3
    elif arg.startswith('--state-parallelization='):
        parsize_bands = int(arg.split('=')[1])
    elif arg.startswith('--augment-grids='):
        augment_grids = bool(int(arg.split('=')[1]))
    elif arg.startswith('--sl_default='):
        # --sl_default=nprow,npcol,mb,cpus_per_node
        # use 'd' for the default of one or more of the parameters
        # --sl_default=default to use all default values
        sl_args = [n for n in arg.split('=')[1].split(',')]
        if len(sl_args) == 1:
            assert sl_args[0] == 'default'
            sl_default = ['d'] * 3
        else:
            sl_default = []
            assert len(sl_args) == 3
            for sl_args_index in range(len(sl_args)):
                assert sl_args[sl_args_index] is not None
                if sl_args[sl_args_index] is not 'd':
                    assert int(sl_args[sl_args_index]) > 0
                    sl_default.append(int(sl_args[sl_args_index]))
                else:
                    sl_default.append(sl_args[sl_args_index])
    elif arg.startswith('--sl_diagonalize='):
        # --sl_diagonalize=nprow,npcol,mb,cpus_per_node
        # use 'd' for the default of one or more of the parameters
        # --sl_diagonalize=default to use all default values
        sl_args = [n for n in arg.split('=')[1].split(',')]
        if len(sl_args) == 1:
            assert sl_args[0] == 'default'
            sl_diagonalize = ['d'] * 3
        else:
            sl_diagonalize = []
            assert len(sl_args) == 3
            for sl_args_index in range(len(sl_args)):
                assert sl_args[sl_args_index] is not None
                if sl_args[sl_args_index] is not 'd':
                    assert int(sl_args[sl_args_index]) > 0
                    sl_diagonalize.append(int(sl_args[sl_args_index]))
                else:
                    sl_diagonalize.append(sl_args[sl_args_index])
    elif arg.startswith('--sl_inverse_cholesky='):
        # --sl_inverse_cholesky=nprow,npcol,mb,cpus_per_node
        # use 'd' for the default of one or more of the parameters
        # --sl_inverse_cholesky=default to use all default values
        sl_args = [n for n in arg.split('=')[1].split(',')]
        if len(sl_args) == 1:
            assert sl_args[0] == 'default'
            sl_inverse_cholesky = ['d'] * 3
        else:
            sl_inverse_cholesky = []
            assert len(sl_args) == 3
            for sl_args_index in range(len(sl_args)):
                assert sl_args[sl_args_index] is not None
                if sl_args[sl_args_index] is not 'd':
                    assert int(sl_args[sl_args_index]) > 0
                    sl_inverse_cholesky.append(int(sl_args[sl_args_index]))
                else:
                    sl_inverse_cholesky.append(sl_args[sl_args_index])
    elif arg.startswith('--sl_lcao='):
        # --sl_lcao=nprow,npcol,mb,cpus_per_node
        # use 'd' for the default of one or more of the parameters
        # --sl_lcao=default to use all default values
        sl_args = [n for n in arg.split('=')[1].split(',')]
        if len(sl_args) == 1:
            assert sl_args[0] == 'default'
            sl_lcao = ['d'] * 3
        else:
            sl_lcao = []
            assert len(sl_args) == 3
            for sl_args_index in range(len(sl_args)):
                assert sl_args[sl_args_index] is not None
                if sl_args[sl_args_index] is not 'd':
                    assert int(sl_args[sl_args_index]) > 0
                    sl_lcao.append(int(sl_args[sl_args_index]))
                else:
                    sl_lcao.append(sl_args[sl_args_index])
    elif arg.startswith('--sl_lrtddft='):
        # --sl_lcao=nprow,npcol,mb,cpus_per_node
        # use 'd' for the default of one or more of the parameters
        # --sl_lcao=default to use all default values
        sl_args = [n for n in arg.split('=')[1].split(',')]
        if len(sl_args) == 1:
            assert sl_args[0] == 'default'
            sl_lrtddft = ['d'] * 3
        else:
            sl_lrtddft = []
            assert len(sl_args) == 3
            for sl_args_index in range(len(sl_args)):
                assert sl_args[sl_args_index] is not None
                if sl_args[sl_args_index] is not 'd':
                    assert int(sl_args[sl_args_index]) > 0
                    sl_lrtddft.append(int(sl_args[sl_args_index]))
                else:
                    sl_lrtddft.append(sl_args[sl_args_index])
    elif arg.startswith('--buffer_size='):
        # Buffer size for MatrixOperator in MB
        buffer_size = int(arg.split('=')[1])
    elif arg.startswith('--gpaw='):
        extra_parameters = parse_extra_parameters(arg[7:])
    elif arg == '--gpaw':
        extra_parameters = parse_extra_parameters(sys.argv.pop(i + 1))
    elif arg.startswith('--profile='):
        profile = arg.split('=')[1]
    else:
        i += 1
        continue
    # Delete used command line argument:
    del sys.argv[i]


dry_run = extra_parameters.pop('dry_run', 0)
debug = extra_parameters.pop('debug', False)

if debug:
    np.seterr(over='raise', divide='raise', invalid='raise', under='ignore')

    oldempty = np.empty

    def empty(*args, **kwargs):
        a = oldempty(*args, **kwargs)
        try:
            a.fill(np.nan)
        except ValueError:
            a.fill(-1000000)
        return a
    np.empty = empty


build_path = join(__path__[0], '..', 'build')
arch = '%s-%s' % (get_platform(), sys.version[0:3])

# If we are running the code from the source directory, then we will
# want to use the extension from the distutils build directory:
sys.path.insert(0, join(build_path, 'lib.' + arch))


def get_gpaw_python_path():
    paths = os.environ['PATH'].split(os.pathsep)
    paths.insert(0, join(build_path, 'bin.' + arch))
    for path in paths:
        if isfile(join(path, 'gpaw-python')):
            return path
    raise RuntimeError('Could not find gpaw-python!')


setup_paths = []


def initialize_data_paths():
    try:
        setup_paths[:] = os.environ['GPAW_SETUP_PATH'].split(os.pathsep)
    except KeyError:
        if os.pathsep == ';':
            setup_paths[:] = [r'C:\gpaw-setups']
        else:
            setup_paths[:] = ['/usr/local/share/gpaw-setups',
                              '/usr/share/gpaw-setups']


initialize_data_paths()


from gpaw.calculator import GPAW
from gpaw.mixer import Mixer, MixerSum, MixerDif, MixerSum2
from gpaw.eigensolvers import Davidson, RMMDIIS, CG, DirectLCAO
from gpaw.poisson import PoissonSolver
from gpaw.occupations import FermiDirac, MethfesselPaxton
from gpaw.wavefunctions.lcao import LCAO
from gpaw.wavefunctions.pw import PW
from gpaw.wavefunctions.fd import FD

RMM_DIIS = RMMDIIS


def restart(filename, Class=GPAW, **kwargs):
    calc = Class(filename, **kwargs)
    atoms = calc.get_atoms()
    return atoms, calc


if profile:
    from cProfile import Profile
    import atexit
    prof = Profile()

    def f(prof, filename):
        prof.disable()
        from gpaw.mpi import rank
        if filename == '-':
            prof.print_stats('time')
        else:
            prof.dump_stats(filename + '.%04d' % rank)
    atexit.register(f, prof, profile)
    prof.enable()


command = os.environ.get('GPAWSTARTUP')
if command is not None:
    exec(command)


def is_parallel_environment():
    """Check if we are running in a parallel environment.

    This function can be redefined in ~/.gpaw/rc.py.  Example::

        def is_parallel_environment():
            import os
            return 'PBS_NODEFILE' in os.environ
    """
    return False


def read_rc_file():
    home = os.environ.get('HOME')
    if home is not None:
        rc = os.path.join(home, '.gpaw', 'rc.py')
        if os.path.isfile(rc):
            # Read file in ~/.gpaw/rc.py
            with open(rc) as fd:
                exec(fd.read())

read_rc_file()
