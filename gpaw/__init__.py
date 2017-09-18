# encoding: utf-8
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Main gpaw module."""

import os
import sys
import warnings
from distutils.util import get_platform

from os.path import join, isfile

import numpy as np

assert not np.version.version.startswith('1.6.0')

__version__ = '1.3.0b1'
__ase_version_required__ = '3.14.1'

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


def parse_extra_parameters(arg):
    from ase.cli.run import str2dict
    return {key.replace('-', '_'): value
            for key, value in str2dict(arg).items()}


is_gpaw_python = '_gpaw' in sys.builtin_module_names


def parse_arguments():
    from argparse import ArgumentParser, REMAINDER

    p = ArgumentParser(usage='%(prog)s [OPTION ...] SCRIPT [SCRIPTOPTION ...]',
                       description='Run a parallel GPAW calculation.')
    p.add_argument('--module', '-m', action='store_true',
                   help='run library module given as SCRIPT')
    p.add_argument('--memory-estimate-depth', default=2, type=int, metavar='N',
                   dest='memory_estimate_depth',
                   help='print memory estimate of object tree to N levels')
    p.add_argument('--domain-decomposition',
                   metavar='N or X,Y,Z', dest='parsize_domain',
                   help='use N or X × Y × Z cores for domain decomposition.')
    p.add_argument('--state-parallelization', metavar='N', type=int,
                   dest='parsize_bands',
                   help='use N cores for state/band/orbital parallelization')
    p.add_argument('--augment-grids', action='store_true',
                   dest='augment_grids',
                   help='when possible, redistribute real-space arrays on '
                   'cores otherwise used for k-point/band parallelization')
    p.add_argument('--buffer-size', type=float, metavar='SIZE',
                   help='buffer size for MatrixOperator in MiB')
    p.add_argument('--profile', metavar='FILE', dest='profile',
                   help='run profiler and save stats to FILE')
    p.add_argument('--gpaw', metavar='VAR=VALUE', action='append', default=[],
                   dest='gpaw_extra_kwargs',
                   help='extra (hacky) GPAW keyword arguments')
    p.add_argument('script', metavar='SCRIPT',
                   help='calculation')
    p.add_argument('options', metavar='...',
                   help='options forwarded to SCRIPT', nargs=REMAINDER)

    if is_gpaw_python:
        argv = sys.argv[1:]
    else:
        argv = sys.argv[:1]  # Effectively disable command line args

    args = p.parse_args(argv)
    extra_parameters = {}

    if is_gpaw_python:
        sys.argv = [args.script] + args.options

    if args.parsize_domain:
        parsize = [int(n) for n in args.parsize_domain.split(',')]
        if len(parsize) == 1:
            parsize = parsize[0]
        else:
            assert len(parsize) == 3
        args.parsize_domain = parsize

    for extra_kwarg in args.gpaw_extra_kwargs:
        keyword, value = extra_kwarg.split('=')
        parse_extra_parameters(extra_kwarg)
        extra_parameters[keyword] = value

    return extra_parameters, args


extra_parameters, gpaw_args = parse_arguments()

# Check for special command line arguments:
memory_estimate_depth = gpaw_args.memory_estimate_depth
parsize_domain = gpaw_args.parsize_domain
parsize_bands = gpaw_args.parsize_bands
augment_grids = gpaw_args.augment_grids
# We deprecate the sl_xxx parameters being set from command line.
# People can satisfy their lusts by setting gpaw.sl_default = something
# if they are perverted enough to use global variables.
sl_default = None
sl_diagonalize = None
sl_inverse_cholesky = None
sl_lcao = None
sl_lrtddft = None
buffer_size = gpaw_args.buffer_size
profile = gpaw_args.profile


def main():
    import runpy
    # Stacktraces can be shortened by running script with
    # PyExec_AnyFile and friends.  Might be nicer
    if gpaw_args.module:
        # Consider gpaw-python [-m MODULE] [SCRIPT]
        # vs       gpaw-python [-m] [MODULE_OR_SCRIPT] (current implementation)
        runpy.run_module(gpaw_args.script, run_name='__main__')
    else:
        runpy.run_path(gpaw_args.script, run_name='__main__')


def old_parse_args():
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

# Check for typos:
for p in extra_parameters:
    # We should get rid of most of these!
    if p not in {'sic', 'log2ng', 'PK', 'vdw0', 'df_dry_run', 'usenewlfc'}:
        warnings.warn('Unknown parameter: ' + p)

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
