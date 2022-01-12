# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.
"""Main gpaw module."""
import os
import sys
import contextlib
from pathlib import Path
from typing import List, Dict, Union, Any

__version__ = '22.1.0'
__ase_version_required__ = '3.22.1'
__all__ = ['GPAW',
           'Mixer', 'MixerSum', 'MixerDif', 'MixerSum2',
           'CG', 'Davidson', 'RMMDIIS', 'DirectLCAO',
           'PoissonSolver',
           'FermiDirac', 'MethfesselPaxton', 'MarzariVanderbilt',
           'PW', 'LCAO', 'FD',
           'restart']


setup_paths: List[Union[str, Path]] = []
is_gpaw_python = '_gpaw' in sys.builtin_module_names
dry_run = 0
debug: bool = bool(sys.flags.debug)


@contextlib.contextmanager
def disable_dry_run():
    """Context manager for temporarily disabling dry-run mode.

    Useful for skipping exit in the GPAW constructor.
    """
    global dry_run
    size = dry_run
    dry_run = 0
    yield
    dry_run = size


if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'


from gpaw.broadcast_imports import broadcast_imports  # noqa

with broadcast_imports:
    import os
    import runpy
    import warnings
    from argparse import ArgumentParser, REMAINDER, RawDescriptionHelpFormatter

    import numpy as np
    import _gpaw


class ConvergenceError(Exception):
    pass


class KohnShamConvergenceError(ConvergenceError):
    pass


class PoissonConvergenceError(ConvergenceError):
    pass


class KPointError(Exception):
    pass


class BadParallelization(Exception):
    """Error indicating missing parallelization support."""
    pass


libraries: Dict[str, str] = {}
if hasattr(_gpaw, 'lxcXCFunctional'):
    libraries['libxc'] = getattr(_gpaw, 'libxc_version', '2.x.y')
else:
    libraries['libxc'] = ''


def parse_arguments(argv):
    version = sys.version.replace('\n', '')
    p = ArgumentParser(usage='%(prog)s [OPTION ...] [-c | -m] SCRIPT'
                       ' [ARG ...]',
                       description='Run a parallel GPAW calculation.\n\n'
                       f'Compiled with:\n  Python {version}',
                       formatter_class=RawDescriptionHelpFormatter)

    p.add_argument('--command', '-c', action='store_true',
                   help='execute Python string given as SCRIPT')
    p.add_argument('--module', '-m', action='store_true',
                   help='run library module given as SCRIPT')
    p.add_argument('-W', metavar='argument',
                   action='append', default=[], dest='warnings',
                   help='warning control.  See the documentation of -W for '
                   'the Python interpreter')
    p.add_argument('script', metavar='SCRIPT',
                   help='calculation script')
    p.add_argument('options', metavar='ARG',
                   help='arguments forwarded to SCRIPT', nargs=REMAINDER)

    args = p.parse_args(argv[1:])

    if args.command and args.module:
        p.error('-c and -m are mutually exclusive')

    sys.argv = [args.script] + args.options

    for w in args.warnings:
        # Need to convert between python -W syntax to call
        # warnings.filterwarnings():
        warn_args = w.split(':')
        assert len(warn_args) <= 5

        if warn_args[0] == 'all':
            warn_args[0] = 'always'
        if len(warn_args) >= 3:
            # e.g. 'UserWarning' (string) -> UserWarning (class)
            warn_args[2] = globals().get(warn_args[2])
        if len(warn_args) == 5:
            warn_args[4] = int(warn_args[4])

        warnings.filterwarnings(*warn_args, append=True)

    return args


def main():
    gpaw_args = parse_arguments(sys.argv)
    # The normal Python interpreter puts . in sys.path, so we also do that:
    sys.path.insert(0, '.')
    # Stacktraces can be shortened by running script with
    # PyExec_AnyFile and friends.  Might be nicer
    if gpaw_args.command:
        d = {'__name__': '__main__'}
        exec(gpaw_args.script, d, d)
    elif gpaw_args.module:
        # Python has: python [-m MOD] [-c CMD] [SCRIPT]
        # We use a much better way: gpaw-python [-m | -c] SCRIPT
        runpy.run_module(gpaw_args.script, run_name='__main__')
    else:
        runpy.run_path(gpaw_args.script, run_name='__main__')


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


with broadcast_imports:
    from gpaw.calculator import GPAW as OldGPAW
    from gpaw.mixer import Mixer, MixerSum, MixerDif, MixerSum2
    from gpaw.eigensolvers import Davidson, RMMDIIS, CG, DirectLCAO
    from gpaw.poisson import PoissonSolver
    from gpaw.occupations import (FermiDirac, MethfesselPaxton,
                                  MarzariVanderbilt)
    from gpaw.wavefunctions.lcao import LCAO
    from gpaw.wavefunctions.pw import PW
    from gpaw.wavefunctions.fd import FD


def GPAW(*args, **kwargs) -> Any:
    if os.environ.get('GPAW_NEW'):
        from gpaw.new.ase_interface import GPAW as NewGPAW
        return NewGPAW(*args, **kwargs)
    return OldGPAW(*args, **kwargs)


def restart(filename, Class=GPAW, **kwargs):
    calc = Class(filename, **kwargs)
    atoms = calc.get_atoms()
    return atoms, calc


def read_rc_file():
    home = os.environ.get('HOME')
    if home is not None:
        rc = os.path.join(home, '.gpaw', 'rc.py')
        if os.path.isfile(rc):
            # Read file in ~/.gpaw/rc.py
            with open(rc) as fd:
                exec(fd.read())


def initialize_data_paths():
    try:
        setup_paths[:0] = os.environ['GPAW_SETUP_PATH'].split(os.pathsep)
    except KeyError:
        if len(setup_paths) == 0:
            if os.pathsep == ';':
                setup_paths[:] = [r'C:\gpaw-setups']
            else:
                setup_paths[:] = ['/usr/local/share/gpaw-setups',
                                  '/usr/share/gpaw-setups']


read_rc_file()
initialize_data_paths()


def RMM_DIIS(*args, **kwargs):
    warnings.warn('Please use RMMDIIS instead of RMM_DIIS')
    return RMMDIIS(*args, **kwargs)
