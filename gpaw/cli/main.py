"""GPAW command-line tool."""
from __future__ import print_function
import os
import sys


from ase.cli.main import main as ase_main

from gpaw import __version__


commands = [
    ('run', 'gpaw.cli.run'),
    ('dos', 'gpaw.cli.dos'),
    ('rpa', 'gpaw.xc.rpa'),
    ('gpw', 'gpaw.cli.gpw'),
    ('info', 'gpaw.cli.info'),
    ('test', 'gpaw.test.test'),
    ('atom', 'gpaw.atom.aeatom'),
    ('diag', 'gpaw.fulldiag'),
    ('quick', 'gpaw.cli.quick'),
    ('python', 'gpaw.cli.py'),
    ('sbatch', 'gpaw.cli.sbatch'),
    ('dataset', 'gpaw.atom.generator2'),
    ('symmetry', 'gpaw.symmetry'),
    ('install-data', 'gpaw.cli.install_data')]


def hook(parser):
    parser.add_argument('-P', '--parallel', type=int, metavar='N', default=1,
                        help="Run on N CPUs.")
    args = parser.parse_args()
    print(args)
    if args.parallel > 1:
        from gpaw.mpi import size
        if size == 1:
            # Start again using gpaw-python in parallel:
            arguments = ['mpiexec', '-np', str(args.parallel),
                         'gpaw-python']
            if args.command == 'python':
                arguments += args.arguments
            else:
                arguments += sys.argv
            os.execvp('mpiexec', arguments)
    return args


def main():
    ase_main('gpaw', 'GPAW command-line tool',
             __version__, commands[4:5], hook)
