from __future__ import print_function
import subprocess
import sys


class CLICommand:
    short_description = 'Submit a GPAW Python script via sbatch.'

    @staticmethod
    def add_arguments(parser):
        parser.usage = (
            'Usage: '
            'gpaw sbatch [-0] -- [sbatch options] script.py [script options]')
        parser.add_argument('-0', '--dry-run', action='store_true')
        parser.add_argument('arguments', nargs='*')

    @staticmethod
    def run(args):
        script = '#!/bin/bash -l\n'
        for i, arg in enumerate(args.arguments):
            if arg.endswith('.py'):
                break
        else:
            print('No script.py found!', file=sys.stderr)
            return

        for line in open(arg):
            if line.startswith('#SBATCH'):
                script += line
        script += ('OMP_NUM_THREADS=1 '
                   'mpiexec `echo $GPAW_MPI_OPTIONS` gpaw-python {} {}\n'
                   .format(arg, ' '.join(args.arguments[i + 1:])))
        cmd = ['sbatch'] + args.arguments[:i]
        if args.dry_run:
            print('sbatch command:')
            print(' '.join(cmd))
            print('\nscript:')
            print(script, end='')
        else:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            p.communicate(script.encode())
