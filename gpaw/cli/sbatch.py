from __future__ import print_function
import subprocess


class CLICommand:
    short_description = 'Submit a GPAW Python script via sbatch.'

    @staticmethod
    def add_arguments(parser):
        parser.usage = (
            'Usage: gpaw sbatch [-0] [-a args] script.py -- [sbatch options]')
        parser.add_argument('script')
        parser.add_argument('-0', '--dry-run', action='store_true')
        parser.add_argument('-a', '--script-options', default='')
        parser.add_argument('sbatch_options', nargs='*')

    @staticmethod
    def run(args):
        script = '#!/bin/bash -l\n'
        for line in open(args.script):
            if line.startswith('#SBATCH'):
                script += line
        script += ('OMP_NUM_THREADS=1 '
                   'mpiexec `echo $GPAW_MPI_OPTIONS` gpaw-python {} {}\n'
                   .format(args.script, args.script_options))
        cmd = ['sbatch'] + args.sbatch_options
        if args.dry_run:
            print('sbatch command:')
            print(' '.join(cmd))
            print('\nscript:')
            print(script, end='')
        else:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            p.communicate(script.encode())
