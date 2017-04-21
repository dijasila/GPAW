from __future__ import print_function
import subprocess


class CLICommand:
    short_description = 'Submit a GPAW Python script via sbatch.'

    @staticmethod
    def add_arguments(parser):
        parser.usage = 'Usage: gpaw sbatch [-0] script.py -- [sbatch options]'
        parser.add_argument('script')
        parser.add_argument('-0', '--dry-run', action='store_true')
        parser.add_argument('arguments', nargs='*')

    @staticmethod
    def run(args):
        script = '#!/bin/bash -l\n'
        for line in open(args.script):
            if line.startswith('#SBATCH'):
                script += line
        script += ('OMP_NUM_THREADS=1 '
                   'mpiexec `echo $GPAW_MPI_OPTIONS` gpaw-python {}\n'
                   .format(args.script))
        cmd = ['sbatch'] + args.arguments
        if args.dry_run:
            print(script, end='')
        else:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            p.communicate(script.encode())
