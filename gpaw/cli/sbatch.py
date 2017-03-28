import subprocess


class CLICommand:
    short_description = 'Submit a GPAW Python script via sbatch.'
    description = ('Usage: gpaw sbatch [sbatch options] script.py '
                   '[script arguments]')

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('arguments', nargs='+')

    @staticmethod
    def run(args):
        for i, arg in enumerate(args.arguments):
            if arg.endswith('.py'):
                break
        else:
            return

        script = '#!/bin/sh\n'
        for line in open(arg):
            if line.startswith('#SBATCH'):
                script += line
        script += ('OMP_NUM_THREADS=1 '
                   'mpiexec gpaw-python ' +
                   ' '.join(args.arguments[i:]) + '\n')
        cmd = ['sbatch'] + args.argument[:i]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        p.communicate(script.encode())
