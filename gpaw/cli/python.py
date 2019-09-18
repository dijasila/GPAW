import runpy
import sys


class CLICommand:
    """Run GPAW's Python interpreter."""

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--module', '-m',
                            help='run library module given as SCRIPT')
        parser.add_argument('arguments', nargs='*')

    @staticmethod
    def run(args):
        sys.argv[:] = [args.module] + args.arguments
        runpy.run_module(args.module, run_name='__main__')
