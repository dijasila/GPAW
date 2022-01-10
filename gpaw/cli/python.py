import runpy
import sys


class CLICommand:
    """Run GPAW's Python interpreter."""

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--dry-run', '-z', type=int, default=0,
                            metavar='NCPUS',
                            help='Dry run on NCPUS cpus.')
        parser.add_argument('-d', '--debug', action='store_true',
                            help='Run in debug-mode.')
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--command', '-c', dest='cmd',
                           help='Program passed in as string.')
        group.add_argument('--module', '-m',
                           help='Run library module as a script.')
        parser.add_argument('arguments', metavar='ARG',
                            help='Arguments passed to program in '
                            'sys.argv[1:].',
                            nargs=-1)

    @staticmethod
    def run(args):
        if args.cmd:
            sys.argv[:] = ['-c'] + args.arguments
            d = {}
            exec(args.cmd, d, d)
        elif args.module:
            sys.argv[:] = [args.module] + args.arguments
            runpy.run_module(args.module, run_name='__main__', alter_sys=True)
        else:
            sys.argv[:] = args.arguments
            runpy.run_path(args.arguments[0], run_name='__main__')
