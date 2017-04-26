class CLICommand:
    short_description = 'Write summary of GPAW-restart file'

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('gpw', metavar='gpw-file')

    @staticmethod
    def run(args):
        from gpaw import GPAW
        GPAW(args.gpw)
