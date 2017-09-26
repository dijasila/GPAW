import os

from gpaw import archive


description = """Archive gpw-file in ARC/<id>.gpw and add row to database
in ARC/gpw.db@id=<id> where <id> is a uniqe integer id.  The folder ARC
defaults to ~/.gpaw/archive/ or $GPAW_ARCHIVE_FOLDER.  Wave functions
will be stripped from the copy of gpw-file."""


class CLICommand:
    short_description = 'Archive gpw-file in central repository'
    description = description

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('filename', help='GPW-file.')
        add('-k', '--key-value-pairs', metavar='key=value,...', default='',
            help='Key-value pair(s) to add to database row.')

    @staticmethod
    def run(args):
        kvp = {}
        if args.key_value_pairs:
            for kv in args.key_value_pairs.split(','):
                key, value = kv.split('=')
                kvp[key] = value

        id, folder = archive(args.filename, None, kvp)

        print('ID: ', id)
        print('GPW:', os.path.join(folder, '{}.gpw'.format(id)))
        print('DB: ', os.path.join(folder, 'gpw.db@id={}'.format(id)))
