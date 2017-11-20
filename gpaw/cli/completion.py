import sys

from ase.cli.completion import update, CLICommand

from gpaw.cli.main import commands


CLICommand.cmd = ('complete -o default -C "{} -m gpaw.cli.complete" gpaw'
                  .format(sys.executable))


if __name__ == '__main__':
    # Path of the complete.py script:
    filename = __file__.rsplit('/', 1)[0] + '/complete.py'
    update(filename, commands)
