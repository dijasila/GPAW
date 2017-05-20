from ase.cli.completion import update, CLICommand

from gpaw.cli.main import commands


# Path of the complete.py script:
filename = __file__.rsplit('/', 1)[0] + '/complete.py'

CLICommand.cmd = 'complete -o default -C {} gpaw'.format(filename)


if __name__ == '__main__':
    update(filename, commands)
