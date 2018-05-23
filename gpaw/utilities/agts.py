"""Run AGTS tests.

Initial setup::

    mkdir agts
    cd agts
    git clone http://gitlab.com/ase/ase.git
    git clone http://gitlab.com/gpaw/gpaw.git

Crontab::

    WEB_PAGE_FOLDER=...
    AGTS=...
    CMD="python $AGTS/gpaw/utilities/agts.py"
    10 20 * * 5 cd $AGTS; $CMD run > agts-run.log
    10 20 * * 1 cd $AGTS; $CMD summary > agts-summary.log

"""
import functools
import os
import subprocess

from myqueue.tasks import Tasks


os.environ['WEB_PAGE_FOLDER']

shell = functools.partial(subprocess.check_call, shell=True)


def agts(cmd):
    with Tasks(verbosity=-1) as t:
        tasks = t.list()

    if cmd == 'run':
        if tasks:
            raise ValueError('Not ready!')

        shell('cd ase; git pull')
        shell('cd gpaw; git clean -fdx; git pull;'
              '. doc/platforms/Linux/Niflheim/compile.sh')
        shell('mq workflow -p agts.py doc gpaw')
        shell('mq workflow -p agts.py doc/devel/ase_optimize')

    elif cmd == 'summary':
        for task in tasks:
            if task.state in {'running', 'queued'}:
                raise ValueError('Not done!')

        for task in tasks:
            if task.state in {'FAILED', 'CANCELED', 'TIMEOUT'}:
                send_email()
                return

        print('ok')

    else:
        1 / 0


if __name__ == '__main__':
    import sys
    agts(sys.argv[1])
