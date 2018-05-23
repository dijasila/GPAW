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
from pathlib import Path

from myqueue.tasks import Tasks, Selection
from myqueue.task import taskstates


os.environ['WEB_PAGE_FOLDER']

shell = functools.partial(subprocess.check_call, shell=True)


def agts(cmd):
    allofthem = Selection(None, '', taskstates, [Path('.').absolute()], True)
    with Tasks(verbosity=-1) as t:
        tasks = t.list(allofthem, '')

    if cmd == 'run':
        if tasks:
            raise ValueError('Not ready!')

        # shell('cd ase; git pull')
        # shell('cd gpaw; git clean -fdx; git pull;'
        #       '. doc/platforms/Linux/Niflheim/compile.sh')
        # shell('mq workflow -p agts.py gpaw')
        shell('mq workflow -p agts.py gpaw/doc/devel/ase_optimize -T')

    elif cmd == 'summary':
        for task in tasks:
            if task.state in {'running', 'queued'}:
                raise ValueError('Not done!')

        for task in tasks:
            if task.state in {'FAILED', 'CANCELED', 'TIMEOUT'}:
                send_email(tasks)
                return

        print('ok')

    else:
        1 / 0


def send_email(tasks):
    pass


if __name__ == '__main__':
    import sys
    import os
    print(sys.path, os.environ['PATH'])
    agts(sys.argv[1])
