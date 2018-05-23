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


#os.environ['WEB_PAGE_FOLDER']

shell = functools.partial(subprocess.check_call, shell=True)


def agts(cmd):
    allofthem = Selection(None, '', taskstates, [Path('.').absolute()], True)
    with Tasks(verbosity=-1) as t:
        tasks = t.list(allofthem, '')

    print(len(tasks))

    if cmd == 'run':
        if tasks:
            raise ValueError('Not ready!')

        shell('cd ase; git pull')
        shell('cd gpaw; git clean -fdx; git pull;'
              '. doc/platforms/Linux/Niflheim/compile.sh')
        # shell('mq workflow -p agts.py gpaw')
        shell('mq workflow -p agts.py gpaw/doc/devel/ase_optimize -T')

    elif cmd == 'summary':
        for task in tasks:
            if task.state in {'running', 'queued'}:
                raise RuntimeError('Not done!')

        for task in tasks:
            if task.state in {'FAILED', 'CANCELED', 'TIMEOUT'}:
                send_email(tasks)
                return

        collect_files_for_web_page()

    else:
        1 / 0


def send_email(tasks):
    import smtplib
    from email.message import EmailMessage

    txt = 'Hi!\n\n'
    for task in tasks:
        if task.state in {'FAILED', 'CANCELED', 'TIMEOUT'}:
            id, dir, name, res, age, status, t, err = task.words()
            txt += ('test: {}/{}@{}: {}\ntime: {}\nerror: {}\n\n'
                    .format(dir.split('agts/gpaw')[1],
                            name,
                            res[:-1],
                            status,
                            t,
                            err))
    txt += 'Best regards,\nNiflheim\n'

    msg = EmailMessage()
    msg.set_content(txt)
    msg['Subject'] = 'Failing Niflheim-tests!'
    msg['From'] = 'agts@niflheim.dtu.dk'
    msg['To'] = 'jjmo@dtu.dk'
    s = smtplib.SMTP('smtp.ait.dtu.dk')
    s.send_message(msg)
    s.quit()


def collect_files_for_web_page():

if __name__ == '__main__':
    import sys
    agts(sys.argv[1])
