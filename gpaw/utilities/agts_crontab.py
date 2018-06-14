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

import datetime
import functools
import json
import os
import subprocess
from pathlib import Path


shell = functools.partial(subprocess.check_call, shell=True)


def agts(cmd):
    from myqueue.task import taskstates
    from myqueue.tasks import Tasks, Selection

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
        arch = 'linux-x86_64-broadwell-el7-3.6'
        path = '/home/niflheim2/jensj/agts'
        pp = ('{path}/ase:{path}/gpaw:{path}/gpaw/build/lib.{arch}'
              .format(path=path, arch=arch))
        shell('PYTHONPATH={pp}:$PYTHONPATH mq workflow -p agts.py gpaw -T'
              .format(pp=pp))

    elif cmd == 'summary':
        write_results(tasks)

        for task in tasks:
            if task.state in {'running', 'queued'}:
                raise RuntimeError('Not done!')

        for task in tasks:
            if task.state in {'FAILED', 'CANCELED', 'TIMEOUT'}:
                send_email(tasks)
                return

        shell('mq delete --state d --recursive .')

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


def find_created_files():
    names = set()
    for path in Path().glob('**/*.py'):
        if path.parts[0] == 'build':
            continue
        line1 = path.read_text().split('\n', 1)[0]
        if not line1.startswith('# Creates:'):
            continue
        for name in line1.split(':')[1].split(','):
            name = name.strip()
            if name in names:
                raise RuntimeError(
                    'The name {!r} is used in more than one place!'
                    .format(name))
            names.add(name)
            yield path.with_name(name)


def collect_files_for_web_page():
    os.chdir('gpaw/doc')
    folder = Path('agts-files')
    if not folder.is_dir():
        folder.mkdir()
    for path in find_created_files():
        print(path)
        (folder / path.name).write_bytes(path.read_bytes())
    # os.environ['WEB_PAGE_FOLDER']


def write_results(tasks):
    now = datetime.datetime.now()
    name = Path('{}-{:02}-{:02}.json'.format(now.year, now.month, now.day))
    text = json.dumps({'tasks': [task.todict() for task in tasks]},
                      indent=2)
    name.write_text(text)


if __name__ == '__main__':
    import sys
    agts(sys.argv[1])
