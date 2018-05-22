"""Run AGTS tests.

Initial setup::

    cd ~
    python3 -m venv agts
    cd agts
    . bin/activate
    pip install matplotlib scipy
    git clone http://gitlab.com/ase/ase.git
    git clone http://gitlab.com/gpaw/gpaw.git
    cd ase
    pip install -e .
    cd ..
    cd gpaw
    python setup.py install

Crontab::

    WEB_PAGE_FOLDER=...
    CMD="python -m gpaw.utilities.build_web_page"
    10 20 * * * cd ~/gpaw-web-page; . bin/activate; cd gpaw; $CMD > ../gpaw.log

"""
import functools
import os
import subprocess

from myqueue.tasks import Tasks


os.environ['WEB_PAGE_FOLDER']

shell = functools.partial(subprocess.check_call, shell=True)


def agts1():
    with Tasks(verbosity=-1) as t:
        tasks = t.list()
    if tasks:
        raise ValueError('Not done yet')
    shell('cd ase; git pull')
    shell('cd gpaw; git clean -fdx; git pull;'
          '. doc/platforms/Linux/Niflheim/compile.sh')
    shell('mq workflow -p agts.py doc gpaw')


if __name__ == '__main__':
    agts1()
