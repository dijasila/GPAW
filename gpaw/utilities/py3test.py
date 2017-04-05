"""Test GPAW.

Initial setup::

    cd ~
    python3 -m venv gpaw-tests
    cd gpaw-tests
    . bin/activate
    pip install scipy
    git clone http://gitlab.com/ase/ase.git
    cd ase
    pip install -U .
    cd ..
    git clone http://gitlab.com/gpaw/gpaw.git
    cd gpaw
    python setup.py install

Crontab::

    WEB_PAGE_FOLDER=...
    CMD="python -m gpaw.utilities.build_web_page"
    10 20 * * * cd ~/gpaw-tests; . bin/activate; cd gpaw; $CMD > ../gpaw.log

"""
from __future__ import print_function
import os
import subprocess
import sys


cmds = """\
touch gpaw-tests.lock
cd ase; git pull; pip install -U .
cd gpaw; git clean -fdx; git pull; python setup.py install
gpaw -P 1 test"""


def build():
    if os.path.isfile('../gpaw-tests.lock'):
        print('Locked', file=sys.stderr)
        return
    try:
        for cmd in cmds.splitlines():
            subprocess.check_call(cmd, shell=True)
    finally:
        os.remove('../gpaw-tests.lock')


if __name__ == '__main__':
    build()
