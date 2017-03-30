"""Build GPAW's web-page.

Initial setup::

    cd ~
    python3 -m venv gpaw-web-page
    cd gpaw-web-page
    . bin/activate
    pip install sphinx-rtd-theme
    pip install Sphinx
    pip install matplotlib scipy
    git clone git@gitlab.com:ase/ase
    cd ase
    pip install .
    git clone git@gitlab.com:gpaw/gpaw
    cd gpaw
    python setup.py install

Crontab::

    build="python -m gpaw.utilities.build_web_page"
    10 20 * * * cd ~/gpaw-web-page; . bin/activate; cd gpaw; $build > ../gpaw.log

"""

import os
import subprocess
import sys

from gpaw import __version__


cmds = """\
touch ../gpaw-web-page.lock
cd ../ase; git checkout web-page; pip install .
git clean -fdx
git checkout web-page
git pull
python setup.py install
cd doc; sphinx-build -b html -d build/doctrees . build/html
mv doc/build/html gpaw-web-page
cd ../ase; git checkout master; pip install .
git clean -fdx doc
rm -r build
git checkout master
git pull
python setup.py install
cd doc; sphinx-build -b html -d build/doctrees . build/html
mv doc/build/html gpaw-web-page/dev
python setup.py sdist
cp dist/gpaw-*.tar.gz gpaw-web-page/
cp dist/gpaw-*.tar.gz gpaw-web-page/dev/
find gpaw-web-page -name install.html | xargs sed -i s/snapshot.tar.gz/{}/g
tar -cf web-page.tar.gz gpaw-web-page""".format(
    'gpaw-' + __version__ + '.tar.gz')


def build():
    if os.path.isfile('../gpaw-web-page.lock'):
        print('Locked', file=sys.stderr)
        return
    try:
        for cmd in cmds.splitlines():
            subprocess.check_call(cmd, shell=True)
    finally:
        os.remove('../gpaw-web-page.lock')


if __name__ == '__main__':
    build()
