"""Build GPAW's web-page."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


cmds = """\
python3 -m venv venv
. venv/bin/activate
pip install sphinx-rtd-theme pillow
pip install git+https://gitlab.com/ase/ase.git@master
git clone http://gitlab.com/gpaw/gpaw.git
cd gpaw
git checkout {branch}
pip install .
python setup.py sdist
cd doc
make
mv build/html gpaw-web-page"""


def build(branch='master'):
    root = Path(f'/tmp/gpaw-docs-{branch}')
    if root.is_dir():
        sys.exit('Locked')
    root.mkdir()
    os.chdir(root)
    cmds2 = ' && '.join(cmds.format(branch=branch).splitlines())
    p = subprocess.run(cmds2, shell=True)
    if p.returncode == 0:
        status = 'ok'
    else:
        print('FAILED!', file=sys.stdout)
        status = 'error'
    f = root.with_name(f'gpaw-docs-{branch}-{status}')
    if f.is_dir():
        shutil.rmtree(f)
    root.rename(f)
    return status


def build_both():
    assert build('master') == 'ok'
    assert build('web-page') == 'ok'
    tar = next(
        Path('/tmp/gpaw-docs-master-ok/gpaw/dist/').glob('gpaw-*.tar.gz'))
    master = Path('/tmp/gpaw-docs-master-ok/gpaw/doc/gpaw-web-page')
    webpage = Path('/tmp/gpaw-docs-web-page-ok/gpaw/doc/gpaw-web-page')
    home = Path.home() / 'web-pages'
    cmds = ' && '.join(
        [f'cp -rp {master} {webpage}/dev',
         f'cp {tar} {webpage}',
         f'cp {tar} {webpage}/dev',
         f'find {webpage} -name install.html | '
         f'xargs sed -i s/snapshot.tar.gz/{tar.name}/g',
         f'cd {webpage}/_sources/setups',  # backwards compatibility
         'cp setups.rst.txt setups.txt',  # with old install-data script
         f'cd {webpage.parent}',
         f'tar -czf gpaw-web-page.tar.gz gpaw-web-page',
         f'cp gpaw-web-page.tar.gz {home}/gpaw.test.tgz'])
    subprocess.run(cmds, shell=True, check=True)


if __name__ == '__main__':
    # build()
    build_both()
