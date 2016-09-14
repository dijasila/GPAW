"""Crontab job for testing GPAW and ASE on Python 3.

Put this in crontab::

    ASE=$HOME//python3-test-gpaw/ase
    58 13 * * * cd $ASE/..; PATH=$ASE/tools:$PATH PYTHONPATH=$ASE:$PYTHONPATH \
                python3 -bb test.py --debug
"""
import os
import shutil
import subprocess
import sys
import os.path as op

dir = op.expanduser('~/python3-test-gpaw')
datasets = op.expanduser('~/datasets/gpaw-setups-0.9.11271')


def run():
    subprocess.check_call('cd ase; git pull > git.out', shell=True)
    subprocess.check_call('cd gpaw; git pull > git.out', shell=True)
    os.chdir('gpaw')
    sys.path[:0] = [op.join(dir, 'ase'), op.join(dir, 'lib', 'python')]
    subprocess.check_call('python3 setup.py install --home=.. > build.out',
                          shell=True)

    import numpy as np
    from gpaw import setup_paths
    setup_paths.insert(0, datasets)
    from gpaw.test import TestRunner, tests

    with open('test.out', 'w') as fd:
        os.mkdir('testing')
        os.chdir('testing')
        failed = TestRunner(tests[::-1], fd, jobs=4, show_output=False).run()
        os.chdir('..')

    from ase.test import test
    results = test(display=False)
    failed.extend(results.errors + results.failures)

    if failed:
        print(failed)
    else:
        shutil.rmtree('testing')

    os.chdir('..')
    shutil.rmtree('lib')
    shutil.rmtree('bin')

if __name__ == '__main__':
    if os.path.isdir('gpaw/testing'):
        print('Failed ...')
    else:
        run()
