# Creates: N2Ru_hollow.png, 2NadsRu.png
from myqueue.task import task


def create_tasks():
    nbrun = 'gpaw.utilities.nbrun'
    return [
        task('run.py', tmax='10h'),
        task(nbrun, args=['convergence.ipynb'], deps='run.py'),
        task(nbrun, args=['n2_on_metal.master.ipynb'], tmax='3h'),
        # task(nbrun, args=['neb.master.ipynb'], tmax='3h'),
        task('ts.py'),
        # task(nbrun, args=['vibrations.master.ipynb'], tmax='3h',
        #      deps='ts.py')
    ]
