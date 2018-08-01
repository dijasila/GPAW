# Creates: N2Ru_hollow.png, 2NadsRu.png, ts.traj
from myqueue.task import task


def create_tasks():
    nbrun = 'gpaw.utilities.nbrun'
    return [
        task('run.py', tmax='5h', cores=8),
        task(nbrun, args=['convergence.ipynb'], deps='run.py'),
        task(nbrun, args=['n2_on_metal.master.ipynb'], tmax='3h'),
        task(nbrun, args=['neb.master.ipynb'], tmax='3h', cores=8,
             deps=nbrun + '+n2_on_metal.master.ipynb'),
        task(nbrun, args=['vibrations.master.ipynb'], tmax='9h', cores=8,
             deps=nbrun + '+neb.master.ipynb')]
