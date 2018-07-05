from pathlib import Path
from myqueue.task import task


def create_tasks():
    return [
        task('run.py', tmax='5h'),
        task('nbrun.py', args=['convergence.ipynb'], tmax='3h',
             deps='run.py'),
        task('nbrun.py', args=['n2_on_metal.master.ipynb'], tmax='3h'),
        task('nbrun.py', args=['neb.master.ipynb'], tmax='3h'),
        task('ts.py'),
        task('nbrun.py', args=['vibrations.master.ipynb'], tmax='3h',
             deps='ts.py')]
