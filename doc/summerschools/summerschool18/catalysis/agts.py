# Creates: N2Ru_hollow.png, 2NadsRu.png, TS.xyz
from myqueue.task import task


def create_tasks():
    return [
        task('check_convergence.py', tmax='5h', cores=8),
        task('convergence.py', deps='check_convergence.py'),
        task('n2_on_metal.py', tmax='6h'),
        task('neb.py', tmax='3h', cores=8, deps='n2_on_metal.py'),
        task('vibrations.py', tmax='9h', cores=8, deps='neb.py')]
