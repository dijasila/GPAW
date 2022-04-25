# web-page: N2Ru_hollow.png, 2NadsRu.png, TS.xyz
from myqueue.workflow import run


def workflow():
    with run(script='check_convergence.py', tmax='5h', cores=8):
        run(script='convergence.py')

    with run(script='n2_on_metal.py', tmax='6h'):
        with run(script='neb.py', tmax='3h', cores=8):
            run(script='vibrations.py', tmax='9h', cores=8)
