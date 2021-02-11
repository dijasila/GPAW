from myqueue.workflow import run


def workflow():
    with run(script='diffusion1.py', cores=4):
        with run(script='neb.py', cores=6, tmax='1h'):
            run(script='check.py')
