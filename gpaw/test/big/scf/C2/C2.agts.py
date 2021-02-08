from myqueue.workflow import run


def workflow():
    run(script='C2.py', cores=4, tmax='1h')
