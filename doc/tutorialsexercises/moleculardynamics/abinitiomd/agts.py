from myqueue.workflow import run


def workflow():
    run(script='atomtransfer.py.py', cores=40, tmax='1h')
