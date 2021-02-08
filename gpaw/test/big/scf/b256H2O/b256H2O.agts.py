from myqueue.workflow import run


def workflow():
    run(script='b256H2O.py', cores=8, tmax='5h')
