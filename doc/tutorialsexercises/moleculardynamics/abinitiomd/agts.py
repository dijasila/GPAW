from myqueue.workflow import run


def workflow():
    run(script='atomtransmission.py', cores=40, tmax='1h')
