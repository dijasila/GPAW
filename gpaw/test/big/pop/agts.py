from myqueue.workflow import run


def workflow():
    run(script='mos2.py', cores=24, tmax='1h')
