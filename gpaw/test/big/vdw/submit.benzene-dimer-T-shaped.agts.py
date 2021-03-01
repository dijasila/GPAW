from myqueue.workflow import run


def workflow():
    run(script='benzene-dimer-T-shaped.py', cores=48, tmax='20h')
