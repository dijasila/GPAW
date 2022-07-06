from myqueue.workflow import run


def workflow():
    run(script='lcaotddft.py', cores=4, tmax='40m')
