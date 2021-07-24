from myqueue.workflow import run


def workflow():
    with run(script='run-sjm.py', cores=72, tmax='4h'):
        run(script='make-plot.py')
