from myqueue.workflow import run


def workflow():
    with run(script='calculate.py', tmax='1h'):
        run(script='plot.py')
