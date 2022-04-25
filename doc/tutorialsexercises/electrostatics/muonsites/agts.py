from myqueue.workflow import run


def workflow():
    with run(script='mnsi.py'):
        run(script='plot2d.py')
