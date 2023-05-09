from myqueue.workflow import run


def workflow():
    with run(script='ethanol_in_water.py', cores=4):
        run(script='check.py')
