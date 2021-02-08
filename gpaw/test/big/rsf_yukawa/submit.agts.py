from myqueue.workflow import run


def workflow():
    return [task('lrtddft.py@4:1m')]
