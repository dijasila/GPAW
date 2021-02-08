from myqueue.workflow import run


def workflow():
    return [task('g21gpaw.py@1:20h'),
            task('analyse.py', deps='g21gpaw.py')]
