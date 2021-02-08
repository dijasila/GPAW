from myqueue.workflow import run


def workflow():
    return [
        task('molecules.py', tmax='1h'),
        task('check.py', deps='molecules.py')]
