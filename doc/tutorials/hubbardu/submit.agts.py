from myqueue.workflow import run


def workflow():
    return [
        task('nio.py'),
        task('n.py'),
        task('check.py', deps='n.py')]
