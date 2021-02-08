from myqueue.workflow import run


def workflow():
    return [
        task('atomize.py@1:30m'),
        task('relax.py@1:30m', deps='atomize.py')]
