from myqueue.task import task


def workflow():
    return [
        task('nio.py'),
        task('n.py'),
        task('check.py', deps='n.py')]
