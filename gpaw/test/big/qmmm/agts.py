from myqueue.task import task


def workflow():
    return [task('qmmm.py@8:15m')]
