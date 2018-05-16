from myqueue.task import task


def workflow():
    return [task('C2.py@4:1h')]
