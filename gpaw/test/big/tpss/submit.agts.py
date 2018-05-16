from myqueue.task import task


def workflow():
    return [task('tpss.py@8:1h')]
