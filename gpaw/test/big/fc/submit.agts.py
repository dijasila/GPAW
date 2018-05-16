from myqueue.task import task


def workflow():
    return [task('fc_butadiene.py@1:30m')]
