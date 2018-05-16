from myqueue.task import task


def workflow():
    return [
        task('NaCl.py@1:30m')]
