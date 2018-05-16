from myqueue.task import task


def workflow():
    return [
        task('bandstructure.py@1:5m')]
