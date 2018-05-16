from myqueue.job import Job


def workflow():
    return [
        task('bandstructure.py@1:5m')]
