from myqueue.job import Job


def workflow():
    return [
        Job('bandstructure.py@1:5m')]
